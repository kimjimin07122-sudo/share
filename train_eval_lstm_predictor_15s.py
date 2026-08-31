"""
15-second-ahead LSTM predictor (2026-08-20, professor requirement: >=15s lead
time). The existing LSTM predictor (train_eval_lstm_predictor.py) forecasts
exactly 1 raw sample ahead, which in real time is ~1ms (train-rate files,
~991Hz) to ~4ms (test-rate files, ~250Hz) -- nowhere near 15s. Two structural
changes were needed, not a parameter tweak:

1. Work in REAL SECONDS, not raw rows. Raw sampling rate varies ~4x by file
   (see PROJECT_SUMMARY sec 3.4), so "predict 15s ahead" only means the same
   thing across files if horizon/context are defined in real time. Engineered
   features (raw/deriv/deriv2/energy, computed at full resolution with the
   real per-row dt as usual) are aggregated into fixed 1-second bins using
   each row's real elapsed time -- this also makes the aggregated series
   immune to the train/test sampling-rate mismatch as a side effect.

2. The training file (x70_all.csv) is only 131.9 REAL seconds long. The
   existing 85/15 held-out split leaves ~19.8s of held-out normal -- not
   enough to build even one (15s context + 15s horizon = 30s) evaluation
   window. So this script trains on the full file (no held-out split from
   it) and instead calibrates a threshold from a POOLED distribution of
   window scores across all 19 test files (excluding the one real labeled
   burst window we know the true location of, see below) -- much larger
   effective sample than any single file's own 19.8s.

Ground truth for the only fault whose real-time location we've precisely
verified (x70_20180611_x76_070620.csv): real burst is [42.19, 42.99]s in a
45.71s file (NOT "1043-1063s" -- that was a stale pseudo-time-axis artifact
of the old channel-diagnosis code using row_idx/Config.DATA_HZ; see chat).
Evaluation here checks whether a window whose *target* (15s-ahead) time
falls inside that interval scores above the pooled-normal threshold, i.e.
whether the model would have raised an alarm 15s before the fault, using
only data available at context-end time (no lookahead).

Also runs a trivial PERSISTENCE baseline (predicted next state = last
context bin, i.e. "assume no change in 15s") alongside the LSTM, given the
very small training set (~100 pairs from one 132s flight) -- if the LSTM
can't beat "assume nothing changes," that's an important negative result on
its own.
"""
import csv
import glob
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_loader import build_scaler
from gru_data_loader import read_raw_csv_with_timestamp
from feature_engineering import compute_sequence_features, compute_window_summary
from networks.lstm_predictor import LSTMPredictor
from data_loader import StandardScaler

BIN_SIZE_SEC = 1.0
CONTEXT_BINS = 15   # 15s of history
HORIZON_BINS = 15   # predict exactly 15s ahead (professor's minimum)
MIN_FILE_SEC = CONTEXT_BINS + HORIZON_BINS  # files shorter than this yield zero windows

INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL
EPOCHS = 200          # tiny dataset (~100 pairs) needs many more passes than the 30 used elsewhere
LEARNING_RATE = 1e-3
RANDOM_SEED = Config.RANDOM_SEED

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_lstm_predictor_15s")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_predictor_15s_model.pth")
IFOREST_PATH = os.path.join(Config.BASE_DIR, "iforest_predictor_15s.joblib")

# Real-time location of the one fault we've precisely verified (chat, 2026-08-20).
BURST_FILE = "x70_20180611_x76_070620.csv"
BURST_REAL_SEC = (42.19, 42.99)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def bin_by_real_second(feat_seq, real_time_sec, bin_size=BIN_SIZE_SEC):
    """(T, D) engineered features + per-row real elapsed time -> (n_bins, D)
    per-bin means, one bin per `bin_size` real seconds. Bin i's mean value
    represents real time [i*bin_size, (i+1)*bin_size)."""
    n_bins = int(np.floor(real_time_sec[-1] / bin_size))
    if n_bins < 1:
        return np.zeros((0, feat_seq.shape[1]), dtype=np.float32)
    bin_idx = np.minimum((real_time_sec / bin_size).astype(int), n_bins - 1)
    binned = np.zeros((n_bins, feat_seq.shape[1]), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(binned, bin_idx, feat_seq)
    np.add.at(counts, bin_idx, 1)
    counts = np.maximum(counts, 1)
    return (binned / counts[:, None]).astype(np.float32)


def build_pairs(binned, context_bins, horizon_bins, stride=1):
    """(n_bins, D) -> (past[N,context_bins,D], next[N,D], target_bin_idx[N])."""
    pasts, nexts, target_idx = [], [], []
    last_start = len(binned) - context_bins - horizon_bins + 1
    for i in range(0, max(last_start, 0), stride):
        pasts.append(binned[i:i + context_bins])
        t_idx = i + context_bins - 1 + horizon_bins
        nexts.append(binned[t_idx])
        target_idx.append(t_idx)
    if not pasts:
        return np.zeros((0, context_bins, binned.shape[1]), dtype=np.float32), \
               np.zeros((0, binned.shape[1]), dtype=np.float32), np.array([], dtype=int)
    return np.asarray(pasts, dtype=np.float32), np.asarray(nexts, dtype=np.float32), np.asarray(target_idx)


def flight_to_binned(raw, real_time, feature_scaler=None, scaler=None, warmup_sec=None):
    """raw channels -> per-second bins of engineered features.

    PIPELINE v4 (2026-08-20): `scaler=None` fits a fresh scaler on THIS
    flight -- per-flight normalization, per Reis & Reis sec 6.1. Before that
    fix a single training-flight scaler was applied to every sortie, and
    since `energy = raw^2` squares any session offset, test features landed
    hundreds of sigma out of distribution (PROJECT_SUMMARY sec 3.9).

    CAUSAL VARIANT (v4.1, 2026-08-20): pass `warmup_sec` to fit that scaler
    on only the first `warmup_sec` seconds instead of the whole flight.
    Whole-flight statistics are NOT causal -- injecting a fault at t=100s
    measurably changes the normalized values at t=10s (mean |diff| 0.358),
    which manufactured 30+ seconds of fake "lead time" for faults that have
    no precursor at all (PROJECT_SUMMARY sec 3.13). It is also unusable
    onboard, where future samples do not exist yet. Warm-up mirrors the
    operational pattern: calibrate over the first seconds of flight, then
    monitor forward.

    The second-stage feature scaler is still needed on top: with real dt the
    derivative/energy features dwarf the raw standardized channels, and
    omitting it made MSE diverge to ~1e13 on the first attempt here."""
    if scaler is None:
        scaler = build_scaler()
        if warmup_sec is None:
            scaler.fit(raw)
        else:
            warm = raw[real_time <= warmup_sec]
            scaler.fit(warm if len(warm) > 10 else raw)
    feats = compute_sequence_features(scaler.transform(raw), real_time)
    binned = bin_by_real_second(feats, real_time)
    if feature_scaler is not None:
        binned = feature_scaler.transform(binned)
    return binned


def file_to_binned(filepath, scaler=None, feature_scaler=None):
    raw, real_time = read_raw_csv_with_timestamp(filepath)
    return flight_to_binned(raw, real_time, feature_scaler=feature_scaler, scaler=scaler)


def fused_score(err_vec, context_window, iforest, channel_mean, channel_std, if_mean, if_std):
    lstm_err = float(err_vec.mean())
    lstm_norm = (lstm_err - channel_mean.mean()) / channel_std.mean()
    if_features = np.concatenate([compute_window_summary(context_window), err_vec])[None, :]
    if_raw = -iforest.score_samples(if_features)[0]
    if_norm = (if_raw - if_mean) / if_std
    return Config.LSTM_AE_FUSION_ALPHA * lstm_norm + (1 - Config.LSTM_AE_FUSION_ALPHA) * if_norm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    print("device:", device)

    # --- train on the FULL x70_all.csv (no held-out split from it: at 15s
    # bins the file is only 131.9s -> a 15% held-out tail (19.8s) can't fit
    # even one 30s [context+horizon] window; see module docstring). ---
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    raw_train, time_train = read_raw_csv_with_timestamp(train_csv)
    binned_train_raw = flight_to_binned(raw_train, time_train)  # per-flight normalization
    print(f"train file: {time_train[-1]:.1f}s real duration -> {len(binned_train_raw)} one-second bins")

    # second-stage scaler on the engineered-feature space (see file_to_binned docstring)
    feature_scaler = StandardScaler()
    binned_train = feature_scaler.fit_transform(binned_train_raw)

    train_pasts, train_nexts, _ = build_pairs(binned_train, CONTEXT_BINS, HORIZON_BINS, stride=1)
    print(f"training pairs: {len(train_pasts)} (context={CONTEXT_BINS}s, horizon={HORIZON_BINS}s)")
    if len(train_pasts) < 20:
        print("WARNING: fewer than 20 training pairs -- results below are exploratory at best.")

    # --- LSTM predictor (same architecture as the 1-step version) ---
    model = LSTMPredictor(input_dim=INPUT_DIM, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                           num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(train_pasts), torch.tensor(train_nexts)),
                         batch_size=min(32, max(len(train_pasts), 1)), shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch+1}/{EPOCHS}  mse={total/n:.5f}")
    torch.save(model.state_dict(), MODEL_PATH)

    def lstm_predict(pasts_t):
        model.eval()
        with torch.no_grad():
            return model(pasts_t.to(device)).cpu().numpy()

    def persistence_predict(pasts_t):
        return pasts_t[:, -1, :].numpy()  # "assume no change in 15s"

    # --- fit IF + stats per method on the training pairs ---
    def fit_stats(predict_fn):
        preds = predict_fn(torch.tensor(train_pasts))
        errs = np.abs(preds - train_nexts)
        if_features = np.stack([
            np.concatenate([compute_window_summary(w), e]) for w, e in zip(train_pasts, errs)
        ], axis=0)
        iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)
        iforest.fit(if_features)
        channel_mean, channel_std = errs.mean(axis=0), errs.std(axis=0)
        channel_std = np.where(channel_std == 0, 1e-6, channel_std)
        if_scores = -iforest.score_samples(if_features)
        if_mean, if_std = if_scores.mean(), if_scores.std() or 1e-6
        return iforest, channel_mean, channel_std, if_mean, if_std

    methods = {
        "lstm_15s": (lstm_predict, *fit_stats(lstm_predict)),
        "persistence": (persistence_predict, *fit_stats(persistence_predict)),
    }
    joblib.dump(methods["lstm_15s"][1], IFOREST_PATH)

    # --- score every window in every test file (skip files too short for a 30s window) ---
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    results = {name: [] for name in methods}  # each: list of dict(file, target_sec, score, in_burst)

    for f in test_files:
        name = os.path.basename(f)
        binned = file_to_binned(f, feature_scaler=feature_scaler)
        if len(binned) < MIN_FILE_SEC:
            print(f"skip {name}: only {len(binned)}s of data, need >= {MIN_FILE_SEC}s for one window")
            continue
        pasts, nexts, target_idx = build_pairs(binned, CONTEXT_BINS, HORIZON_BINS, stride=1)
        pasts_t = torch.tensor(pasts)
        for method_name, (predict_fn, iforest, channel_mean, channel_std, if_mean, if_std) in methods.items():
            preds = predict_fn(pasts_t)
            errs = np.abs(preds - nexts)
            for w, e, t_idx in zip(pasts, errs, target_idx):
                score = fused_score(e, w, iforest, channel_mean, channel_std, if_mean, if_std)
                # bin t_idx covers real time [t_idx, t_idx+1) -- flag it if that
                # interval overlaps the (non-integer) verified burst interval at all
                in_burst = (name == BURST_FILE) and (t_idx < BURST_REAL_SEC[1]) and (t_idx + 1 > BURST_REAL_SEC[0])
                results[method_name].append({"file": name, "target_sec": int(t_idx), "score": score, "in_burst": in_burst})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary = {}
    for method_name, rows in results.items():
        normal_pool = [r["score"] for r in rows if not r["in_burst"]]
        burst_scores = [r["score"] for r in rows if r["in_burst"]]
        threshold = float(np.percentile(normal_pool, 99)) if normal_pool else 0.0
        hits = sum(1 for s in burst_scores if s > threshold)
        fp_rate = float(np.mean([s > threshold for s in normal_pool])) if normal_pool else float("nan")

        print(f"\n=== {method_name} ===")
        print(f"pooled 'normal-ish' windows: {len(normal_pool)}  (99th pct threshold = {threshold:.3f})")
        print(f"false-positive rate on that pool at this threshold: {fp_rate:.4f}")
        print(f"burst-window ({BURST_FILE} @ {BURST_REAL_SEC}s) scores: {[f'{s:.2f}' for s in burst_scores]}")
        print(f"burst windows caught: {hits}/{len(burst_scores)}")

        with open(os.path.join(RESULTS_DIR, f"{method_name}_scores.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "target_sec", "score", "in_burst"])
            writer.writeheader()
            writer.writerows(rows)

        summary[method_name] = {
            "threshold": threshold, "fp_rate": fp_rate,
            "hits": hits, "n_burst": len(burst_scores), "rows": rows,
        }

    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "threshold_99pct", "fp_rate", "burst_hits", "n_burst_windows"])
        for name, s in summary.items():
            writer.writerow([name, s["threshold"], s["fp_rate"], s["hits"], s["n_burst"]])
    print(f"\nsaved results to {RESULTS_DIR}/")

    plot_timeline(summary, os.path.join(FIGURES_DIR, "lstm_predictor_15s_burst_timeline.png"))
    plot_pool_comparison(summary, os.path.join(FIGURES_DIR, "lstm_predictor_15s_lstm_vs_persistence.png"))
    print(f"saved figures to {FIGURES_DIR}/")


def plot_timeline(summary, save_path):
    """Score vs real target-time for the one file with verified fault timing,
    LSTM vs persistence, with each method's pooled-normal threshold."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"lstm_15s": CAT_BLUE, "persistence": CAT_ORANGE}
    for ax, (name, s) in zip(axes, summary.items()):
        rows = [r for r in s["rows"] if r["file"] == BURST_FILE]
        rows.sort(key=lambda r: r["target_sec"])
        xs = [r["target_sec"] for r in rows]
        ys = [r["score"] for r in rows]
        in_burst = [r["in_burst"] for r in rows]
        ax.plot(xs, ys, color=colors[name], linewidth=1.6)
        ax.scatter([x for x, b in zip(xs, in_burst) if b], [y for y, b in zip(ys, in_burst) if b],
                   color=STATUS_CRITICAL, s=30, zorder=5, label="real fault window (target time)")
        ax.axhline(s["threshold"], color=INK_PRIMARY, linewidth=1.0, linestyle="--",
                   label=f"99th-pct pooled-normal threshold ({s['threshold']:.1f})")
        ax.axvspan(BURST_REAL_SEC[0], BURST_REAL_SEC[1], color=STATUS_CRITICAL, alpha=0.08)
        ax.set_title(f"{name}: forecast-error score vs. target time (15s-ahead)", color=INK_PRIMARY, loc="left", fontsize=11)
        ax.set_ylabel("fused score")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[-1].set_xlabel(f"target time (real seconds into {BURST_FILE}) -- context ends 15s earlier")
    fig.suptitle("15s-ahead forecast: does the alarm fire before the real fault (shaded, 42.19-42.99s)?",
                 color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_pool_comparison(summary, save_path):
    """Headline chart: LSTM vs persistence on the two numbers that matter --
    burst detection rate and false-positive rate on the pooled normal-ish
    windows from all 19 test files."""
    names = list(summary.keys())
    hit_rate = [summary[n]["hits"] / max(summary[n]["n_burst"], 1) for n in names]
    fp_rate = [summary[n]["fp_rate"] for n in names]

    x = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(names):
        color = CAT_BLUE if n == "lstm_15s" else CAT_ORANGE
        offset = (i - 0.5) * width
        vals = [hit_rate[i], fp_rate[i]]
        xs = x + offset
        ax.bar(xs, vals, width * 0.9, color=color, label=n)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(["burst hit rate\n(higher is better)", "false-positive rate\n(lower is better)"])
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=2, fontsize=9)
    ax.set_title("15s-ahead forecast: LSTM vs. persistence (\"assume no change\") baseline", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
