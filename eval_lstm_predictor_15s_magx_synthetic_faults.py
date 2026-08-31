"""
Synthetic bias-drift fault injection for the 15s-ahead mag_x-specific
predictor (2026-08-20, follow-up to eval_lstm_predictor_15s_magx_specific.py).

The mag_x-specific score correctly caught the one real verified fault
(42.19-42.99s in x70_20180611_x76_070620.csv), but n=1 isn't a statistically
meaningful evaluation. Reis & Reis 2025 (the architecture paper this project
reproduces) reports handling exactly this data-scarcity problem with
"controlled fault injection ... four main types: additive noise bursts,
bias drifts, signal dropouts, step discontinuities ... covering 3-10% of
each log duration" (sec 6.2). We implement the "bias drift" variant (a
triangular ramp added directly to raw mag_x), since it matches the known
fault signature here: the Ahn & Chung dataset paper's Fig. 8 shows a yaw
error that the control logic compensates for until it "became
uncontrollable" -- i.e. a growing magnetometer calibration-style bias, not
a sudden spike.

Injected ONLY into the 4 files held out from multisession training (never
seen during training, so no leakage), at multiple non-overlapping locations
per file, well clear of each file's start/end margins and (for the one file
with a real fault) clear of the real burst window.

CAVEAT (found after first run): a window's earliest possible target time is
context_bins + horizon_bins - 1 = 29s (needs 15s of context before it even
starts), so the first injection placed in each file (~16-20s) can never
become any window's target -- 5 of the 10 planned injections were silently
unreachable and contributed zero evaluation windows. Only 5 fault events
(4 synthetic + the 1 real one) actually landed in the evaluated range,
giving n=13 fault windows (each multi-second injection spans a few target
bins). Still a real improvement over n=1, just not the n=10 the injection
list implies -- INJECTIONS below is left as originally run for
reproducibility; move injections later than ~30s if you rerun this.

Reuses the already-trained multisession model -- no retraining, this only
changes what's fed into it at evaluation time.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data_loader import build_scaler, StandardScaler
from gru_data_loader import read_raw_csv_with_timestamp
from feature_engineering import compute_sequence_features
from networks.lstm_predictor import LSTMPredictor
from train_eval_lstm_predictor_15s import (
    CONTEXT_BINS, HORIZON_BINS, MIN_FILE_SEC, INPUT_DIM,
    INK_PRIMARY, INK_SECONDARY, GRIDLINE, STATUS_GOOD, STATUS_CRITICAL, CAT_BLUE, CAT_ORANGE,
    bin_by_real_second, build_pairs, flight_to_binned,
)
from train_eval_lstm_predictor_15s_multisession import DUPLICATES, EVAL_ONLY

MAG_X_CHANNEL_IDX = Config.CHANNEL_NAMES.index("mag_x")
MAG_X_COLS = slice(MAG_X_CHANNEL_IDX * 4, MAG_X_CHANNEL_IDX * 4 + 4)

MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_predictor_15s_multisession_model.pth")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")

BURST_FILE = "x70_20180611_x76_070620.csv"
REAL_FAULT_SEC = (42.19, 42.99)
DRIFT_PEAK_STD = 6.0   # comparable to the real fault's z=7.55 peak
DRIFT_DURATION = 3.0   # seconds, triangular ramp up/down

# (file, [(start_sec, duration_sec), ...]) -- spaced apart, clear of file
# edges (need >=CONTEXT_BINS of real data before each injection) and, for
# the burst file, clear of the real fault window.
INJECTIONS = {
    "x70_02.csv": [(20.0, DRIFT_DURATION)],
    "x70_20180601_x76_083916.csv": [(20.0, DRIFT_DURATION), (60.0, DRIFT_DURATION), (100.0, DRIFT_DURATION)],
    "x70_20180611_x66_082253.csv": [(20.0, DRIFT_DURATION), (55.0, DRIFT_DURATION), (95.0, DRIFT_DURATION)],
    BURST_FILE: [(16.0, DRIFT_DURATION), (26.0, DRIFT_DURATION)],
}


def triangular_bump(real_time, start, duration, peak):
    """0 -> peak over the first half, peak -> 0 over the second half."""
    t = real_time - start
    half = duration / 2
    bump = np.zeros_like(real_time)
    rising = (t >= 0) & (t < half)
    falling = (t >= half) & (t < duration)
    bump[rising] = peak * (t[rising] / half)
    bump[falling] = peak * (1 - (t[falling] - half) / half)
    return bump


def inject_magx_bias_drift(raw, real_time, injections):
    raw = raw.copy()
    mag_x_std = raw[:, MAG_X_CHANNEL_IDX].std()
    peak = DRIFT_PEAK_STD * mag_x_std
    windows = []
    for start, duration in injections:
        raw[:, MAG_X_CHANNEL_IDX] += triangular_bump(real_time, start, duration, peak)
        windows.append((start, start + duration))
    return raw, windows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment_files = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]
    train_file_paths = [train_csv] + augment_files

    # reproduce the multisession scaler + feature_scaler exactly (deterministic refit)
    # pipeline v4: per-flight normalization (see flight_to_binned). Still a
    # deterministic refit of the multisession run's feature scaler, so it
    # reproduces exactly given the same file list and order.
    raw_time_pairs = [read_raw_csv_with_timestamp(f) for f in train_file_paths]
    binned_per_file = [flight_to_binned(raw, rt) for raw, rt in raw_time_pairs]
    feature_scaler = StandardScaler()
    feature_scaler.fit(np.vstack(binned_per_file))

    model = LSTMPredictor(input_dim=INPUT_DIM, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                           num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    def lstm_predict(pasts_t):
        with torch.no_grad():
            return model(pasts_t.to(device)).cpu().numpy()

    def persistence_predict(pasts_t):
        return pasts_t[:, -1, :].numpy()

    methods = {"lstm_15s_magx": lstm_predict, "persistence_magx": persistence_predict}
    results = {name: [] for name in methods}  # rows: file, target_sec, score, is_fault

    print(f"{'file':40s} {'fault windows (real+synthetic)'}")
    for f in eval_files:
        name = os.path.basename(f)
        raw, real_time = read_raw_csv_with_timestamp(f)

        fault_windows = list(INJECTIONS.get(name, []))
        raw, injected_windows = inject_magx_bias_drift(raw, real_time, fault_windows)
        if name == BURST_FILE:
            fault_windows = injected_windows + [REAL_FAULT_SEC]
        else:
            fault_windows = injected_windows
        print(f"{name:40s} {fault_windows}")

        binned = flight_to_binned(raw, real_time, feature_scaler=feature_scaler)
        if len(binned) < MIN_FILE_SEC:
            continue
        pasts, nexts, target_idx = build_pairs(binned, CONTEXT_BINS, HORIZON_BINS, stride=1)
        pasts_t = torch.tensor(pasts)
        for method_name, predict_fn in methods.items():
            preds = predict_fn(pasts_t)
            err_magx = np.abs(preds[:, MAG_X_COLS] - nexts[:, MAG_X_COLS]).mean(axis=1)
            for score, t_idx in zip(err_magx, target_idx):
                is_fault = any(lo < t_idx + 1 and t_idx < hi for lo, hi in fault_windows)
                results[method_name].append({"file": name, "target_sec": int(t_idx),
                                              "score": float(score), "is_fault": is_fault})

    print(f"\n{'method':20s} {'n_normal':>9s} {'n_fault':>8s} {'thresh_99p':>11s} {'recall':>8s} {'fp_rate':>8s}")
    summary = {}
    for method_name, rows in results.items():
        normal_scores = [r["score"] for r in rows if not r["is_fault"]]
        fault_rows = [r for r in rows if r["is_fault"]]
        threshold = float(np.percentile(normal_scores, 99))
        hits = sum(1 for r in fault_rows if r["score"] > threshold)
        recall = hits / len(fault_rows) if fault_rows else float("nan")
        fp_rate = float(np.mean([s > threshold for s in normal_scores]))
        print(f"{method_name:20s} {len(normal_scores):9d} {len(fault_rows):8d} {threshold:11.4f} "
              f"{recall:8.2%} {fp_rate:8.4f}")
        summary[method_name] = {"threshold": threshold, "recall": recall, "hits": hits,
                                 "n_fault": len(fault_rows), "fp_rate": fp_rate, "rows": rows}

    plot_recall_comparison(summary, os.path.join(FIGURES_DIR, "lstm_predictor_15s_synthetic_fault_recall.png"))
    plot_example_timeline(summary, "x70_20180601_x76_083916.csv",
                           os.path.join(FIGURES_DIR, "lstm_predictor_15s_synthetic_fault_timeline.png"))
    print(f"\nsaved figures to {FIGURES_DIR}/")


def plot_recall_comparison(summary, save_path):
    names = list(summary.keys())
    x = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(names):
        color = CAT_BLUE if "lstm" in n else CAT_ORANGE
        offset = (i - 0.5) * width
        vals = [summary[n]["recall"], summary[n]["fp_rate"]]
        xs = x + offset
        ax.bar(xs, vals, width * 0.9, color=color, label=n)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    n_fault = summary[names[0]]["n_fault"]
    ax.set_xticklabels([f"recall (n={n_fault} windows, 5 fault events)", "false-positive rate"])
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=2, fontsize=9)
    ax.set_title("mag_x-specific 15s-ahead score: recall on real+synthetic faults (5 events)", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_example_timeline(summary, filename, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (name, s) in zip(axes, summary.items()):
        rows = [r for r in s["rows"] if r["file"] == filename]
        rows.sort(key=lambda r: r["target_sec"])
        xs = [r["target_sec"] for r in rows]
        ys = [r["score"] for r in rows]
        colors = [STATUS_CRITICAL if r["is_fault"] else (CAT_BLUE if "lstm" in name else CAT_ORANGE) for r in rows]
        ax.scatter(xs, ys, c=colors, s=14)
        ax.plot(xs, ys, color="#c3c2b7", linewidth=0.8, zorder=0)
        ax.axhline(s["threshold"], color=INK_PRIMARY, linewidth=1.0, linestyle="--",
                   label=f"99th-pct threshold ({s['threshold']:.2f})")
        ax.set_title(f"{name} on {filename} (3 synthetic bias-drift injections)", color=INK_PRIMARY, loc="left", fontsize=10)
        ax.set_ylabel("mag_x-only error")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[-1].set_xlabel("target time (real seconds) -- red dots = injected fault windows")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
