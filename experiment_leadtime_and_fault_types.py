"""
Lead-time sweep x fault-type sweep for the mag_x-specific predictor
(2026-08-20). Supersedes eval_lstm_predictor_15s_magx_synthetic_faults.py,
fixing that script's two limitations and answering the question the
professor's "at least 15s" requirement actually poses.

What was wrong before:
  * Injection placement. A window's target time is at least
    context + horizon seconds into the flight, so injections placed earlier
    than that can never be any window's target. 5 of 10 planned injections
    were silently unreachable. Here every injection is placed after the
    largest (context + horizon) under test, so all of them count at every
    lead time.
  * Only one fault type. Reis & Reis sec 6.2 lists four -- additive noise
    burst, bias drift, signal dropout, step discontinuity -- and we had only
    implemented bias drift, so "recall 61.5%" said nothing about whether the
    model generalizes past the one signature it was tuned against.

What this adds:
  * All four injection types, reported separately, so a per-type recall
    breakdown replaces a single conflated number.
  * A LEAD-TIME SWEEP (5 / 10 / 15 / 20 / 30s). A model is retrained per
    horizon, because the horizon is baked into what the network learns to
    predict. This produces the curve that answers "15s is required -- is it
    achievable, and what does it cost?", rather than testing 15s alone and
    having no idea whether it sits on a cliff or a plateau.

Everything runs on pipeline v4 (per-flight normalization; PROJECT_SUMMARY
sec 3.9). Faults are injected only into the 4 files held out from
multisession training, and each model is trained on the other 13 + x70_all.

RESULT, AND THE PROBLEM IT EXPOSED
----------------------------------
Recall came out at 100% for bias_drift / noise_burst / step at EVERY horizon
from 5s to 30s, and 0% for dropout -- a perfectly flat lead-time curve. A
sweep that shows no degradation from 5s to 30s is not a good result, it is a
sign the sweep is not measuring what it claims to.

It isn't. The anomaly score is |prediction - actual| AT THE TARGET TIME, and
the injected fault lives in the target. So the error spikes the moment the
fault is observed, whatever the horizon -- the horizon only changes how
stale the context was, never when the alarm can fire. Direct check
(bias_drift at 80-83s, horizon 15s, threshold 1.020):

    target 79s  fault in neither        score 0.413   no alarm
    target 81s  fault in TARGET         score 6.403   ALARM
    target 96s  fault in context only   score 0.219   no alarm
    target 98s  fault in context only   score 0.224   no alarm

The alarm fires only once the fault reaches the target -- i.e. at fault
time, with ZERO advance warning. When the fault sits in the context but not
the target, the score is not merely below threshold, it is *lower* than the
no-fault baseline.

So this architecture does not satisfy "predict at least 15s ahead" in any
operational sense, and the 100% numbers above must not be reported as if it
does. Getting real lead time needs a different scoring rule -- e.g. judging
how anomalous the PREDICTED future state is against a model of normal,
rather than comparing the prediction to an actual you can only have once the
fault has already happened. That also requires the fault to be foreseeable
from the preceding 15s at all, which for an abrupt mag_x burst it may not
be. See PROJECT_SUMMARY sec 3.12.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_loader import StandardScaler
from gru_data_loader import read_raw_csv_with_timestamp
from networks.lstm_predictor import LSTMPredictor
from train_eval_lstm_predictor_15s import (
    CONTEXT_BINS, INPUT_DIM, EPOCHS, LEARNING_RATE, RANDOM_SEED,
    INK_PRIMARY, INK_SECONDARY, GRIDLINE, SURFACE,
    build_pairs, flight_to_binned,
)
from train_eval_lstm_predictor_15s_multisession import DUPLICATES, EVAL_ONLY

MAG_X = Config.CHANNEL_NAMES.index("mag_x")
MAG_X_COLS = slice(MAG_X * 4, MAG_X * 4 + 4)

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_leadtime_sweep")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")

HORIZONS = [5, 10, 15, 20, 30]           # seconds ahead
FAULT_AMPLITUDE_STD = 6.0                # comparable to the real fault's z=7.55
FAULT_DURATION = 3.0                     # seconds
# Placed past max(context + horizon) = 15 + 30 = 45s so every injection is
# reachable at EVERY horizon tested. Files shorter than ~50s can't host one.
INJECT_STARTS = [50.0, 65.0, 80.0, 95.0, 110.0]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------- injections
def _mask(real_time, start, duration):
    return (real_time >= start) & (real_time < start + duration)


def inject(raw, real_time, kind, start, duration, amplitude_std):
    """Reis & Reis sec 6.2's four types, applied to raw mag_x."""
    raw = raw.copy()
    col = raw[:, MAG_X]
    sd = col.std()
    amp = amplitude_std * sd
    m = _mask(real_time, start, duration)
    if not m.any():
        return raw, False

    if kind == "bias_drift":
        # triangular ramp up then back down -- calibration drift
        t = (real_time[m] - start) / duration
        col[m] += amp * np.where(t < 0.5, t * 2, (1 - t) * 2)
    elif kind == "noise_burst":
        rng = np.random.default_rng(RANDOM_SEED + int(start))
        col[m] += rng.normal(0.0, amp, size=int(m.sum()))
    elif kind == "dropout":
        # sensor disconnect: value freezes at the last pre-fault sample
        idx = np.flatnonzero(m)
        col[idx] = col[max(idx[0] - 1, 0)]
    elif kind == "step":
        # abrupt offset that persists for the fault's duration
        col[m] += amp
    else:
        raise ValueError(kind)
    raw[:, MAG_X] = col
    return raw, True


# ------------------------------------------------------------------ training
def train_for_horizon(horizon, train_binned, device):
    pasts, nexts = [], []
    for b in train_binned:
        p, n, _ = build_pairs(b, CONTEXT_BINS, horizon, stride=1)
        if len(p):
            pasts.append(p)
            nexts.append(n)
    X = np.concatenate(pasts, axis=0)
    Y = np.concatenate(nexts, axis=0)

    torch.manual_seed(RANDOM_SEED)
    model = LSTMPredictor(input_dim=INPUT_DIM, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                          num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossf = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)), batch_size=32, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model, len(X)


def magx_scores(model, binned, horizon, device, persistence=False):
    pasts, nexts, target_idx = build_pairs(binned, CONTEXT_BINS, horizon, stride=1)
    if len(pasts) == 0:
        return np.array([]), np.array([])
    pt = torch.tensor(pasts)
    if persistence:
        preds = pasts[:, -1, :]
    else:
        with torch.no_grad():
            preds = model(pt.to(device)).cpu().numpy()
    err = np.abs(preds[:, MAG_X_COLS] - nexts[:, MAG_X_COLS]).mean(axis=1)
    return err, target_idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]
    train_paths = [train_csv] + augment

    train_raw = [read_raw_csv_with_timestamp(f) for f in train_paths]
    train_binned_raw = [flight_to_binned(r, t) for r, t in train_raw]
    feature_scaler = StandardScaler()
    feature_scaler.fit(np.vstack(train_binned_raw))
    train_binned = [feature_scaler.transform(b) for b in train_binned_raw]

    # which (file, start) injection slots actually fit inside each eval file
    eval_raw = [(os.path.basename(f), *read_raw_csv_with_timestamp(f)) for f in eval_files]
    slots = [(name, s) for name, raw, t in eval_raw for s in INJECT_STARTS
             if s + FAULT_DURATION < t[-1]]
    print(f"eval files: {[n for n, _, _ in eval_raw]}")
    print(f"usable injection slots: {len(slots)} "
          f"({', '.join(sorted({n for n, _ in slots}))})")

    kinds = ["bias_drift", "noise_burst", "dropout", "step"]
    rows = []
    for horizon in HORIZONS:
        model, n_pairs = train_for_horizon(horizon, train_binned, device)
        print(f"\n=== horizon {horizon}s (context {CONTEXT_BINS}s, {n_pairs} training pairs) ===")

        # clean-run threshold: 99th pct of scores on the UNINJECTED eval files
        clean = []
        for name, raw, t in eval_raw:
            b = flight_to_binned(raw, t, feature_scaler=feature_scaler)
            s, _ = magx_scores(model, b, horizon, device)
            clean.append(s)
        clean_all = np.concatenate([c for c in clean if len(c)])
        threshold = float(np.percentile(clean_all, 99))
        fp_rate = float((clean_all > threshold).mean())

        for kind in kinds:
            hits = total = 0
            for name, start in slots:
                raw, t = next((r, tt) for n, r, tt in eval_raw if n == name)
                inj, ok = inject(raw, t, kind, start, FAULT_DURATION, FAULT_AMPLITUDE_STD)
                if not ok:
                    continue
                b = flight_to_binned(inj, t, feature_scaler=feature_scaler)
                s, tidx = magx_scores(model, b, horizon, device)
                if len(s) == 0:
                    continue
                m = (tidx < start + FAULT_DURATION) & (tidx + 1 > start)
                if not m.any():
                    continue
                total += 1
                if s[m].max() > threshold:
                    hits += 1
            recall = hits / total if total else float("nan")
            rows.append({"horizon_sec": horizon, "fault_type": kind, "n_events": total,
                         "hits": hits, "recall": recall, "threshold": threshold, "fp_rate": fp_rate})
            print(f"  {kind:12s} recall {hits}/{total} = {recall:6.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "leadtime_fault_type_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved results to {RESULTS_DIR}/")
    plot(rows, kinds, os.path.join(FIGURES_DIR, "leadtime_vs_fault_type.png"))
    print(f"saved figure to {FIGURES_DIR}/")


def plot(rows, kinds, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, kind in enumerate(kinds):
        pts = [(r["horizon_sec"], r["recall"]) for r in rows if r["fault_type"] == kind]
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=PALETTE[i], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=kind)
        ax.annotate(kind, (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 0),
                    fontsize=9, color=INK_SECONDARY, va="center")

    ax.axvline(15, color=INK_PRIMARY, linestyle="--", linewidth=1.0)
    ax.annotate("15s required", (15, 1.06), textcoords="offset points", xytext=(6, 0),
                fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("prediction lead time (real seconds ahead)")
    ax.set_ylabel("recall on injected mag_x faults")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(min(HORIZONS) - 2, max(HORIZONS) + 9)
    ax.set_xticks(HORIZONS)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    ax.set_title("Recall is flat across lead time -- because the alarm fires AT fault time, not before",
                 color=INK_PRIMARY, fontsize=11)
    ax.text(0.5, -0.34, "the score is |prediction - actual| at the TARGET, so the horizon changes only how "
                        "stale the context was, never when detection can happen",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
