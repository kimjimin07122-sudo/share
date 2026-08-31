"""
Ensemble predictive uncertainty as a precursor signal (2026-08-20).

The last principled model class left to try without labels. Every previous
attempt scored a POINT forecast, and experiment_forecaster_comparison.py
showed why that stalls: a normal-trained LSTM regresses to the mean (0/4 on a
gradual ramp, losing outright to a training-free linear extrapolator at 4/4),
and the lead time the working methods do earn is matched by persistence --
which cannot anticipate by definition -- so it comes from threshold
sensitivity, not from prediction.

This scores the SPREAD instead of the point. Train N forecasters that differ
only in initialization and data order, then take their disagreement about
T+H as the signal. The reasoning (Precursor-of-Anomaly / FATE line of work):
while the input stays on the training manifold the members agree, and as it
starts to drift off-manifold they diverge -- and that divergence can begin
before the fault is large enough to be visible in the state itself. Crucially
this sidesteps mean-regression entirely: every member may predict something
normal-looking, but they need not predict the SAME normal-looking thing.

Same evaluation harness as before, so results are comparable: causal
normalization, attributable alarms (fires on the injected run but not the
clean run of the same flight), slow_ramp as the forecastable case and step as
the control that must show nothing.
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
from experiment_early_warning import (
    inject, HORIZON, FAULT_AMPLITUDE_STD, RAMP_SEC, SUSTAIN_SEC,
    VISIBLE_SIGMA, ONSET_STARTS, WARMUP_SEC,
)
from experiment_leadtime_and_fault_types import MAG_X_COLS

N_MEMBERS = 5
RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_ensemble_uncertainty")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def train_member(X, Y, seed, device):
    torch.manual_seed(seed)
    model = LSTMPredictor(input_dim=INPUT_DIM, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                          num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)),
                        batch_size=32, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]

    train_raw = [read_raw_csv_with_timestamp(f) for f in [train_csv] + augment]
    tb_raw = [flight_to_binned(r, t, warmup_sec=WARMUP_SEC) for r, t in train_raw]
    fscaler = StandardScaler(); fscaler.fit(np.vstack(tb_raw))
    train_binned = [fscaler.transform(b) for b in tb_raw]

    P, N = [], []
    for b in train_binned:
        p, nx, _ = build_pairs(b, CONTEXT_BINS, HORIZON, stride=1)
        if len(p):
            P.append(p); N.append(nx)
    X, Y = np.concatenate(P, axis=0), np.concatenate(N, axis=0)
    print(f"training {N_MEMBERS} ensemble members on {len(X)} pairs (context {CONTEXT_BINS}s -> +{HORIZON}s)")
    members = [train_member(X, Y, RANDOM_SEED + i, device) for i in range(N_MEMBERS)]

    def spread(binned):
        """-> (context_end_time, full-vector disagreement, mag_x disagreement)"""
        pasts, _, tgt = build_pairs(binned, CONTEXT_BINS, HORIZON, stride=1)
        if len(pasts) == 0:
            return None
        pt = torch.tensor(pasts).to(device)
        with torch.no_grad():
            preds = np.stack([m(pt).cpu().numpy() for m in members], axis=0)  # (M, N, D)
        return (np.asarray(tgt) - HORIZON,
                preds.std(axis=0).mean(axis=1),
                preds[:, :, MAG_X_COLS].std(axis=0).mean(axis=1))

    eval_raw = [(os.path.basename(f), *read_raw_csv_with_timestamp(f)) for f in eval_files]
    clean = {}
    c_all, c_mx = [], []
    for fn, raw, t in eval_raw:
        out = spread(flight_to_binned(raw, t, feature_scaler=fscaler, warmup_sec=WARMUP_SEC))
        if out is None:
            continue
        clean[fn] = out
        c_all.append(out[1]); c_mx.append(out[2])
    thr_all = float(np.percentile(np.concatenate(c_all), 99))
    thr_mx = float(np.percentile(np.concatenate(c_mx), 99))
    print(f"disagreement thresholds (99th pct, clean flights): all={thr_all:.4f}  mag_x={thr_mx:.4f}")

    rows = []
    for kind in ["slow_ramp", "step"]:
        print(f"\n=== {kind} ===")
        leads = {"ensemble_all": [], "ensemble_magx": []}
        for fn, raw, t in eval_raw:
            if fn not in clean:
                continue
            _, ca, cm = clean[fn]
            for onset in ONSET_STARTS:
                if onset + RAMP_SEC + SUSTAIN_SEC > t[-1]:
                    continue
                inj, dev = inject(raw, t, kind, onset, FAULT_AMPLITUDE_STD)
                vis = t[dev >= VISIBLE_SIGMA]
                if len(vis) == 0:
                    continue
                manifest = float(vis[0])
                out = spread(flight_to_binned(inj, t, feature_scaler=fscaler, warmup_sec=WARMUP_SEC))
                if out is None:
                    continue
                ce, sa, sm = out
                for key, s, c, th in (("ensemble_all", sa, ca, thr_all),
                                      ("ensemble_magx", sm, cm, thr_mx)):
                    n = min(len(ce), len(s), len(c))
                    m = (s[:n] > th) & ~(c[:n] > th) & (ce[:n] <= manifest)
                    leads[key].append(manifest - ce[:n][m].min() if m.any() else np.nan)
        for key, v in leads.items():
            a = np.array(v, dtype=float); det = ~np.isnan(a)
            med = float(np.median(a[det])) if det.any() else float("nan")
            rows.append({"fault_type": kind, "method": key, "n": len(a),
                         "detected": int(det.sum()), "median_lead_sec": med})
            msg = f"median lead {med:+.1f}s" if det.any() else "never fired early"
            print(f"  {key:16s} detected {int(det.sum())}/{len(a)}   {msg}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ensemble_uncertainty.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(rows, os.path.join(FIGURES_DIR, "ensemble_uncertainty_lead_time.png"))
    print(f"\nsaved results to {RESULTS_DIR}/ and figure to {FIGURES_DIR}/")


def plot(rows, save_path):
    kinds = ["slow_ramp", "step"]
    methods = ["ensemble_all", "ensemble_magx"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(kinds)); w = 0.34
    for i, meth in enumerate(methods):
        vals, dets = [], []
        for k in kinds:
            r = next((r for r in rows if r["fault_type"] == k and r["method"] == meth), None)
            v = r["median_lead_sec"] if r else float("nan")
            vals.append(0.0 if (v is None or np.isnan(v)) else v)
            dets.append(f"{r['detected']}/{r['n']}" if r else "-")
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals, w * 0.9, color=PALETTE[i], label=meth)
        for xi, v, d in zip(xs, vals, dets):
            ax.text(xi, v + 0.3, f"{v:.0f}s\n({d})" if v > 0 else f"none\n({d})",
                    ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.axhline(HORIZON, color=INK_PRIMARY, linestyle="--", linewidth=1.0)
    ax.text(1.45, HORIZON + 0.4, f"{HORIZON}s required", fontsize=9, color=INK_SECONDARY, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["slow_ramp\n(gradual -- forecastable)", "step\n(abrupt -- control)"], fontsize=9)
    ax.set_ylabel("median lead time before the fault is visible (s)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Ensemble disagreement ({N_MEMBERS} members) as a precursor signal", color=INK_PRIMARY)
    fig.tight_layout(); fig.savefig(save_path, dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
