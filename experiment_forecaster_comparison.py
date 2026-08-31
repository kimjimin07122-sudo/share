"""
Building a forecaster that can actually anticipate (2026-08-20).

experiment_early_warning.py established that forecast-then-detect is the
right scoring rule but that our LSTM cannot supply the forecast it needs:
trained only on normal flight, it emits normal-looking futures. Through an
injected ramp the true mag_x rose +1.35 -> +5.85 while the LSTM forecast
moved -0.83 -> +0.19, regressing to the mean instead of extrapolating
(PROJECT_SUMMARY sec 3.13).

That is a property of the FORECASTER, not of the scoring rule. So this
compares four forecasters under one identical scoring rule and one identical
evaluation, to find one that extrapolates:

  lstm            the current model. Learns normal dynamics; expected to
                  regress to the mean.
  persistence     forecast(T+H) = state(T). Cannot anticipate anything, but
                  it does at least inherit the current level, so during a
                  ramp its "forecast" is already elevated. Serves as the
                  floor: any lead time it earns is not anticipation, it is
                  just detecting that NOW is abnormal.
  linear_extrap   least-squares slope over the context, extended H seconds.
                  This is the one that can genuinely anticipate: on a rising
                  ramp the extrapolated value exceeds the current one, so it
                  should cross the alarm threshold EARLIER than persistence.
                  The gap between linear_extrap and persistence is the real
                  value added by forecasting.
  lstm_residual   persistence + an LSTM-learned correction. Standard fix for
                  mean-regression: the forecast inherits the current level by
                  construction, and the network only has to learn the delta.

Scoring, thresholds, causal normalization and the attributable-alarm rule are
inherited unchanged from experiment_early_warning.py, so the only variable is
the forecaster. Faults span forecastable (slow_ramp, 25s onset) to
unforecastable (step, instantaneous). A method that shows lead time on step
is broken, not good -- step is the control.
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
from sklearn.ensemble import IsolationForest
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

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_forecaster_comparison")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ------------------------------------------------------------- forecasters
def fc_persistence(pasts, _model=None):
    return pasts[:, -1, :].copy()


def fc_linear_extrap(pasts, _model=None):
    """Least-squares slope per feature over the context, extended HORIZON bins
    past its end. Bins are 1s, so the extrapolation distance is HORIZON
    seconds. No training and no normal-data prior -- which is exactly why it
    can follow a trend out of the normal range."""
    n = pasts.shape[1]
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    # slope per (sample, feature) via the closed-form least-squares estimate
    slope = np.einsum("t,ntd->nd", xc, pasts.astype(np.float64)) / denom
    intercept = pasts.mean(axis=1) - slope * x.mean()
    return (intercept + slope * (n - 1 + HORIZON)).astype(np.float32)


def train_lstm(train_binned, device, residual=False):
    """residual=True trains the network on (target - last context bin), so the
    forecast is persistence plus a learned correction."""
    P, N = [], []
    for b in train_binned:
        p, nx, _ = build_pairs(b, CONTEXT_BINS, HORIZON, stride=1)
        if len(p):
            P.append(p); N.append(nx)
    X = np.concatenate(P, axis=0)
    Y = np.concatenate(N, axis=0)
    if residual:
        Y = Y - X[:, -1, :]

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
    return model


def make_lstm_fc(model, device, residual=False):
    def fc(pasts, _m=None):
        with torch.no_grad():
            out = model(torch.tensor(pasts).to(device)).cpu().numpy()
        return pasts[:, -1, :] + out if residual else out
    return fc


# ------------------------------------------------------------------- eval
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]

    train_raw = [read_raw_csv_with_timestamp(f) for f in [train_csv] + augment]
    train_binned_raw = [flight_to_binned(r, t, warmup_sec=WARMUP_SEC) for r, t in train_raw]
    fscaler = StandardScaler()
    fscaler.fit(np.vstack(train_binned_raw))
    train_binned = [fscaler.transform(b) for b in train_binned_raw]

    print("training forecasters...")
    m_plain = train_lstm(train_binned, device, residual=False)
    m_resid = train_lstm(train_binned, device, residual=True)
    forecasters = {
        "lstm": make_lstm_fc(m_plain, device, residual=False),
        "persistence": fc_persistence,
        "linear_extrap": fc_linear_extrap,
        "lstm_residual": make_lstm_fc(m_resid, device, residual=True),
    }

    normal_states = np.concatenate(train_binned, axis=0)
    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS,
                              random_state=RANDOM_SEED, n_jobs=-1).fit(normal_states)
    mu = normal_states[:, MAG_X_COLS].mean(axis=0)
    sd = np.where(normal_states[:, MAG_X_COLS].std(axis=0) == 0, 1e-6,
                  normal_states[:, MAG_X_COLS].std(axis=0))

    def scores(fc, binned):
        pasts, _, tgt = build_pairs(binned, CONTEXT_BINS, HORIZON, stride=1)
        if len(pasts) == 0:
            return None
        pred = fc(pasts)
        ctx_end = np.asarray(tgt) - HORIZON
        return ctx_end, -iforest.score_samples(pred), np.abs((pred[:, MAG_X_COLS] - mu) / sd).mean(axis=1)

    eval_raw = [(os.path.basename(f), *read_raw_csv_with_timestamp(f)) for f in eval_files]

    # thresholds per forecaster, from the CLEAN eval flights
    thr = {}
    clean_cache = {}
    for name, fc in forecasters.items():
        c_if, c_mz = [], []
        for fn, raw, t in eval_raw:
            out = scores(fc, flight_to_binned(raw, t, feature_scaler=fscaler, warmup_sec=WARMUP_SEC))
            if out is None:
                continue
            clean_cache[(name, fn)] = out
            c_if.append(out[1]); c_mz.append(out[2])
        thr[name] = (float(np.percentile(np.concatenate(c_if), 99)),
                     float(np.percentile(np.concatenate(c_mz), 99)))

    kinds = ["slow_ramp", "step"]        # forecastable vs control
    rows = []
    for kind in kinds:
        print(f"\n=== {kind} ===")
        for name, fc in forecasters.items():
            leads_if, leads_mz = [], []
            for fn, raw, t in eval_raw:
                if (name, fn) not in clean_cache:
                    continue
                _, c_if, c_mz = clean_cache[(name, fn)]
                for onset in ONSET_STARTS:
                    if onset + RAMP_SEC + SUSTAIN_SEC > t[-1]:
                        continue
                    inj, dev = inject(raw, t, kind, onset, FAULT_AMPLITUDE_STD)
                    vis = t[dev >= VISIBLE_SIGMA]
                    if len(vis) == 0:
                        continue
                    manifest = float(vis[0])
                    out = scores(fc, flight_to_binned(inj, t, feature_scaler=fscaler, warmup_sec=WARMUP_SEC))
                    if out is None:
                        continue
                    ce, s_if, s_mz = out
                    for store, s, c, th in ((leads_if, s_if, c_if, thr[name][0]),
                                            (leads_mz, s_mz, c_mz, thr[name][1])):
                        n = min(len(ce), len(s), len(c))
                        m = (s[:n] > th) & ~(c[:n] > th) & (ce[:n] <= manifest)
                        store.append(manifest - ce[:n][m].min() if m.any() else np.nan)

            for tag, leads in (("IF", leads_if), ("magx", leads_mz)):
                a = np.array(leads, dtype=float)
                det = ~np.isnan(a)
                med = float(np.median(a[det])) if det.any() else float("nan")
                rows.append({"fault_type": kind, "forecaster": name, "scorer": tag,
                             "n": len(a), "detected": int(det.sum()), "median_lead_sec": med})
                msg = f"median lead {med:+.1f}s" if det.any() else "never fired early"
                print(f"  {name:14s} [{tag:4s}] detected {int(det.sum())}/{len(a)}   {msg}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "forecaster_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(rows, list(forecasters), os.path.join(FIGURES_DIR, "forecaster_comparison_lead_time.png"))
    print(f"\nsaved results to {RESULTS_DIR}/ and figure to {FIGURES_DIR}/")


def plot(rows, names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kind, sub in zip(axes, ["slow_ramp", "step"],
                             ["gradual 25s onset -- forecastable", "instantaneous -- control, must show none"]):
        x = np.arange(len(names)); w = 0.38
        for i, tag in enumerate(["IF", "magx"]):
            vals = []
            for nm in names:
                r = next((r for r in rows if r["fault_type"] == kind and r["forecaster"] == nm
                          and r["scorer"] == tag), None)
                v = r["median_lead_sec"] if r else float("nan")
                vals.append(0.0 if (v is None or np.isnan(v)) else v)
            xs = x + (i - 0.5) * w
            ax.bar(xs, vals, w * 0.9, color=PALETTE[i], label=f"scorer: {tag}")
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.4, f"{v:.0f}s" if v > 0 else "none", ha="center",
                        fontsize=8.5, color=INK_SECONDARY)
        ax.axhline(HORIZON, color=INK_PRIMARY, linestyle="--", linewidth=1.0)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=12, ha="right")
        ax.set_title(f"{kind}\n{sub}", color=INK_PRIMARY, fontsize=10)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("median lead time before the fault is visible (s)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].text(0.02, HORIZON + 0.6, f"{HORIZON}s required", fontsize=8.5, color=INK_SECONDARY)
    fig.suptitle("Which forecaster can actually anticipate? (identical scoring, identical evaluation)",
                 color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
