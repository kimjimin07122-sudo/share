"""
Input-pipeline ablation (2026-08-20): fixing the three preprocessing defects
found while auditing the LSTM-AE against Reis & Reis 2025 + Ahn & Chung 2024.

The audit found the LSTM-AE isn't actually reconstructing test files at all:
after the train-fit feature scaler, test features land at std ~275 / max
~46000 (train: std ~1 / max ~123), the model saturates, and its output ends
up uncorrelated with its input (corr ~ -0.007). So today's "anomaly score"
mostly measures *how far outside the training distribution a file sits* --
which is exactly the session drift this project has been chasing all along.

Three defects, each fixed as a separate ablation step so we can see what
each one actually buys:

  1. ENERGY BLOWUP. energy = raw^2, so a session offset is amplified
     QUADRATICALLY. The 8 most out-of-distribution features on a test file
     are all `.energy` (accel_z.energy peaks at 46404). Fix: per-flight
     normalization (below) keeps raw ~ N(0,1) per file, so energy ~ chi2(1)
     with mean 1 in every file; variant D additionally drops energy.

  2. NO PER-FLIGHT NORMALIZATION. Reis & Reis sec 6.1 states "All channels
     were normalized per flight to ensure unit consistency and model
     generalization" -- we instead fit one scaler on the training flight and
     apply it to every test flight, which is what lets the session offset
     through in the first place. (PROJECT_SUMMARY lists per-flight
     normalization as a ruled-out session-drift fix; that earlier test was
     on raw channels only and did not address the energy-squaring path.)

  3. WINDOW LENGTH IS A SAMPLE COUNT, NOT A DURATION. LSTM_AE_WINDOW=32
     samples means 32ms on the ~991Hz training flight but 128ms on the
     ~250Hz test sorties -- the same "window" spans 4x more real time at
     evaluation than at training. Reis & Reis uses 5s windows. Fix:
     resample every flight to a common rate first, so a window is a fixed
     real duration everywhere.

Variants (each trains its own LSTM-AE from scratch, same architecture):
  A  current pipeline (baseline, for comparison)
  B  A + per-flight normalization
  C  B + resample to a common rate, window defined in real seconds
  D  C + drop the energy feature entirely (3 features/channel instead of 4)

Reported per variant: feature-distribution match, whether reconstruction
actually tracks the input, detection metrics, and the channel-diagnosis
control check (the session-drift symptom: does mag_x falsely rank #1 in a
NORMAL stretch of the file?).
"""
import csv
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
from data_loader import build_scaler
from gru_data_loader import read_raw_csv_with_timestamp
from networks.lstm_ae import LSTMAutoencoder
from eval import compute_binary_metrics

import glob

HOLDOUT_FRAC = 0.15
EPOCHS = Config.LSTM_AE_EPOCHS
RESAMPLE_HZ = 50.0          # common rate for variants C/D
WINDOW_SEC = 1.0            # window duration in REAL seconds for C/D
RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_pipeline_ablation")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")

BURST_FILE = "x70_20180611_x76_070620.csv"
BURST_SEC = (42.19, 42.99)
CONTROL_SEC = [(8.0, 9.2), (20.0, 21.2), (36.5, 37.7)]  # real-time controls (see PROJECT_SUMMARY 3.8)

INK_PRIMARY, INK_SECONDARY = "#0b0b0b", "#52514e"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"
STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"
CAT_BLUE, CAT_ORANGE, CAT_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def resample(raw, real_time, hz):
    """Bin to a fixed rate so a window means the same real duration everywhere."""
    n = int(np.floor(real_time[-1] * hz))
    if n < 1:
        return raw[:0], real_time[:0]
    idx = np.minimum((real_time * hz).astype(int), n - 1)
    out = np.zeros((n, raw.shape[1]), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    np.add.at(out, idx, raw)
    np.add.at(cnt, idx, 1)
    cnt = np.maximum(cnt, 1)
    return (out / cnt[:, None]).astype(np.float32), (np.arange(n) / hz).astype(np.float64)


def make_features(scaled, time_sec, use_energy):
    """raw / d1 / d2 [/ energy] per channel, derivatives on the real time axis."""
    d1 = np.gradient(scaled, time_sec, axis=0)
    d2 = np.gradient(d1, time_sec, axis=0)
    parts = [scaled, d1, d2] + ([scaled ** 2] if use_energy else [])
    stacked = np.stack(parts, axis=-1)
    T, C, F = stacked.shape
    return stacked.reshape(T, C * F).astype(np.float32)


def windows_of(feats, w, stride):
    if len(feats) < w:
        return np.zeros((0, w, feats.shape[1]), dtype=np.float32)
    return np.stack([feats[i:i + w] for i in range(0, len(feats) - w + 1, stride)]).astype(np.float32)


class Variant:
    def __init__(self, name, per_flight_norm, resample_hz, window, stride, use_energy, desc):
        self.name, self.per_flight_norm = name, per_flight_norm
        self.resample_hz, self.window, self.stride = resample_hz, window, stride
        self.use_energy, self.desc = use_energy, desc
        self.feat_per_ch = 4 if use_energy else 3


VARIANTS = [
    Variant("A_current", False, None, Config.LSTM_AE_WINDOW, Config.LSTM_AE_STRIDE, True,
            "current pipeline (global scaler, 32-sample window, energy)"),
    Variant("B_perflight", True, None, Config.LSTM_AE_WINDOW, Config.LSTM_AE_STRIDE, True,
            "+ per-flight normalization"),
    Variant("C_resampled", True, RESAMPLE_HZ, int(WINDOW_SEC * RESAMPLE_HZ), 2, True,
            f"+ resample {RESAMPLE_HZ:.0f}Hz, {WINDOW_SEC}s window"),
    Variant("D_no_energy", True, RESAMPLE_HZ, int(WINDOW_SEC * RESAMPLE_HZ), 2, False,
            "+ drop energy feature"),
]


def prep_file(raw, real_time, v, ref_scaler):
    """Returns (features, time_axis). ref_scaler is used only when NOT per-flight."""
    if v.resample_hz:
        raw, real_time = resample(raw, real_time, v.resample_hz)
    if v.per_flight_norm:
        sc = build_scaler()
        sc.fit(raw)
    else:
        sc = ref_scaler
    return make_features(sc.transform(raw), real_time, v.use_energy), real_time


def run_variant(v, device):
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    raw_all, time_all = read_raw_csv_with_timestamp(train_csv)
    split = int(len(raw_all) * (1 - HOLDOUT_FRAC))
    raw_tr, time_tr = raw_all[:split], time_all[:split]
    raw_ho, time_ho = raw_all[split:], time_all[split:] - time_all[split]

    ref_scaler = build_scaler()
    ref_scaler.fit(raw_tr)

    feat_tr_raw, _ = prep_file(raw_tr, time_tr, v, ref_scaler)
    feat_ho_raw, _ = prep_file(raw_ho, time_ho, v, ref_scaler)

    fscaler = build_scaler()
    fscaler.fit(feat_tr_raw)
    feat_tr = fscaler.transform(feat_tr_raw)
    feat_ho = fscaler.transform(feat_ho_raw)

    input_dim = Config.NUM_CHANNELS * v.feat_per_ch
    tr_w = windows_of(feat_tr, v.window, v.stride)

    model = LSTMAutoencoder(input_dim, Config.LSTM_AE_HIDDEN_SIZE,
                            Config.LSTM_AE_NUM_LAYERS, Config.LSTM_AE_BOTTLENECK_DIM).to(device)
    opt = optim.Adam(model.parameters(), lr=Config.LSTM_AE_LEARNING_RATE)
    lossf = nn.MSELoss()
    t = torch.tensor(tr_w)
    loader = DataLoader(TensorDataset(t, t), batch_size=Config.LSTM_AE_BATCH_SIZE, shuffle=True)
    for ep in range(EPOCHS):
        model.train()
        tot = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
    train_mse = tot / n
    model.eval()

    def recon(win):
        outs = []
        with torch.no_grad():
            for i in range(0, len(win), 512):
                outs.append(model(torch.tensor(win[i:i + 512]).to(device)).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros_like(win)

    # Isolation Forest + fusion stats on the training portion
    r_tr = recon(tr_w)
    res_tr = np.abs(r_tr - tr_w)
    if_feat_tr = np.concatenate([tr_w.mean(axis=1), res_tr.mean(axis=1)], axis=1)
    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS,
                              random_state=Config.RANDOM_SEED, n_jobs=-1).fit(if_feat_tr)
    if_tr = -iforest.score_samples(if_feat_tr)
    if_mu, if_sd = if_tr.mean(), (if_tr.std() or 1e-6)
    err_mu, err_sd = res_tr.mean(), (res_tr.std() or 1e-6)

    def fused(win):
        if len(win) == 0:
            return np.array([])
        r = recon(win)
        res = np.abs(r - win)
        lstm_n = (res.mean(axis=(1, 2)) - err_mu) / err_sd
        iff = np.concatenate([win.mean(axis=1), res.mean(axis=1)], axis=1)
        if_n = (-iforest.score_samples(iff) - if_mu) / if_sd
        return Config.LSTM_AE_FUSION_ALPHA * lstm_n + (1 - Config.LSTM_AE_FUSION_ALPHA) * if_n

    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))

    # threshold from held-out normal blocks
    bounds = np.linspace(0, len(feat_ho), len(test_files) + 1, dtype=int)
    normal_scores = []
    for i in range(len(test_files)):
        s = fused(windows_of(feat_ho[bounds[i]:bounds[i + 1]], v.window, v.stride))
        if len(s):
            normal_scores.append(float(s.max()))
    threshold = max(normal_scores) if normal_scores else 0.0

    y_true = [0] * len(normal_scores)
    scores_all = list(normal_scores)
    diag = {}
    feat_stats = None
    for f in test_files:
        raw, rt = read_raw_csv_with_timestamp(f)
        feats_raw, taxis = prep_file(raw, rt, v, ref_scaler)
        feats = fscaler.transform(feats_raw)
        win = windows_of(feats, v.window, v.stride)
        s = fused(win)
        if len(s) == 0:
            continue
        y_true.append(1); scores_all.append(float(s.max()))

        if os.path.basename(f) == BURST_FILE:
            feat_stats = (float(feats.std()), float(np.abs(feats).max()))
            w1 = windows_of(feats, v.window, 1)
            r1 = recon(w1)
            diag["recon_corr"] = float(np.corrcoef(r1.ravel(), w1.ravel())[0, 1])
            diag["recon_mae"] = float(np.abs(r1 - w1).mean())
            # per-window, per-channel error -> file-wide own-baseline ranking
            pw = np.abs(r1 - w1).mean(axis=1)
            pc = pw.reshape(len(pw), Config.NUM_CHANNELS, v.feat_per_ch).mean(axis=2)
            wt = taxis[np.arange(len(pc)) + v.window - 1]
            bm = (wt >= BURST_SEC[0]) & (wt <= BURST_SEC[1])
            base = pc[~bm].mean(axis=0)
            if bm.sum():
                ex = np.maximum(pc[bm].mean(axis=0) - base, 0)
                diag["burst_ch11_rank"] = int(np.where(np.argsort(-ex) + 1 == 11)[0][0]) + 1
            ranks = []
            for lo, hi in CONTROL_SEC:
                cm = (wt >= lo) & (wt <= hi)
                if cm.sum():
                    ex = np.maximum(pc[cm].mean(axis=0) - base, 0)
                    ranks.append(int(np.where(np.argsort(-ex) + 1 == 11)[0][0]) + 1)
            diag["control_ch11_ranks"] = ranks

    m = compute_binary_metrics(y_true, scores_all, threshold)
    return {
        "variant": v.name, "desc": v.desc, "train_mse": train_mse,
        "train_windows": len(tr_w), "input_dim": input_dim,
        "train_feat_std": float(feat_tr.std()), "train_feat_max": float(np.abs(feat_tr).max()),
        "test_feat_std": feat_stats[0] if feat_stats else float("nan"),
        "test_feat_max": feat_stats[1] if feat_stats else float("nan"),
        **diag, **{k: m[k] for k in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN"]},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(Config.RANDOM_SEED)
    print("device:", device)
    rows = []
    for v in VARIANTS:
        print(f"\n=== {v.name}: {v.desc} ===")
        r = run_variant(v, device)
        rows.append(r)
        print(f"  train windows={r['train_windows']}  input_dim={r['input_dim']}  train mse={r['train_mse']:.4f}")
        print(f"  feature scale  train std={r['train_feat_std']:8.3f} max={r['train_feat_max']:10.1f}")
        print(f"                 test  std={r['test_feat_std']:8.3f} max={r['test_feat_max']:10.1f}")
        print(f"  reconstruction on test: corr={r.get('recon_corr', float('nan')):+.4f}  MAE={r.get('recon_mae', float('nan')):.3f}")
        print(f"  detection: AUROC={r['AUROC']:.4f}  Recall={r['Recall']:.4f}  FP={r['FP']}  FN={r['FN']}")
        print(f"  channel diagnosis: burst ch11 rank={r.get('burst_ch11_rank','-')}  control ranks={r.get('control_ch11_ranks','-')}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RESULTS_DIR, "ablation.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\nsaved results to {RESULTS_DIR}/")
    plot_ablation(rows, os.path.join(FIGURES_DIR, "pipeline_fixes_ablation.png"))
    print(f"saved figure to {FIGURES_DIR}/")


def plot_ablation(rows, save_path):
    names = [r["variant"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    vals = [r["test_feat_max"] for r in rows]
    ax.bar(x, vals, color=[STATUS_CRITICAL if v > 1000 else STATUS_GOOD for v in vals], width=0.6)
    ax.set_yscale("log")
    ax.set_title("test-feature blowup\n(max |scaled value|, log)", color=INK_PRIMARY, fontsize=11)
    ax.axhline(rows[0]["train_feat_max"], color=INK_PRIMARY, linestyle="--", linewidth=1.0,
               label="train-side max")
    ax.legend(frameon=False, fontsize=8)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

    ax = axes[1]
    vals = [r.get("recon_corr", 0.0) for r in rows]
    ax.bar(x, vals, color=[STATUS_GOOD if v > 0.5 else STATUS_CRITICAL for v in vals], width=0.6)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="#c3c2b7", linewidth=1.0)
    ax.set_title("does reconstruction track\nthe input? (corr on test)", color=INK_PRIMARY, fontsize=11)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:+.2f}", ha="center", fontsize=9, color=INK_SECONDARY)

    ax = axes[2]
    w = 0.38
    auroc = [r["AUROC"] for r in rows]
    recall = [r["Recall"] for r in rows]
    ax.bar(x - w / 2, auroc, w, color=CAT_BLUE, label="AUROC")
    ax.bar(x + w / 2, recall, w, color=CAT_ORANGE, label="Recall")
    ax.set_ylim(0, 1.15)
    ax.set_title("detection performance", color=INK_PRIMARY, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")
    for xi, v in zip(x - w / 2, auroc):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)
    for xi, v in zip(x + w / 2, recall):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("LSTM-AE input-pipeline fixes: per-flight normalization, real-time windows, energy feature",
                 color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
