"""
mag_x-specific scoring for the 15s-ahead predictor (2026-08-20 follow-up).

The multi-session model (train_eval_lstm_predictor_15s_multisession.py)
still missed the one verified fault, and its biggest response across the
whole file was a large, unrelated event at ~31s (confirmed via mag_x's own
raw z-score: the file's single largest mag_x deviation, z=7.55, is at
42.65s -- inside the verified burst window -- while the ~31s event is NOT
mag_x-specific, it moves the FUSED score because it's large across many
channels). The fused score (averaged over all 16 channels) can't tell those
two apart. This reuses the already-trained multisession model (no
retraining) but scores using ONLY the 4 engineered features belonging to
mag_x (columns [40:44] of the 64-dim output -- channel c's 4 features live
at [4c:4c+4], mag_x is channel index 10) instead of the 16-channel fused
score, to see if isolating the known fault channel filters out the
unrelated event.
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
    CONTEXT_BINS, HORIZON_BINS, MIN_FILE_SEC, INPUT_DIM, BURST_FILE, BURST_REAL_SEC,
    INK_PRIMARY, INK_SECONDARY, GRIDLINE, STATUS_CRITICAL, CAT_BLUE, CAT_ORANGE,
    bin_by_real_second, build_pairs, flight_to_binned,
)
from train_eval_lstm_predictor_15s_multisession import DUPLICATES, EVAL_ONLY

MAG_X_CHANNEL_IDX = Config.CHANNEL_NAMES.index("mag_x")  # 10
MAG_X_COLS = slice(MAG_X_CHANNEL_IDX * 4, MAG_X_CHANNEL_IDX * 4 + 4)

MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_predictor_15s_multisession_model.pth")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- reproduce the exact scaler + feature_scaler from the multisession
    # run (neither was persisted to disk -- both are deterministic refits
    # given the same file list/order, same trick used in eval_robust_baseline.py) ---
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment_files = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]
    train_file_paths = [train_csv] + augment_files

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
    results = {name: [] for name in methods}

    for f in eval_files:
        name = os.path.basename(f)
        raw, real_time = read_raw_csv_with_timestamp(f)
        binned = flight_to_binned(raw, real_time, feature_scaler=feature_scaler)
        if len(binned) < MIN_FILE_SEC:
            continue
        pasts, nexts, target_idx = build_pairs(binned, CONTEXT_BINS, HORIZON_BINS, stride=1)
        pasts_t = torch.tensor(pasts)
        for method_name, predict_fn in methods.items():
            preds = predict_fn(pasts_t)
            err_magx = np.abs(preds[:, MAG_X_COLS] - nexts[:, MAG_X_COLS]).mean(axis=1)  # (N,)
            for score, t_idx in zip(err_magx, target_idx):
                in_burst = (name == BURST_FILE) and (t_idx < BURST_REAL_SEC[1]) and (t_idx + 1 > BURST_REAL_SEC[0])
                results[method_name].append({"file": name, "target_sec": int(t_idx),
                                              "score": float(score), "in_burst": in_burst})

    print(f"{'method':20s} {'pool_n':>7s} {'thresh_99p':>11s} {'fp_rate':>8s} {'burst_score':>12s} {'hit':>4s}")
    summary = {}
    for method_name, rows in results.items():
        normal_pool = [r["score"] for r in rows if not r["in_burst"]]
        burst_scores = [r["score"] for r in rows if r["in_burst"]]
        threshold = float(np.percentile(normal_pool, 99)) if normal_pool else 0.0
        fp_rate = float(np.mean([s > threshold for s in normal_pool])) if normal_pool else float("nan")
        hits = sum(1 for s in burst_scores if s > threshold)
        print(f"{method_name:20s} {len(normal_pool):7d} {threshold:11.4f} {fp_rate:8.4f} "
              f"{burst_scores[0] if burst_scores else float('nan'):12.4f} {hits}/{len(burst_scores)}")
        summary[method_name] = {"threshold": threshold, "rows": rows, "hits": hits, "n_burst": len(burst_scores)}

    plot_magx_vs_fused_timeline(summary, os.path.join(FIGURES_DIR, "lstm_predictor_15s_magx_specific_timeline.png"))
    print(f"\nsaved figure to {FIGURES_DIR}/")


def plot_magx_vs_fused_timeline(summary, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"lstm_15s_magx": CAT_BLUE, "persistence_magx": CAT_ORANGE}
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
                   label=f"99th-pct pooled-normal threshold ({s['threshold']:.2f})")
        ax.axvspan(BURST_REAL_SEC[0], BURST_REAL_SEC[1], color=STATUS_CRITICAL, alpha=0.08)
        ax.set_title(f"{name}: mag_x-only forecast-error vs. target time", color=INK_PRIMARY, loc="left", fontsize=11)
        ax.set_ylabel("mag_x-only error")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[-1].set_xlabel(f"target time (real seconds into {BURST_FILE}) -- context ends 15s earlier")
    fig.suptitle("mag_x-only score: does isolating the known fault channel filter out the ~31s unrelated event?",
                 color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
