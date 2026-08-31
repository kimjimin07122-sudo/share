"""
15s-ahead LSTM predictor, v2: train on MULTIPLE flight sessions instead of
just x70_all.csv (2026-08-20, user's suggestion after v1's ~100-training-pair
result came out too noisy/inconclusive to trust -- see train_eval_lstm_predictor_15s.py).

Tried locating the real fault window in the other 18 test files first (so
they could be trained on with the fault excluded), using the single global
argmax of the already-validated LSTM-AE fused score per file -- deliberately
NOT the per-channel search already ruled out in PROJECT_SUMMARY sec 3.3.
That validation failed: for the one file with verified ground truth
(x70_20180611_x76_070620.csv, real burst 42.19-42.99s) the auto-located peak
was at 31.26s, a clean miss, and peaks clustered suspiciously at the very
last second of many files (likely landing dynamics or a derivative-boundary
artifact, not the labeled fault). So fault exclusion was abandoned.

Instead: pool MOST of the 19 test files' FULL, unmodified content into
training alongside x70_all.csv, accepting that each file's brief (<1s per
the one verified example) fault is a small, likely-tolerable contamination
fraction of file's total duration (this is standard practice in anomaly
detection when clean data is scarce). A handful of files are held out
completely untouched for evaluation, including the one file with verified
fault timing -- so the eval side stays honest even though the train side is
now weakly contaminated.

File split (duplicates x70_20180601_x76_083322_02.csv /
x70_20180601_x76_083916_02.csv are byte-identical to their non-_02
counterparts -- diff confirmed -- so dropped entirely to avoid leakage):

  EVAL_ONLY (4 files, never touched during training):
    x70_02.csv, x70_20180601_x76_083916.csv,
    x70_20180611_x66_082253.csv, x70_20180611_x76_070620.csv (verified fault)

  TRAIN_AUGMENT (13 remaining test files, full content, weak contamination
  accepted) + x70_all.csv
"""
import csv
import glob
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_loader import build_scaler, StandardScaler
from gru_data_loader import read_raw_csv_with_timestamp
from feature_engineering import compute_sequence_features, compute_window_summary
from networks.lstm_predictor import LSTMPredictor
from train_eval_lstm_predictor_15s import (
    BIN_SIZE_SEC, CONTEXT_BINS, HORIZON_BINS, MIN_FILE_SEC, INPUT_DIM,
    EPOCHS, LEARNING_RATE, RANDOM_SEED, BURST_FILE, BURST_REAL_SEC,
    INK_PRIMARY, INK_SECONDARY, GRIDLINE, STATUS_CRITICAL, CAT_BLUE, CAT_ORANGE,
    bin_by_real_second, build_pairs, fused_score, flight_to_binned,
    plot_timeline, plot_pool_comparison,
)
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_lstm_predictor_15s_multisession")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_predictor_15s_multisession_model.pth")

DUPLICATES = {"x70_20180601_x76_083322_02.csv", "x70_20180601_x76_083916_02.csv"}
EVAL_ONLY = {
    "x70_02.csv",
    "x70_20180601_x76_083916.csv",
    "x70_20180611_x66_082253.csv",
    BURST_FILE,
}


def raw_and_time(filepath):
    return read_raw_csv_with_timestamp(filepath)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment_files = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]
    print(f"training on x70_all.csv + {len(augment_files)} augmented sessions:")
    for f in augment_files:
        print(f"  + {os.path.basename(f)}")
    print(f"held out purely for evaluation ({len(eval_files)} files):")
    for f in eval_files:
        print(f"  - {os.path.basename(f)}" + ("  <-- verified fault 42.19-42.99s" if os.path.basename(f) == BURST_FILE else ""))

    train_file_paths = [train_csv] + augment_files

    # --- pipeline v4: each flight is normalized by its OWN statistics
    # (Reis & Reis sec 6.1) rather than by one scaler pooled across flights.
    # Pooling let session offsets through, and `energy = raw^2` squared them
    # (PROJECT_SUMMARY sec 3.9). Features are still computed per file, so a
    # bin never spans a file boundary. ---
    raw_time_pairs = [raw_and_time(f) for f in train_file_paths]
    binned_per_file = []
    total_sec = 0.0
    for f, (raw, real_time) in zip(train_file_paths, raw_time_pairs):
        binned = flight_to_binned(raw, real_time)
        binned_per_file.append(binned)
        total_sec += real_time[-1]
        print(f"  {os.path.basename(f):45s} {real_time[-1]:7.1f}s -> {len(binned)} bins")
    print(f"total training duration across {len(train_file_paths)} sessions: {total_sec:.1f}s "
          f"(v1 had only x70_all.csv's 131.9s)")

    feature_scaler = StandardScaler()
    feature_scaler.fit(np.vstack(binned_per_file))
    binned_per_file = [feature_scaler.transform(b) for b in binned_per_file]

    all_pasts, all_nexts = [], []
    for binned in binned_per_file:
        pasts, nexts, _ = build_pairs(binned, CONTEXT_BINS, HORIZON_BINS, stride=1)
        all_pasts.append(pasts)
        all_nexts.append(nexts)
    train_pasts = np.concatenate(all_pasts, axis=0)
    train_nexts = np.concatenate(all_nexts, axis=0)
    print(f"total training pairs: {len(train_pasts)} (v1 had 102)")

    # --- train LSTM predictor (same architecture as v1) ---
    model = LSTMPredictor(input_dim=INPUT_DIM, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                           num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(train_pasts), torch.tensor(train_nexts)),
                         batch_size=32, shuffle=True)
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
        return pasts_t[:, -1, :].numpy()

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

    # --- evaluate ONLY on the 4 held-out files (never seen in training) ---
    results = {name: [] for name in methods}
    for f in eval_files:
        name = os.path.basename(f)
        raw, real_time = raw_and_time(f)
        binned = flight_to_binned(raw, real_time, feature_scaler=feature_scaler)
        if len(binned) < MIN_FILE_SEC:
            print(f"skip {name}: only {len(binned)}s, need >= {MIN_FILE_SEC}s")
            continue
        pasts, nexts, target_idx = build_pairs(binned, CONTEXT_BINS, HORIZON_BINS, stride=1)
        pasts_t = torch.tensor(pasts)
        for method_name, (predict_fn, iforest, channel_mean, channel_std, if_mean, if_std) in methods.items():
            preds = predict_fn(pasts_t)
            errs = np.abs(preds - nexts)
            for w, e, t_idx in zip(pasts, errs, target_idx):
                score = fused_score(e, w, iforest, channel_mean, channel_std, if_mean, if_std)
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
        print(f"held-out normal-ish pool (4 eval files only): {len(normal_pool)} windows  "
              f"(99th pct threshold = {threshold:.3f})")
        print(f"false-positive rate on that pool: {fp_rate:.4f}")
        print(f"burst-window scores: {[f'{s:.2f}' for s in burst_scores]}")
        print(f"burst windows caught: {hits}/{len(burst_scores)}")

        with open(os.path.join(RESULTS_DIR, f"{method_name}_scores.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "target_sec", "score", "in_burst"])
            writer.writeheader()
            writer.writerows(rows)
        summary[method_name] = {"threshold": threshold, "fp_rate": fp_rate, "hits": hits,
                                 "n_burst": len(burst_scores), "rows": rows}

    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "threshold_99pct", "fp_rate", "burst_hits", "n_burst_windows", "n_training_pairs"])
        for name, s in summary.items():
            writer.writerow([name, s["threshold"], s["fp_rate"], s["hits"], s["n_burst"], len(train_pasts)])
    print(f"\nsaved results to {RESULTS_DIR}/")

    plot_timeline({"lstm_15s": summary["lstm_15s"],
                   "persistence": summary["persistence"]},
                  os.path.join(FIGURES_DIR, "lstm_predictor_15s_multisession_burst_timeline.png"))
    plot_pool_comparison(summary, os.path.join(FIGURES_DIR, "lstm_predictor_15s_multisession_vs_persistence.png"))
    plot_v1_vs_v2(len(train_pasts), summary, os.path.join(FIGURES_DIR, "lstm_predictor_15s_v1_vs_v2_training_size.png"))
    print(f"saved figures to {FIGURES_DIR}/")


def plot_v1_vs_v2(n_pairs_v2, summary, save_path):
    """Headline chart for the 'does more session data help' question:
    training-pair count and burst hit-rate, v1 (single session) vs v2
    (multi-session), read from each run's own summary.csv."""
    v1_summary_path = os.path.join(Config.BASE_DIR, "results_lstm_predictor_15s", "summary.csv")
    v1 = {}
    if os.path.exists(v1_summary_path):
        with open(v1_summary_path) as fh:
            for row in csv.DictReader(fh):
                if row["method"] == "lstm_15s":
                    v1 = row

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    n_v1 = 102  # v1's fixed training-pair count (one 131.9s session)
    ax.bar([0, 1], [n_v1, n_pairs_v2], color=[CAT_ORANGE, CAT_BLUE], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v1: single session", "v2: multi-session"])
    ax.set_ylabel("training pairs")
    ax.set_title("training data size", color=INK_PRIMARY, fontsize=11)
    for xi, v in zip([0, 1], [n_v1, n_pairs_v2]):
        ax.text(xi, v, f"{v}", ha="center", va="bottom", color=INK_SECONDARY, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    v1_hit = float(v1.get("burst_hits", 0)) / max(float(v1.get("n_burst_windows", 1)), 1) if v1 else 0.0
    v2_hit = summary["lstm_15s"]["hits"] / max(summary["lstm_15s"]["n_burst"], 1)
    ax.bar([0, 1], [v1_hit, v2_hit], color=[CAT_ORANGE, CAT_BLUE], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v1: single session", "v2: multi-session"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("burst hit rate")
    ax.set_title("does the alarm fire 15s early?", color=INK_PRIMARY, fontsize=11)
    for xi, v in zip([0, 1], [v1_hit, v2_hit]):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", color=INK_SECONDARY, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle("LSTM 15s-ahead predictor: single-session (v1) vs multi-session (v2) training", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
