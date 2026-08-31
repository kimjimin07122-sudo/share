"""
v4 (2026-08-20): v3's held-out protocol, now on the FIXED input pipeline.

v3 kept the paper's held-out-normal thresholding protocol but fed the model
through a preprocessing path with three defects (PROJECT_SUMMARY sec 3.9):
no per-flight normalization, a window defined as a sample count rather than
a duration, and `energy = raw^2` amplifying session offsets quadratically.
Together those pushed test features hundreds of sigma out of distribution
(max scaled feature 46,404 vs 123 on train), saturating the LSTM-AE so its
output no longer tracked its input at all (corr +0.25, MAE 46.2) -- meaning
the "anomaly score" largely measured distance from the training session
rather than anomalousness. All of that now lives in preprocessing.py.

Unchanged from v3: the held-out protocol itself. The paper's evaluation
(Ahn & Chung sec 4.3) splits a DEDICATED normal-test sortie into n blocks
for thresholding; we don't have that file (see PROJECT_SUMMARY sec 6), so
the last 15% of x70_all.csv is held out as a stand-in. Architecture, IF
fusion, and the fault-identification idea are also unchanged.
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
from data_loader import build_scaler
from feature_engineering import compute_window_summary
from networks.lstm_ae import LSTMAutoencoder
from eval import compute_binary_metrics
from eval_lstm_ae import compute_window_residuals, fused_score_batch
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import (
    prepare_flight, prepare_flight_file, build_windows,
    window_length, stride_length, window_end_times,
)

INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL
HOLDOUT_FRAC = 0.15

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_lstm_ae_heldout")
MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth")
IFOREST_PATH = os.path.join(Config.BASE_DIR, "iforest_model_heldout.joblib")
STATS_PATH = os.path.join(Config.BASE_DIR, "lstm_ae_stats_heldout.npz")
FEATURE_SCALER_PATH = os.path.join(Config.BASE_DIR, "lstm_ae_feature_scaler_heldout.joblib")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(Config.RANDOM_SEED)
    print("device:", device)

    window = window_length()
    stride = stride_length()
    eval_stride = stride_length(stride_sec=Config.EVAL_STRIDE_SEC)
    print(f"pipeline v4: resample {Config.RESAMPLE_HZ:.0f}Hz, window {Config.WINDOW_SEC}s "
          f"({window} samples), per-flight normalization")

    # --- split the training flight temporally BEFORE anything is fit ---
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    raw_all, time_all = read_raw_csv_with_timestamp(train_csv)
    split_idx = int(len(raw_all) * (1 - HOLDOUT_FRAC))
    raw_train, raw_heldout = raw_all[:split_idx], raw_all[split_idx:]
    time_train = time_all[:split_idx]
    time_heldout = time_all[split_idx:] - time_all[split_idx]
    print(f"train {time_train[-1]:.1f}s  held-out normal {time_heldout[-1]:.1f}s (real seconds)")

    # Each unit is normalized by its own statistics -- the held-out stretch
    # and every test sortie get the same treatment, so none of them carries a
    # session offset into the model.
    feat_train_raw, _, _ = prepare_flight(raw_train, time_train)
    feat_heldout_raw, time_ho_axis, _ = prepare_flight(raw_heldout, time_heldout)

    # Second-stage scaler on the engineered-feature space: derivatives and
    # energy sit on wildly different scales from the raw standardized
    # channels, which otherwise dominates the MSE loss.
    feature_scaler = build_scaler()
    feature_scaler.fit(feat_train_raw)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)

    feat_train = feature_scaler.transform(feat_train_raw)
    feat_heldout = feature_scaler.transform(feat_heldout_raw)

    train_windows = build_windows(feat_train, window, stride)
    print(f"training windows: {len(train_windows)} (window={window} samples = {Config.WINDOW_SEC}s, "
          f"stride={stride} = {Config.STRIDE_SEC}s)")

    train_tensor = torch.tensor(train_windows)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor),
                              batch_size=Config.LSTM_AE_BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
        num_layers=Config.LSTM_AE_NUM_LAYERS,
        bottleneck_dim=Config.LSTM_AE_BOTTLENECK_DIM,
    ).to(device)
    print(f"LSTM-AE parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=Config.LSTM_AE_LEARNING_RATE)
    loss_fn = nn.MSELoss()
    for epoch in range(Config.LSTM_AE_EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        if epoch % 5 == 0 or epoch == Config.LSTM_AE_EPOCHS - 1:
            print(f"epoch {epoch+1}/{Config.LSTM_AE_EPOCHS}  mse={total_loss/n:.5f}")
    torch.save(model.state_dict(), MODEL_PATH)

    # --- Isolation Forest + fusion stats on the training portion ---
    model.eval()
    summary_rows, all_residuals = [], []
    with torch.no_grad():
        for i in range(0, len(train_tensor), 512):
            batch = train_tensor[i:i + 512]
            recon = model(batch.to(device)).cpu().numpy()
            target = batch.numpy()
            residual = np.abs(recon - target)
            all_residuals.append(residual)
            for w_orig, w_resid in zip(target, residual):
                summary_rows.append(np.concatenate([compute_window_summary(w_orig),
                                                     compute_window_summary(w_resid)]))
    if_features = np.stack(summary_rows, axis=0)

    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS,
                              random_state=Config.RANDOM_SEED, n_jobs=-1)
    iforest.fit(if_features)
    joblib.dump(iforest, IFOREST_PATH)

    residuals = np.concatenate(all_residuals, axis=0)
    channel_mean = residuals.mean(axis=(0, 1))
    channel_std = np.where(residuals.std(axis=(0, 1)) == 0, 1e-6, residuals.std(axis=(0, 1)))
    if_scores = -iforest.score_samples(if_features)
    if_mean, if_std = if_scores.mean(), if_scores.std() or 1e-6
    np.savez(STATS_PATH, channel_mean=channel_mean, channel_std=channel_std,
             if_mean=if_mean, if_std=if_std)
    print("saved model/IF/stats (v4 pipeline)")

    def score_windows(feats):
        residuals_b, windows_b = compute_window_residuals(model, feats, device,
                                                          window=window, stride=eval_stride)
        if residuals_b is None:
            return None
        return fused_score_batch(residuals_b, windows_b, iforest,
                                 channel_mean, channel_std, if_mean, if_std)

    # --- threshold from held-out normal blocks (never trained on) ---
    print("\n=== Detection evaluation (held-out normal threshold, v4 pipeline) ===")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    n_blocks = len(test_files)
    bounds = np.linspace(0, len(feat_heldout), n_blocks + 1, dtype=int)

    normal_scores, rows = [], []
    for i in range(n_blocks):
        scores = score_windows(feat_heldout[bounds[i]:bounds[i + 1]])
        if scores is None or len(scores) == 0:
            continue
        score = float(scores.max())
        normal_scores.append(score)
        rows.append({"file": f"heldout_normal_block_{i+1:02d}", "true_label": "normal", "score": score})
    print(f"held-out normal blocks yielding >=1 window: {len(normal_scores)}/{n_blocks} "
          f"(held-out is only {time_heldout[-1]:.1f}s, so blocks are ~{time_heldout[-1]/n_blocks:.2f}s each)")

    threshold = max(normal_scores) if normal_scores else 0.0
    y_true = [0] * len(normal_scores)
    scores_all = list(normal_scores)

    for f in test_files:
        feats_raw, _, _ = prepare_flight_file(f)
        scores = score_windows(feature_scaler.transform(feats_raw))
        if scores is None or len(scores) == 0:
            print(f"[{os.path.basename(f)}] too short for one {Config.WINDOW_SEC}s window, skipped")
            continue
        score = float(scores.max())
        predicted = "abnormal" if score > threshold else "normal"
        rows.append({"file": os.path.basename(f), "true_label": "abnormal",
                     "predicted_label": predicted, "score": score})
        y_true.append(1)
        scores_all.append(score)
        print(f"[{os.path.basename(f)}] score={score:.4f} pred={predicted}")

    for r in rows:
        r.setdefault("predicted_label", "normal")

    metrics = compute_binary_metrics(y_true, scores_all, threshold)
    print(f"\nThreshold (held-out normal): {threshold:.4f}")
    for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score"]:
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Confusion matrix: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "lstm_ae_sortie_scores.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "true_label", "predicted_label", "score"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "lstm_ae_metrics.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN"]:
            writer.writerow([name, metrics[name]])
    print(f"saved results to {RESULTS_DIR}/")

    # --- channel diagnosis, now on the REAL time axis (see Config.BURST_REAL_SEC) ---
    print("\n=== Channel diagnosis: file-wide own-baseline (real seconds) ===")
    target = os.path.join(Config.TEST_DATA_PATH, Config.BURST_FILE)
    feats_raw, time_axis, _ = prepare_flight_file(target)
    feats = feature_scaler.transform(feats_raw)
    residuals_t, windows_t = compute_window_residuals(model, feats, device, window=window, stride=1)
    per_window_channel_err = residuals_t.mean(axis=1).reshape(
        len(residuals_t), Config.NUM_CHANNELS, Config.LSTM_AE_FEATURES_PER_CHANNEL).mean(axis=2)
    wt = window_end_times(time_axis, len(per_window_channel_err), window=window, stride=1)

    lo, hi = Config.BURST_REAL_SEC
    burst_mask = (wt >= lo) & (wt <= hi)
    file_baseline = per_window_channel_err[~burst_mask].mean(axis=0)

    def report(mask, label):
        """Report the EXCESS values, not just ranks. Most channels sit at or
        below the file baseline in a normal stretch, so their excess clips to
        exactly 0 -- and a rank among all-zero ties is a sort artifact, not a
        finding. 'mag_x excess = 0' is the statement that actually means
        something."""
        excess = np.maximum(per_window_channel_err[mask].mean(axis=0) - file_baseline, 0.0)
        n_nonzero = int((excess > 0).sum())
        top = int(np.argmax(excess)) + 1
        verdict = f"ch11 excess={excess[10]:.5f}"
        if n_nonzero == 0:
            verdict += "  (no channel above baseline at all)"
        else:
            verdict += f"  top channel=ch{top} ({excess.max():.5f})"
        print(f"  {label:26s} nonzero-excess {n_nonzero:2d}/16   {verdict}")

    report(burst_mask, f"burst {lo}-{hi}s (REAL fault)")
    for lo_c, hi_c in Config.CONTROL_REAL_SEC:
        mask = (wt >= lo_c) & (wt <= hi_c)
        if mask.any():
            report(mask, f"control {lo_c}-{hi_c}s")


if __name__ == "__main__":
    main()
