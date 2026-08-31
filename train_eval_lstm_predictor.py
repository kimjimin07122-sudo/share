"""
Prediction (forecasting) variant of the on-device model: instead of the
LSTM-AE reconstructing its own input window, an LSTM predictor reads a past
window of engineered features and forecasts the single next timestep, and
the forecast error (instead of reconstruction error) drives the same
IF/fusion/channel-diagnosis pipeline used for the LSTM-AE (v3, held-out
split). Kept as a separate script (doesn't touch train_eval_lstm_ae_heldout.py)
so both variants can be compared side by side.
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
from networks.lstm_predictor import LSTMPredictor
from eval import compute_binary_metrics
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import (
    prepare_flight, prepare_flight_file, window_length, stride_length, window_end_times,
)

INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL
HOLDOUT_FRAC = 0.15
WINDOW = window_length()
STRIDE = stride_length()

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_lstm_predictor")
MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_predictor_model.pth")
IFOREST_PATH = os.path.join(Config.BASE_DIR, "iforest_model_predictor.joblib")
STATS_PATH = os.path.join(Config.BASE_DIR, "lstm_predictor_stats.npz")
FEATURE_SCALER_PATH = os.path.join(Config.BASE_DIR, "lstm_predictor_feature_scaler.joblib")


def build_pairs(feats, window, stride):
    """(T, D) feature sequence -> (past[N,window,D], next[N,D]) pairs."""
    pasts, nexts = [], []
    for i in range(0, len(feats) - window, stride):
        pasts.append(feats[i:i + window])
        nexts.append(feats[i + window])
    return np.asarray(pasts, dtype=np.float32), np.asarray(nexts, dtype=np.float32)


def predict_full_sequence(model, feats, device, window, stride=1):
    """Scan the whole sequence and predict every valid next-step (stride=1
    for precise channel diagnosis, larger stride for faster block scoring).
    Returns (pred_errors[M,D], context_windows[M,window,D]) where entry k
    corresponds to predicting feats[k*stride+window] from feats[k*stride:k*stride+window]."""
    pasts, nexts = build_pairs(feats, window, stride)
    if len(pasts) == 0:
        return None, None
    pasts_t = torch.tensor(pasts)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pasts_t), 512):
            batch = pasts_t[i:i + 512].to(device)
            preds.append(model(batch).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    errors = np.abs(preds - nexts)
    return errors, pasts


def fused_score_batch_pred(pred_errors, context_windows, iforest, channel_mean, channel_std, if_mean, if_std):
    lstm_err = pred_errors.mean(axis=1)
    lstm_norm = (lstm_err - channel_mean.mean()) / channel_std.mean()

    if_features = np.stack([
        np.concatenate([compute_window_summary(w), r])
        for w, r in zip(context_windows, pred_errors)
    ], axis=0)
    if_raw = -iforest.score_samples(if_features)
    if_norm = (if_raw - if_mean) / if_std

    return Config.LSTM_AE_FUSION_ALPHA * lstm_norm + (1 - Config.LSTM_AE_FUSION_ALPHA) * if_norm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    raw_all, time_all = read_raw_csv_with_timestamp(train_csv)

    split_idx = int(len(raw_all) * (1 - HOLDOUT_FRAC))
    raw_train, raw_heldout = raw_all[:split_idx], raw_all[split_idx:]
    time_train = time_all[:split_idx]
    time_heldout = time_all[split_idx:] - time_all[split_idx]
    print(f"pipeline v4: resample {Config.RESAMPLE_HZ:.0f}Hz, window {Config.WINDOW_SEC}s "
          f"({WINDOW} samples), per-flight normalization")
    print(f"train {time_train[-1]:.1f}s  held-out normal {time_heldout[-1]:.1f}s (real seconds)")

    feat_train_raw, _, _ = prepare_flight(raw_train, time_train)
    feat_heldout_raw, _, _ = prepare_flight(raw_heldout, time_heldout)

    feature_scaler = build_scaler()
    feature_scaler.fit(feat_train_raw)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)

    feat_train = feature_scaler.transform(feat_train_raw)

    train_pasts, train_nexts = build_pairs(feat_train, WINDOW, STRIDE)
    print(f"training pairs: {len(train_pasts)} (window={WINDOW}, stride={STRIDE})")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_pasts), torch.tensor(train_nexts)),
        batch_size=Config.LSTM_AE_BATCH_SIZE, shuffle=True,
    )

    model = LSTMPredictor(input_dim=INPUT_DIM, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                           num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM predictor parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=Config.LSTM_AE_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(Config.LSTM_AE_EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        if epoch % 5 == 0 or epoch == Config.LSTM_AE_EPOCHS - 1:
            print(f"epoch {epoch+1}/{Config.LSTM_AE_EPOCHS}  mse={total_loss/n:.5f}")

    torch.save(model.state_dict(), MODEL_PATH)

    # --- Isolation Forest + residual stats on TRAIN portion ---
    print("computing prediction errors on training data for IF fit...")
    pred_errors, context_windows = predict_full_sequence(model, feat_train, device, WINDOW, stride=STRIDE)

    summary_rows = [
        np.concatenate([compute_window_summary(w), r])
        for w, r in zip(context_windows, pred_errors)
    ]
    if_features = np.stack(summary_rows, axis=0)

    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS, random_state=Config.RANDOM_SEED, n_jobs=-1)
    iforest.fit(if_features)
    joblib.dump(iforest, IFOREST_PATH)

    channel_mean = pred_errors.mean(axis=0)
    channel_std = pred_errors.std(axis=0)
    channel_std = np.where(channel_std == 0, 1e-6, channel_std)
    if_scores = -iforest.score_samples(if_features)
    if_mean, if_std = if_scores.mean(), if_scores.std() or 1e-6

    np.savez(STATS_PATH, channel_mean=channel_mean, channel_std=channel_std, if_mean=if_mean, if_std=if_std)
    print("saved model/IF/stats (predictor variant)")

    # --- evaluation: threshold from HELD-OUT normal blocks ---
    print("\n=== Detection evaluation (held-out normal threshold, prediction error) ===")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    n_blocks = len(test_files)

    feat_heldout = feature_scaler.transform(feat_heldout_raw)
    bounds = np.linspace(0, len(feat_heldout), n_blocks + 1, dtype=int)

    normal_scores = []
    rows = []
    for i in range(n_blocks):
        block = feat_heldout[bounds[i]:bounds[i + 1]]
        errs, ctx = predict_full_sequence(model, block, device, WINDOW, stride=STRIDE)
        if errs is None:
            continue
        scores = fused_score_batch_pred(errs, ctx, iforest, channel_mean, channel_std, if_mean, if_std)
        score = float(scores.max())
        normal_scores.append(score)
        rows.append({"file": f"heldout_normal_block_{i+1:02d}", "true_label": "normal", "score": score})

    threshold = max(normal_scores) if normal_scores else 0.0
    y_true = [0] * len(normal_scores)
    scores_all = list(normal_scores)

    for f in test_files:
        feats_raw, _, _ = prepare_flight_file(f)
        feats = feature_scaler.transform(feats_raw)
        errs, ctx = predict_full_sequence(model, feats, device, WINDOW, stride=STRIDE)
        if errs is None:
            print(f"[{os.path.basename(f)}] too short for one window, skipped")
            continue
        scores = fused_score_batch_pred(errs, ctx, iforest, channel_mean, channel_std, if_mean, if_std)
        score = float(scores.max())
        predicted = "abnormal" if score > threshold else "normal"
        rows.append({"file": os.path.basename(f), "true_label": "abnormal", "predicted_label": predicted, "score": score})
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
    with open(os.path.join(RESULTS_DIR, "lstm_predictor_sortie_scores.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "true_label", "predicted_label", "score"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "lstm_predictor_metrics.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN"]:
            writer.writerow([name, metrics[name]])
    print(f"saved results to {RESULTS_DIR}/")

    # --- channel diagnosis: file-wide own-baseline on known burst window ---
    print("\n=== Channel diagnosis (prediction error, file-wide own-baseline) ===")
    target_file = os.path.join(Config.TEST_DATA_PATH, Config.BURST_FILE)
    feats_raw, time_axis, _ = prepare_flight_file(target_file)
    feats = feature_scaler.transform(feats_raw)
    errs, _ctx = predict_full_sequence(model, feats, device, WINDOW, stride=1)
    per_step_channel_err = errs.reshape(len(errs), Config.NUM_CHANNELS, Config.LSTM_AE_FEATURES_PER_CHANNEL).mean(axis=2)

    # real seconds now -- the old 1043-1063s constant was a pseudo-time
    # artifact of row_idx/DATA_HZ (PROJECT_SUMMARY sec 3.8)
    wt = window_end_times(time_axis, len(per_step_channel_err), window=WINDOW, stride=1)
    lo, hi = Config.BURST_REAL_SEC
    burst_mask = (wt >= lo) & (wt <= hi)
    baseline = per_step_channel_err[~burst_mask].mean(axis=0)

    def report(mask, label):
        excess = np.maximum(per_step_channel_err[mask].mean(axis=0) - baseline, 0.0)
        n_nonzero = int((excess > 0).sum())
        verdict = f"ch11 excess={excess[10]:.5f}"
        if n_nonzero == 0:
            verdict += "  (no channel above baseline at all)"
        else:
            verdict += f"  top channel=ch{int(np.argmax(excess)) + 1} ({excess.max():.5f})"
        print(f"  {label:26s} nonzero-excess {n_nonzero:2d}/16   {verdict}")

    report(burst_mask, f"burst {lo}-{hi}s (REAL fault)")
    for lo_c, hi_c in Config.CONTROL_REAL_SEC:
        mask = (wt >= lo_c) & (wt <= hi_c)
        if mask.any():
            report(mask, f"control {lo_c}-{hi_c}s")


if __name__ == "__main__":
    main()
