"""
Evaluation for the LSTM-AE + Isolation Forest baseline (v2: engineered
features + compact IF summary stats, per the paper reread).
  1) Full detection evaluation across all 19 test sorties.
  2) Channel-level diagnosis check on the known mag_x fault burst using the
     file-wide own-baseline trick, aggregating the 4 engineered features
     per channel back down to one score per physical channel (16 total) so
     it's comparable to the VAE/GRU results.
"""
import glob
import os

import joblib
import numpy as np
import torch

from config import Config
from lstm_ae_data_loader import LSTMAEDataset
from gru_data_loader import load_scaled_csv_with_time
from feature_engineering import compute_sequence_features, compute_window_summary
from networks.lstm_ae import LSTMAutoencoder
from eval import compute_binary_metrics

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_lstm_ae")
INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL


def load_models(device):
    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
        num_layers=Config.LSTM_AE_NUM_LAYERS,
        bottleneck_dim=Config.LSTM_AE_BOTTLENECK_DIM,
    ).to(device)
    model.load_state_dict(torch.load(Config.LSTM_AE_MODEL_SAVE_PATH, map_location=device))
    model.eval()

    iforest = joblib.load(Config.IFOREST_MODEL_SAVE_PATH)
    stats = np.load(Config.LSTM_AE_STATS_PATH)
    return model, iforest, stats["channel_mean"], stats["channel_std"], float(stats["if_mean"]), float(stats["if_std"])


def compute_window_residuals(model, feat_seq, device, window=None, stride=1):
    window = window or Config.LSTM_AE_WINDOW
    windows = []
    for i in range(0, len(feat_seq) - window + 1, stride):
        windows.append(feat_seq[i:i + window])
    if not windows:
        return None, None
    windows = np.asarray(windows, dtype=np.float32)
    windows_t = torch.tensor(windows)

    recons = []
    with torch.no_grad():
        for i in range(0, len(windows_t), 512):
            batch = windows_t[i:i + 512].to(device)
            recons.append(model(batch).cpu().numpy())
    recons = np.concatenate(recons, axis=0)
    residuals = np.abs(recons - windows)  # (N, window, input_dim)
    return residuals, windows


def fused_score_batch(residuals, windows, iforest, channel_mean, channel_std, if_mean, if_std):
    lstm_err = residuals.mean(axis=(1, 2))
    lstm_norm = (lstm_err - channel_mean.mean()) / channel_std.mean()

    if_features = np.stack([
        np.concatenate([compute_window_summary(w), compute_window_summary(r)])
        for w, r in zip(windows, residuals)
    ], axis=0)
    if_raw = -iforest.score_samples(if_features)
    if_norm = (if_raw - if_mean) / if_std

    return Config.LSTM_AE_FUSION_ALPHA * lstm_norm + (1 - Config.LSTM_AE_FUSION_ALPHA) * if_norm


def evaluate_detection(model, iforest, scaler, feature_scaler, channel_mean, channel_std, if_mean, if_std, device):
    print("\n=== Full-sortie detection evaluation (19 test files) ===")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    scaled_train, _, time_train = load_scaled_csv_with_time(train_csv, scaler=scaler)
    feat_train = feature_scaler.transform(compute_sequence_features(scaled_train, time_train))
    n_blocks = len(test_files)
    bounds = np.linspace(0, len(feat_train), n_blocks + 1, dtype=int)

    normal_scores = []
    rows = []
    for i in range(n_blocks):
        block = feat_train[bounds[i]:bounds[i + 1]]
        residuals, windows = compute_window_residuals(model, block, device, stride=Config.LSTM_AE_STRIDE)
        if residuals is None:
            continue
        scores = fused_score_batch(residuals, windows, iforest, channel_mean, channel_std, if_mean, if_std)
        score = float(scores.max())
        normal_scores.append(score)
        rows.append({"file": f"normal_block_{i+1:02d}", "true_label": "normal", "score": score})

    threshold = max(normal_scores) if normal_scores else 0.0
    y_true = [0] * len(normal_scores)
    scores_all = list(normal_scores)

    for f in test_files:
        scaled, _, time_sec = load_scaled_csv_with_time(f, scaler=scaler)
        feats = feature_scaler.transform(compute_sequence_features(scaled, time_sec))
        residuals, windows = compute_window_residuals(model, feats, device, stride=Config.LSTM_AE_STRIDE)
        scores = fused_score_batch(residuals, windows, iforest, channel_mean, channel_std, if_mean, if_std)
        score = float(scores.max())
        predicted = "abnormal" if score > threshold else "normal"
        rows.append({"file": os.path.basename(f), "true_label": "abnormal", "predicted_label": predicted, "score": score})
        y_true.append(1)
        scores_all.append(score)
        print(f"[{os.path.basename(f)}] score={score:.4f} pred={predicted}")

    for r in rows:
        r.setdefault("predicted_label", "normal")

    metrics = compute_binary_metrics(y_true, scores_all, threshold)
    print(f"\nThreshold: {threshold:.4f}")
    for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score"]:
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Confusion matrix: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    import csv
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


def evaluate_channel_diagnosis(model, scaler, feature_scaler, device):
    print("\n=== Channel diagnosis: file-wide own-baseline check ===")
    target_file = os.path.join(Config.TEST_DATA_PATH, "x70_20180611_x76_070620.csv")
    scaled, _, real_time_sec = load_scaled_csv_with_time(target_file, scaler=scaler)
    feats = feature_scaler.transform(compute_sequence_features(scaled, real_time_sec))
    residuals, windows = compute_window_residuals(model, feats, device, stride=1)
    time_sec = (np.arange(len(residuals)) + Config.LSTM_AE_WINDOW) / Config.DATA_HZ

    # (N, window, C*4) -> per-window, per-channel error by averaging the 4
    # engineered features and the window's timesteps back down to 1 value/channel.
    per_window_feat_err = residuals.mean(axis=1)  # (N, C*4)
    per_window_channel_err = per_window_feat_err.reshape(len(per_window_feat_err), Config.NUM_CHANNELS,
                                                           Config.LSTM_AE_FEATURES_PER_CHANNEL).mean(axis=2)

    burst_idx = np.where((time_sec >= 1043) & (time_sec <= 1063))[0]
    rest_idx = np.where((time_sec < 1043) | (time_sec > 1063))[0]
    file_baseline = per_window_channel_err[rest_idx].mean(axis=0)
    burst_ce = per_window_channel_err[burst_idx].mean(axis=0)
    excess = np.maximum(burst_ce - file_baseline, 0.0)

    ranking = np.argsort(-excess) + 1
    ch11_rank = int(np.where(ranking == 11)[0][0]) + 1
    print(f"burst window (1043-1063s): top5={ranking[:5].tolist()} ch11_rank={ch11_rank}")

    for lo, hi in [(200, 230), (500, 530), (900, 930)]:
        idx = np.where((time_sec >= lo) & (time_sec <= hi))[0]
        ce = per_window_channel_err[idx].mean(axis=0)
        ctrl_excess = np.maximum(ce - file_baseline, 0.0)
        ctrl_ranking = np.argsort(-ctrl_excess) + 1
        ctrl_ch11 = int(np.where(ctrl_ranking == 11)[0][0]) + 1
        print(f"control {lo}-{hi}s: top5={ctrl_ranking[:5].tolist()} ch11_rank={ctrl_ch11}")


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    feature_scaler = joblib.load(Config.LSTM_AE_FEATURE_SCALER_PATH)
    train_dataset = LSTMAEDataset(train_csv, feature_scaler=feature_scaler)
    scaler = train_dataset.scaler

    model, iforest, channel_mean, channel_std, if_mean, if_std = load_models(device)

    evaluate_detection(model, iforest, scaler, feature_scaler, channel_mean, channel_std, if_mean, if_std, device)
    evaluate_channel_diagnosis(model, scaler, feature_scaler, device)


if __name__ == "__main__":
    evaluate_all()
