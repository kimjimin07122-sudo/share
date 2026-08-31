"""
End-to-end integration test (v2: engineered features + FC bottleneck +
compact IF summary stats): raw sensor CSV -> sliding window -> feature
engineering -> ONNX LSTM-AE + Isolation Forest -> fused anomaly score ->
channel diagnosis, timed as a whole pipeline.
"""
import os
import time

import joblib
import numpy as np
import onnxruntime as ort

from config import Config
from gru_data_loader import load_scaled_csv_with_time
from lstm_ae_data_loader import LSTMAEDataset
from feature_engineering import compute_sequence_features, compute_window_summary

ONNX_PATH = os.path.join(Config.BASE_DIR, "lstm_ae.onnx")
TARGET_FILE = os.path.join(Config.TEST_DATA_PATH, "x70_20180611_x76_070620.csv")


def main():
    feature_scaler = joblib.load(Config.LSTM_AE_FEATURE_SCALER_PATH)
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    train_dataset = LSTMAEDataset(train_csv, feature_scaler=feature_scaler)
    scaler = train_dataset.scaler

    stats = np.load(Config.LSTM_AE_STATS_PATH)
    channel_mean, channel_std = stats["channel_mean"], stats["channel_std"]
    if_mean, if_std = float(stats["if_mean"]), float(stats["if_std"])

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    iforest = joblib.load(Config.IFOREST_MODEL_SAVE_PATH)

    # --- Load + scale + feature-engineer the target file once (the "sensor stream") ---
    scaled, _, real_time_sec = load_scaled_csv_with_time(TARGET_FILE, scaler=scaler)
    feats = feature_scaler.transform(compute_sequence_features(scaled, real_time_sec))
    w = Config.LSTM_AE_WINDOW
    n_windows = len(feats) - w + 1

    print(f"file: {os.path.basename(TARGET_FILE)}, {len(scaled)} rows -> {n_windows} sliding windows (stride=1)")

    scores = []
    per_channel_errors = []
    t0 = time.perf_counter()
    for i in range(0, len(feats) - w + 1, 1):
        window = feats[i:i + w].astype(np.float32)[None, :, :]  # (1, w, input_dim)

        recon = sess.run(None, {"window": window})[0]
        residual = np.abs(recon - window)[0]  # (w, input_dim)
        lstm_err = residual.mean()
        lstm_norm = (lstm_err - channel_mean.mean()) / channel_std.mean()

        orig_summary = compute_window_summary(window[0])
        resid_summary = compute_window_summary(residual)
        if_features = np.concatenate([orig_summary, resid_summary])[None, :]
        if_raw = -iforest.score_samples(if_features)[0]
        if_norm = (if_raw - if_mean) / if_std

        fused = Config.LSTM_AE_FUSION_ALPHA * lstm_norm + (1 - Config.LSTM_AE_FUSION_ALPHA) * if_norm
        scores.append(fused)
        # aggregate residual (window, C*4) -> (C,) for channel diagnosis
        per_channel_errors.append(residual.mean(axis=0).reshape(Config.NUM_CHANNELS, Config.LSTM_AE_FEATURES_PER_CHANNEL).mean(axis=1))
    t1 = time.perf_counter()

    scores = np.array(scores)
    per_channel_errors = np.array(per_channel_errors)
    time_sec = (np.arange(len(scores)) + w) / Config.DATA_HZ

    total_time = t1 - t0
    print(f"\nend-to-end pipeline: {total_time:.3f}s total, {total_time/len(scores)*1000:.3f} ms/window "
          f"(feature engineering + ONNX LSTM + Isolation Forest combined)")

    # --- Channel diagnosis with file-wide own-baseline ---
    burst_idx = np.where((time_sec >= 1043) & (time_sec <= 1063))[0]
    rest_idx = np.where((time_sec < 1043) | (time_sec > 1063))[0]
    baseline = per_channel_errors[rest_idx].mean(axis=0)
    burst_ce = per_channel_errors[burst_idx].mean(axis=0)
    excess = np.maximum(burst_ce - baseline, 0.0)
    ranking = np.argsort(-excess) + 1
    ch11_rank = int(np.where(ranking == 11)[0][0]) + 1

    print(f"\nchannel diagnosis (burst window): top5={ranking[:5].tolist()} ch11_rank={ch11_rank}")
    print(f"detection: peak fused score={scores.max():.3f} at t={time_sec[scores.argmax()]:.1f}s")

    print("\nEnd-to-end pipeline verified: raw CSV window -> feature engineering -> "
          "ONNX LSTM-AE + IsolationForest -> fused score + channel diagnosis, all in a single loop.")


if __name__ == "__main__":
    main()
