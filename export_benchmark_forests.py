"""
Fit one IsolationForest per tree count and ship them all (2026-08-23).

The tree-count sweep in experiment_iforest_lightweight.py ran on the x86 host
and estimated the board's latency by scaling with a measured ratio. Estimates
were close -- 4.4 ms predicted against 4.39 ms measured -- but a presentation
is better served by numbers the board produced itself.

So this exports a forest at each count, together with the fusion statistics
that belong to it (if_mean / if_std are properties of that specific forest and
cannot be shared across counts). benchmark_iforest_onboard.py then runs the
real streaming pipeline against each one on the Jetson and reports measured
latency, so the comparison needs no scaling assumption at all.
"""
import json
import os

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from config import Config
from feature_engineering import compute_window_summary
from networks.lstm_ae import LSTMAutoencoder
from eval_lstm_ae import compute_window_residuals
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import prepare_flight, build_windows, window_length, stride_length

TREE_COUNTS = [100, 50, 30, 20, 10, 5]
HOLDOUT_FRAC = 0.15
OUT = os.path.join(Config.BASE_DIR, "deploy", "benchmark_forests")


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = window_length()

    fscaler = joblib.load(os.path.join(Config.BASE_DIR, "lstm_ae_feature_scaler_heldout.joblib"))
    model = LSTMAutoencoder(Config.NUM_CHANNELS * 4, Config.LSTM_AE_HIDDEN_SIZE,
                            Config.LSTM_AE_NUM_LAYERS, Config.LSTM_AE_BOTTLENECK_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth"),
                                     map_location=device))
    model.eval()

    raw, t = read_raw_csv_with_timestamp(os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv"))
    split = int(len(raw) * (1 - HOLDOUT_FRAC))
    feat, _, _ = prepare_flight(raw[:split], t[:split])
    feat = fscaler.transform(feat)
    wins = build_windows(feat, W, stride_length())
    res, _ = compute_window_residuals(model, feat, device, window=W, stride=stride_length())
    if_feat = np.stack([np.concatenate([compute_window_summary(a), compute_window_summary(b)])
                        for a, b in zip(wins, res)], axis=0)
    print(f"IF training features: {if_feat.shape}")

    manifest = {}
    for n in TREE_COUNTS:
        f = IsolationForest(n_estimators=n, random_state=Config.RANDOM_SEED, n_jobs=-1).fit(if_feat)
        s = -f.score_samples(if_feat)
        path = os.path.join(OUT, f"iforest_{n:03d}.joblib")
        joblib.dump(f, path)
        manifest[str(n)] = {"file": os.path.basename(path),
                            "if_mean": float(s.mean()), "if_std": float(s.std() or 1e-6),
                            "size_kb": round(os.path.getsize(path) / 1024, 1)}
        print(f"  {n:3d} trees -> {os.path.basename(path)}  {manifest[str(n)]['size_kb']:7.1f} KB")

    # channel-error statistics are shared: they come from the LSTM-AE, not the forest
    np.savez(os.path.join(OUT, "shared_error_stats.npz"),
             channel_mean=res.mean(axis=(0, 1)),
             channel_std=np.where(res.std(axis=(0, 1)) == 0, 1e-6, res.std(axis=(0, 1))))
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(m["size_kb"] for m in manifest.values())
    print(f"\nwrote {len(manifest)} forests + shared stats to deploy/benchmark_forests/ "
          f"({total:.0f} KB total)")


if __name__ == "__main__":
    main()
