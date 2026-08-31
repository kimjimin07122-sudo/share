"""
Training for the LSTM-AE + Isolation Forest baseline (UAV on-device plan).
Two-stage: 1) train the LSTM-AE on engineered per-timestep features
(raw/deriv/deriv2/energy), 2) fit Isolation Forest on COMPACT per-window
summary statistics of both the original features and the LSTM-AE's own
reconstruction residual -- matching the paper's "trained on both original
and residual features" (Section 4.4) and keeping IF's input small (this is
what fixed the earlier IF-is-the-bottleneck problem: 512-dim raw flattened
windows -> ~128-dim summary stats).
"""
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

from config import Config
from lstm_ae_data_loader import LSTMAEDataset
from feature_engineering import compute_window_summary
from networks.lstm_ae import LSTMAutoencoder

INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    train_dataset = LSTMAEDataset(train_csv)
    train_loader = DataLoader(train_dataset, batch_size=Config.LSTM_AE_BATCH_SIZE, shuffle=True)
    print(f"training windows: {len(train_dataset)} (window={Config.LSTM_AE_WINDOW}, stride={Config.LSTM_AE_STRIDE}, "
          f"input_dim={INPUT_DIM})")
    joblib.dump(train_dataset.feature_scaler, Config.LSTM_AE_FEATURE_SCALER_PATH)
    print(f"saved feature scaler to {Config.LSTM_AE_FEATURE_SCALER_PATH}")

    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
        num_layers=Config.LSTM_AE_NUM_LAYERS,
        bottleneck_dim=Config.LSTM_AE_BOTTLENECK_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM-AE parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=Config.LSTM_AE_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(Config.LSTM_AE_EPOCHS):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        if epoch % 5 == 0 or epoch == Config.LSTM_AE_EPOCHS - 1:
            print(f"epoch {epoch+1}/{Config.LSTM_AE_EPOCHS}  mse={total_loss/n:.5f}")

    torch.save(model.state_dict(), Config.LSTM_AE_MODEL_SAVE_PATH)
    print(f"saved LSTM-AE to {Config.LSTM_AE_MODEL_SAVE_PATH}")

    # --- Build compact (original + residual) summary features for Isolation Forest ---
    print("computing per-window summary features (original + residual) for Isolation Forest...")
    model.eval()
    eval_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    summary_rows = []
    all_residuals_for_stats = []
    with torch.no_grad():
        for xb, yb in eval_loader:
            recon = model(xb.to(device)).cpu().numpy()
            target = yb.numpy()
            residual = np.abs(recon - target)
            all_residuals_for_stats.append(residual)
            for w_orig, w_resid in zip(target, residual):
                orig_summary = compute_window_summary(w_orig)
                resid_summary = compute_window_summary(w_resid)
                summary_rows.append(np.concatenate([orig_summary, resid_summary]))
    if_features = np.stack(summary_rows, axis=0)
    print(f"Isolation Forest input dim: {if_features.shape[1]} (was 512 with raw flattened windows)")

    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS, random_state=Config.RANDOM_SEED, n_jobs=-1)
    iforest.fit(if_features)
    joblib.dump(iforest, Config.IFOREST_MODEL_SAVE_PATH)
    print(f"saved Isolation Forest to {Config.IFOREST_MODEL_SAVE_PATH}")

    # --- Residual stats (per-feature, for normalizing LSTM-AE reconstruction error) ---
    residuals = np.concatenate(all_residuals_for_stats, axis=0)  # (N, window, input_dim)
    channel_mean = residuals.mean(axis=(0, 1))
    channel_std = residuals.std(axis=(0, 1))
    channel_std = np.where(channel_std == 0, 1e-6, channel_std)

    if_scores = -iforest.score_samples(if_features)
    if_mean, if_std = if_scores.mean(), if_scores.std() or 1e-6

    np.savez(
        Config.LSTM_AE_STATS_PATH,
        channel_mean=channel_mean,
        channel_std=channel_std,
        if_mean=if_mean,
        if_std=if_std,
    )
    print(f"saved stats to {Config.LSTM_AE_STATS_PATH}")


if __name__ == "__main__":
    train()
