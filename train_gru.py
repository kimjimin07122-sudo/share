"""
Training loop for the GRU forecasting baseline (TiTAD adoption plan, step 2).
Data loading lives in gru_data_loader.py, model architecture in
networks/gru_predictor.py -- this file only trains and saves residual
stats used later by eval_gru.py for anomaly scoring.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from gru_data_loader import GRUPredictionDataset
from networks.gru_predictor import GRUPredictor


def compute_residual_stats(model, dataset, device):
    """Median/IQR of |pred - target| per channel on training data (TiTAD Eq. 16)."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    residuals = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            residuals.append(np.abs(pred - yb.numpy()))
    residuals = np.concatenate(residuals, axis=0)

    median = np.median(residuals, axis=0)
    q75 = np.percentile(residuals, 75, axis=0)
    q25 = np.percentile(residuals, 25, axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-6, q75 - q25)
    return median, iqr


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    train_dataset = GRUPredictionDataset(train_csv)
    train_loader = DataLoader(train_dataset, batch_size=Config.GRU_BATCH_SIZE, shuffle=True)
    print(f"training pairs: {len(train_dataset)} (window={Config.GRU_WINDOW}, stride={Config.GRU_STRIDE})")

    model = GRUPredictor(
        num_channels=Config.NUM_CHANNELS,
        hidden_size=Config.GRU_HIDDEN_SIZE,
        num_layers=Config.GRU_NUM_LAYERS,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.GRU_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(Config.GRU_EPOCHS):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        if epoch % 5 == 0 or epoch == Config.GRU_EPOCHS - 1:
            print(f"epoch {epoch+1}/{Config.GRU_EPOCHS}  mse={total_loss/n:.5f}")

    torch.save(model.state_dict(), Config.GRU_MODEL_SAVE_PATH)
    print(f"saved model to {Config.GRU_MODEL_SAVE_PATH}")

    median, iqr = compute_residual_stats(model, train_dataset, device)
    np.savez(Config.GRU_RESIDUAL_STATS_PATH, median=median, iqr=iqr)
    print(f"saved residual stats to {Config.GRU_RESIDUAL_STATS_PATH}")
    for i, name in enumerate(Config.CHANNEL_NAMES):
        print(f"  {name:28s} median={median[i]:.4f} iqr={iqr[i]:.4f}")


if __name__ == "__main__":
    train()
