"""
Data loading for the LSTM predictor (forecasting) baseline.

Unlike lstm_ae_data_loader.py (window -> same window, autoencoding), this
yields (past_window -> next_step) pairs, built on the SAME feature-engineered
sequence (raw/deriv/deriv2/energy) as the LSTM-AE so the surrounding
pipeline (IF summary stats, fusion, file-wide-baseline channel diagnosis)
needs no changes -- only the "residual" now means prediction error instead
of reconstruction error.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config
from gru_data_loader import load_scaled_csv_with_time
from feature_engineering import compute_sequence_features
from data_loader import StandardScaler


class LSTMPredictorDataset(Dataset):
    """Yields (past_window, next_step) pairs of engineered features."""

    def __init__(self, data_source, scaler=None, feature_scaler=None, window=None, stride=None):
        self.window = window or Config.LSTM_AE_WINDOW
        self.stride = stride or Config.LSTM_AE_STRIDE

        if os.path.isdir(data_source):
            files = sorted(glob.glob(os.path.join(data_source, "*.csv")))
        else:
            files = [data_source]

        raw_feature_sequences = []
        for f in files:
            scaled, scaler, time_sec = load_scaled_csv_with_time(f, scaler=scaler)
            raw_feature_sequences.append(compute_sequence_features(scaled, time_sec))
        self.scaler = scaler

        if feature_scaler is None:
            feature_scaler = StandardScaler()
            feature_scaler.fit(np.vstack(raw_feature_sequences))
        self.feature_scaler = feature_scaler
        feature_sequences = [feature_scaler.transform(f) for f in raw_feature_sequences]

        pasts, nexts = [], []
        for feats in feature_sequences:
            for i in range(0, len(feats) - self.window, self.stride):
                pasts.append(feats[i:i + self.window])
                nexts.append(feats[i + self.window])
        if not pasts:
            raise ValueError(f"No windows could be built from {data_source}.")

        self.pasts = np.asarray(pasts, dtype=np.float32)
        self.nexts = np.asarray(nexts, dtype=np.float32)

    def __len__(self):
        return len(self.pasts)

    def __getitem__(self, idx):
        return torch.tensor(self.pasts[idx]), torch.tensor(self.nexts[idx])
