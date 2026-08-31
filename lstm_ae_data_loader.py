"""
Data loading for the LSTM-AE + Isolation Forest baseline.

Reconstruction target = the window itself (unlike gru_data_loader.py, which
builds past-window -> next-step prediction pairs). Reuses load_scaled_csv
from gru_data_loader.py so both baselines share the same scaler logic.

Feature engineering (raw/deriv/deriv2/energy, per feature_engineering.py) is
applied to the whole continuous per-file sequence BEFORE windowing, so
derivatives are computed across true neighbors rather than at window edges.
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


class LSTMAEDataset(Dataset):
    """Yields (window, window) pairs of engineered features -- input equals
    reconstruction target."""

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

        # Engineered features (esp. 2nd derivative / energy) have wildly
        # different scales than the raw standardized channels, which made
        # MSE loss dominated by them and training loss never approach 0.
        # Standardize the C*4 feature space too, fit once on training data.
        if feature_scaler is None:
            feature_scaler = StandardScaler()
            feature_scaler.fit(np.vstack(raw_feature_sequences))
        self.feature_scaler = feature_scaler
        feature_sequences = [feature_scaler.transform(f) for f in raw_feature_sequences]

        windows = []
        for feats in feature_sequences:
            for i in range(0, len(feats) - self.window + 1, self.stride):
                windows.append(feats[i:i + self.window])
        if not windows:
            raise ValueError(f"No windows could be built from {data_source}.")

        self.windows = np.asarray(windows, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = torch.tensor(self.windows[idx])
        return w, w
