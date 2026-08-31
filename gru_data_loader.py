"""
Data loading for the GRU forecasting baseline.

Kept separate from data_loader.py on purpose: DroneSensorDataset builds
same-window reconstruction targets for the VAE, whereas the GRU branch
needs (past-window -> next-step) prediction pairs. Reuses the same
scaler classes so results stay comparable to the VAE pipeline.
"""
import glob
import os

import numpy as np
import pandas as pd

from config import Config
from data_loader import build_scaler

# Same reason as data_loader.py: the CSV readers below are on the onboard
# inference path, which scores through ONNX Runtime and never touches torch.
# Only GRUPredictionDataset needs it, so the import stays optional.
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_OK = True
except ImportError:  # inference-only environment
    _TORCH_OK = False
    Dataset = object


def read_raw_csv_with_timestamp(filepath):
    """Load one raw sensor CSV, returning (channels, real_time_sec).

    The leading column (dropped everywhere else, since only the last
    NUM_CHANNELS columns are sensor channels) is each row's PX4-style
    hrt_absolute_time in microseconds. real_time_sec is that column
    converted to seconds elapsed since the file's first row -- the true
    per-row sampling times, which vary by file/session (train ~1.2kHz,
    test sorties ~250Hz) instead of the fixed 1/Config.DATA_HZ that
    feature_engineering.py used to assume.

    A few hundred rows in x70_all.csv share an identical timestamp with
    their neighbor (0 spacing); those are nudged forward by 1us so
    downstream np.gradient calls never divide by a zero spacing.
    """
    df = pd.read_csv(filepath, comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.shape[1] <= Config.NUM_CHANNELS:
        raise ValueError(f"{filepath}: no leading timestamp column to derive real dt from.")
    timestamps_us = df.iloc[:, 0].to_numpy(dtype=np.float64)
    df = df.iloc[:, -Config.NUM_CHANNELS:]
    if df.shape[1] != Config.NUM_CHANNELS:
        raise ValueError(
            f"{filepath} has {df.shape[1]} usable columns; expected {Config.NUM_CHANNELS}."
        )
    raw = df.to_numpy(dtype=np.float32)

    diffs = np.diff(timestamps_us)
    diffs = np.where(diffs <= 0, 1.0, diffs)  # 1us floor, breaks ties/non-monotonic rows
    timestamps_us = np.concatenate([timestamps_us[:1], timestamps_us[0] + np.cumsum(diffs)])
    real_time_sec = (timestamps_us - timestamps_us[0]) / 1e6
    return raw, real_time_sec


def load_scaled_csv(filepath, scaler=None):
    """Load one raw sensor CSV and scale it. Fits a new scaler if none given."""
    raw, _ = read_raw_csv_with_timestamp(filepath)
    if scaler is None:
        scaler = build_scaler()
        scaler.fit(raw)
    return scaler.transform(raw), scaler


def load_scaled_csv_with_time(filepath, scaler=None):
    """Like load_scaled_csv, but also returns each row's true elapsed time
    (seconds) so callers can derive real per-row dt instead of assuming
    Config.DATA_HZ."""
    raw, real_time_sec = read_raw_csv_with_timestamp(filepath)
    if scaler is None:
        scaler = build_scaler()
        scaler.fit(raw)
    return scaler.transform(raw), scaler, real_time_sec


class GRUPredictionDataset(Dataset):
    """Yields (past_window, next_step) pairs from one or more scaled CSVs."""

    def __init__(self, data_source, scaler=None, window=None, stride=None):
        if not _TORCH_OK:
            raise ImportError("GRUPredictionDataset needs torch; the CSV readers in "
                              "this module do not.")
        self.window = window or Config.GRU_WINDOW
        self.stride = stride or Config.GRU_STRIDE

        if os.path.isdir(data_source):
            files = sorted(glob.glob(os.path.join(data_source, "*.csv")))
        else:
            files = [data_source]

        scaled_sequences = []
        for f in files:
            scaled, scaler = load_scaled_csv(f, scaler=scaler)
            scaled_sequences.append(scaled)
        self.scaler = scaler

        xs, ys = [], []
        for scaled in scaled_sequences:
            for i in range(0, len(scaled) - self.window, self.stride):
                xs.append(scaled[i:i + self.window])
                ys.append(scaled[i + self.window])
        if not xs:
            raise ValueError(f"No prediction pairs could be built from {data_source}.")

        self.X = np.asarray(xs, dtype=np.float32)
        self.y = np.asarray(ys, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
