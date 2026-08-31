"""
Flight preprocessing shared by every model script (pipeline v4, 2026-08-20).

Replaces the ad-hoc "read CSV -> global scaler -> features -> fixed-sample
window" path that every script used to inline. That path had three defects,
found by auditing against Reis & Reis 2025 / Ahn & Chung 2024 and quantified
in eval_pipeline_fixes_ablation.py (PROJECT_SUMMARY sec 3.9):

  1. `energy` = raw^2 amplified session offsets QUADRATICALLY, so test files
     landed hundreds of sigma outside the training distribution (max scaled
     feature 46,404 vs 123 on train) and the LSTM-AE saturated -- its output
     stopped tracking its input entirely (corr +0.25, MAE 46.2).
  2. No per-flight normalization, though Reis & Reis sec 6.1 specifies
     "All channels were normalized per flight".
  3. Window length was a SAMPLE COUNT, so the same window spanned 32ms at
     training (~991Hz flight) and 128ms at evaluation (~250Hz sorties).

This module fixes 2 and 3 (and thereby most of 1): every flight is resampled
to a common rate, normalized by its own statistics, and windowed by real
duration. `energy` is kept -- dropping it gives better detection but breaks
the mag_x channel diagnosis, which is this project's headline result (see
the ablation's variant C vs D).

v4.1 (2026-08-20) makes the per-flight normalization CAUSAL: the scaler is
fit on the flight's first Config.WARMUP_SEC seconds rather than on the whole
flight. Whole-flight statistics let data that has not happened yet influence
the present -- injecting a fault at t=100s shifted the normalized values at
t=10s by 0.358 on average -- which is both a validity problem offline and an
outright impossibility onboard, where the rest of the flight does not exist
when a window is scored. See PROJECT_SUMMARY sec 3.13.
"""
import numpy as np

from config import Config
from data_loader import build_scaler
from feature_engineering import compute_sequence_features
from gru_data_loader import read_raw_csv_with_timestamp


def resample_to_rate(raw, real_time, hz=None):
    """(T, C) samples on an irregular real-time axis -> (N, C) bin means at a
    fixed rate, plus the matching regular time axis in seconds.

    Bin i holds the mean of every sample whose real timestamp falls in
    [i/hz, (i+1)/hz), so a window of k bins is always k/hz real seconds --
    on the ~991Hz training flight and the ~250Hz test sorties alike."""
    hz = hz or Config.RESAMPLE_HZ
    n = int(np.floor(real_time[-1] * hz))
    if n < 1:
        return raw[:0], real_time[:0]
    idx = np.minimum((real_time * hz).astype(int), n - 1)
    binned = np.zeros((n, raw.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(binned, idx, raw)
    np.add.at(counts, idx, 1)
    counts = np.maximum(counts, 1)
    return (binned / counts[:, None]).astype(np.float32), (np.arange(n) / hz).astype(np.float64)


def prepare_flight(raw, real_time, scaler=None, hz=None, warmup_sec=None):
    """raw channels + real timestamps -> (engineered features, time axis, scaler).

    When `scaler` is None a fresh one is fit on this flight's first
    `warmup_sec` seconds (causal per-flight normalization; pass
    warmup_sec=float("inf") for the old whole-flight behaviour). Pass an
    existing scaler only when you deliberately want a shared one."""
    resampled, time_axis = resample_to_rate(raw, real_time, hz)
    if len(resampled) == 0:
        return resampled, time_axis, scaler
    if scaler is None:
        warmup_sec = Config.WARMUP_SEC if warmup_sec is None else warmup_sec
        warm = resampled[time_axis <= warmup_sec]
        scaler = build_scaler()
        scaler.fit(warm if len(warm) >= 10 else resampled)
    feats = compute_sequence_features(scaler.transform(resampled), time_axis)
    return feats, time_axis, scaler


def prepare_flight_file(filepath, scaler=None, hz=None, warmup_sec=None):
    """prepare_flight() straight from a CSV path."""
    raw, real_time = read_raw_csv_with_timestamp(filepath)
    return prepare_flight(raw, real_time, scaler=scaler, hz=hz, warmup_sec=warmup_sec)


def window_length(hz=None, window_sec=None):
    """Window length in samples for the configured real duration."""
    hz = hz or Config.RESAMPLE_HZ
    return int(round((window_sec or Config.WINDOW_SEC) * hz))


def stride_length(hz=None, stride_sec=None):
    return max(1, int(round((stride_sec or Config.STRIDE_SEC) * (hz or Config.RESAMPLE_HZ))))


def build_windows(feats, window=None, stride=None):
    """(T, D) -> (N, window, D). Empty array if the sequence is too short."""
    window = window or window_length()
    stride = stride or stride_length()
    if len(feats) < window:
        return np.zeros((0, window, feats.shape[1]), dtype=np.float32)
    return np.stack([feats[i:i + window] for i in range(0, len(feats) - window + 1, stride)]).astype(np.float32)


def window_end_times(time_axis, n_windows, window=None, stride=None):
    """Real time (seconds) at the END of each window -- the timestamp a
    window's score should be attributed to."""
    window = window or window_length()
    stride = stride or stride_length()
    idx = np.arange(n_windows) * stride + window - 1
    idx = np.minimum(idx, len(time_axis) - 1)
    return time_axis[idx]
