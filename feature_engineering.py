"""
Feature engineering matching the B-Spline UAV paper's Section 4.3 (we skip
the B-spline reconstruction itself since our data is already uniformly
sampled at 10Hz -- only the downstream feature set is reimplemented):
first/second derivatives, instantaneous energy, and per-window summary
statistics used to keep the Isolation Forest's input compact (this is what
the paper's "low-dimensional, interpretable features" phrasing implies, and
what keeps their reported Isolation Forest under 200KB).
"""
import numpy as np

from config import Config


def compute_sequence_features(scaled_sequence, timestamps=None):
    """(T, C) raw standardized signal -> (T, C*4) per-timestep feature stream:
    [raw, first_derivative, second_derivative, energy] per channel, kept
    interleaved so channel c's 4 features live at columns [4c : 4c+4].
    Used as the LSTM-AE's input/reconstruction target (needs full sequences).

    timestamps: optional (T,) array of each row's true elapsed time in
    seconds (see gru_data_loader.read_raw_csv_with_timestamp). When given,
    derivatives use the file's real per-row spacing instead of assuming a
    fixed 1/Config.DATA_HZ -- actual sampling rate varies a lot by file
    (train ~1.2kHz vs test sorties ~250Hz), so the fixed-dt assumption
    scaled derivative magnitudes very differently per file.
    """
    raw = scaled_sequence
    coords = timestamps if timestamps is not None else 1.0 / Config.DATA_HZ
    first_deriv = np.gradient(raw, coords, axis=0)
    second_deriv = np.gradient(first_deriv, coords, axis=0)
    energy = raw ** 2
    stacked = np.stack([raw, first_deriv, second_deriv, energy], axis=-1)  # (T, C, 4)
    T, C, F = stacked.shape
    return stacked.reshape(T, C * F).astype(np.float32)


def compute_window_summary(window_seq_features, num_channels=None):
    """(window, C*4) sequence features -> compact per-window summary for the
    Isolation Forest: per channel [mean(raw), std(raw), mean(|first_deriv|),
    mean(energy)] = 4 scalars/channel instead of window*4 values/channel.
    """
    num_channels = num_channels or Config.NUM_CHANNELS
    w = window_seq_features.reshape(window_seq_features.shape[0], num_channels, 4)
    raw, first_deriv, _second_deriv, energy = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    summary = np.stack([
        raw.mean(axis=0), raw.std(axis=0),
        np.abs(first_deriv).mean(axis=0), energy.mean(axis=0),
    ], axis=-1)  # (C, 4)
    return summary.reshape(-1).astype(np.float32)  # (C*4,)
