"""Signal preprocessing and feature extraction for battery cycle data."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.config import S, SEQUENCE_LENGTH, PREDICTION_INTERVAL


def average_over_intervals(data: np.ndarray, s: int = S) -> list[float]:
    """Divide a 1-D signal into *s* equal segments and return their means."""
    chunk_size = len(data) // s
    if chunk_size <= 0:
        return [float(np.mean(data))] * s
    return [float(np.mean(data[i * chunk_size:(i + 1) * chunk_size])) for i in range(s)]


def gaussian_smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply 1-D Gaussian filter for noise reduction."""
    return gaussian_filter1d(arr, sigma=sigma)


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-cycle min-max normalization to [0, 1]."""
    mn, mx = np.min(arr), np.max(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def preprocess_battery_features(
    charge_data,
    discharge_data,
    cycle_count: int,
    cap_divisor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-cycle features and SOH labels.

    For each cycle:
        1. Gaussian-smooth voltage / current / temperature signals
        2. Min-max normalize each signal independently (no future leakage)
        3. Downsample via interval averaging (S segments each)
        4. Append raw capacity

    Returns
    -------
    features : ndarray, shape (cycle_count, 3*S+1)
    capacity_soh : ndarray, shape (cycle_count,)  — SOH = cap[t] / cap[0]
    """
    combined, capacities = [], []

    for i in range(cycle_count):
        v = min_max_normalize(gaussian_smooth(charge_data[i][0][0][0][0]))
        c = min_max_normalize(gaussian_smooth(charge_data[i][0][0][1][0]))
        t = min_max_normalize(gaussian_smooth(charge_data[i][0][0][2][0]))

        cap = float(discharge_data[i][0][0][6][0][0] / cap_divisor)
        capacities.append(cap)
        combined.append(
            average_over_intervals(v) + average_over_intervals(c) + average_over_intervals(t) + [cap]
        )

    initial_cap = capacities[0]
    capacity_soh = np.array([c / initial_cap for c in capacities], dtype=float)

    features = np.array(combined, dtype=float)
    features[:, 3 * S] /= initial_cap          # normalize capacity column to SOH

    return features, capacity_soh


def create_sequences(
    features: np.ndarray,
    capacity_soh: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
    pred_interval: int = PREDICTION_INTERVAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences with delta-SOH targets.

    Returns
    -------
    X          : ndarray, shape (N, seq_len, 3*S+1)
    y_delta    : ndarray, shape (N,) — target = SOH_future - SOH_current
    y_true     : ndarray, shape (N,) — absolute future SOH (for evaluation)
    """
    X, y_delta, y_true = [], [], []
    n_cycles = len(features)

    for i in range(n_cycles - pred_interval - seq_len + 1):
        x_seq = features[i:i + seq_len].copy()
        soh_future = float(capacity_soh[i + seq_len + pred_interval - 1])
        soh_current = float(x_seq[-1, 3 * S])

        X.append(x_seq)
        y_delta.append(soh_future - soh_current)
        y_true.append(soh_future)

    return np.array(X), np.array(y_delta), np.array(y_true)
