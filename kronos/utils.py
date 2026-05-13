"""Utility functions for Kronos time series prediction.

Provides helpers for data preprocessing, normalization, date handling,
and result post-processing used across the Kronos pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


def normalize_series(
    data: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, dict]:
    """Normalize a 1-D or 2-D array and return the array plus scaling params.

    Args:
        data: Input array of shape (T,) or (T, F).
        method: One of ``'minmax'`` or ``'zscore'``.
        feature_range: Target range used only when *method* is ``'minmax'``.

    Returns:
        Tuple of (normalized_array, params_dict) where *params_dict* contains
        the values needed to invert the transformation.
    """
    data = np.asarray(data, dtype=np.float64)
    if method == "minmax":
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        scale = hi - lo
        # Avoid division by zero for constant features
        scale = np.where(scale == 0, 1.0, scale)
        a, b = feature_range
        normalized = (data - lo) / scale * (b - a) + a
        params = {"method": "minmax", "lo": lo, "hi": hi, "feature_range": feature_range}
    elif method == "zscore":
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        normalized = (data - mu) / sigma
        params = {"method": "zscore", "mu": mu, "sigma": sigma}
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")
    return normalized, params


def denormalize_series(data: np.ndarray, params: dict) -> np.ndarray:
    """Invert a normalization produced by :func:`normalize_series`.

    Args:
        data: Normalized array.
        params: The *params_dict* returned by :func:`normalize_series`.

    Returns:
        Array in the original scale.
    """
    data = np.asarray(data, dtype=np.float64)
    method = params["method"]
    if method == "minmax":
        a, b = params["feature_range"]
        lo, hi = params["lo"], params["hi"]
        scale = hi - lo
        scale = np.where(scale == 0, 1.0, scale)
        return (data - a) / (b - a) * scale + lo
    elif method == "zscore":
        return data * params["sigma"] + params["mu"]
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def create_sliding_windows(
    series: np.ndarray,
    window_size: int,
    horizon: int = 1,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) pairs from a time series using a sliding window.

    Args:
        series: 1-D array of length T.
        window_size: Number of past steps used as input.
        horizon: Number of future steps to predict.
        step: Stride between consecutive windows.

    Returns:
        Tuple (X, y) where X has shape (N, window_size) and
        y has shape (N, horizon).
    """
    series = np.asarray(series, dtype=np.float64)
    xs, ys = [], []
    total = len(series)
    for start in range(0, total - window_size - horizon + 1, step):
        end = start + window_size
        xs.append(series[start:end])
        ys.append(series[end: end + horizon])
    return np.array(xs), np.array(ys)


def align_dataframe_to_trading_days(
    df: pd.DataFrame,
    date_col: str = "date",
    freq: str = "B",
) -> pd.DataFrame:
    """Re-index a DataFrame to a continuous business-day (or custom) frequency.

    Missing rows are forward-filled so that gaps caused by holidays or
    missing data do not break downstream windowing.

    Args:
        df: Input DataFrame with a date column.
        date_col: Name of the date column (will become the index).
        freq: Pandas offset alias, e.g. ``'B'`` for business days.

    Returns:
        Re-indexed DataFrame with *date_col* as a DatetimeIndex.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range).ffill()
    df.index.name = date_col
    return df


def compute_returns(
    prices: Union[pd.Series, np.ndarray],
    log: bool = False,
) -> np.ndarray:
    """Compute simple or log returns from a price series.

    Args:
        prices: 1-D price array or Series.
        log: If ``True``, compute log returns; otherwise simple returns.

    Returns:
        Array of returns with length ``len(prices) - 1``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if log:
        return np.diff(np.log(prices))
    return np.diff(prices) / prices[:-1]
