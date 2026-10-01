"""Indicator primitives.

Textbook, unit-tested building blocks. Each is causal by construction: it reads
only the current and preceding bars, so a feature assembled from these is safe
under walk-forward validation as long as the assembly itself respects ordering.

Kept free of any project-specific signal so they can be reused and verified in
isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with ``adjust=False`` (recursive, causal)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing window."""
    return series.rolling(window, min_periods=window).mean()


def roc(series: pd.Series, window: int) -> pd.Series:
    """Rate of change over ``window`` bars, as a fraction."""
    return series.pct_change(window, fill_method=None)


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Uses Wilder smoothing, not a simple mean of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # alpha = 1/window is exactly Wilder's smoothing factor.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken up-run -> RSI 100 by definition.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna() | avg_loss.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max of the three classic spans, using *previous* close."""
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing (matches TA libraries)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def realized_vol(close: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """Annualised close-to-close volatility."""
    return close.pct_change(fill_method=None).rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score against a trailing window. Uses only past observations."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)
