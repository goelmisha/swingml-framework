"""Shared fixtures: deterministic offline price + delivery panels."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingml.config import DataConfig, FeatureConfig
from swingml.data.delivery import MockDeliveryProvider
from swingml.data.prices import MockPriceProvider

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")
CAL_START = "2020-01-01"
CAL_END = "2022-08-01"


def _make(symbols=SYMBOLS, start=CAL_START, end=CAL_END, seed=3):
    """Build aligned synthetic OHLCV + delivery frames on one calendar.

    Generated ONCE over the full window and then sliced by callers, because the
    mock RNG's draw count depends on the window length -- regenerating over a
    shorter window would produce different values and break truncation tests.
    """
    dcfg = DataConfig(price_provider="mock", delivery_provider="mock", start=start, end=end)
    dcfg.mock.seed = seed
    dcfg.mock.n_days = 5000  # large enough that the requested window is never trimmed

    prices = MockPriceProvider(dcfg)
    delivery = MockDeliveryProvider(dcfg)

    px = {s: prices.get_ohlcv(s, start, end) for s in symbols}
    trading_days = sorted({ts.date() for ts in px[symbols[0]].index})
    panel = delivery.get_delivery(start, end, symbols=list(symbols), trading_days=trading_days)
    bench = prices.get_benchmark(start, end)
    return px, panel, bench


@pytest.fixture(scope="session")
def synth():
    """Full-window synthetic frames: ``(prices, delivery_panel, benchmark)``."""
    return _make()


@pytest.fixture(scope="session")
def feature_cfg():
    return FeatureConfig()
