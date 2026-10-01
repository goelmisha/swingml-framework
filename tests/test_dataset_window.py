"""The price-fetch window must include the last session, not stop one short.

Regression tests for a defect: ``build_dataset`` passed the calendar's last
session straight to a yfinance-backed fetch, whose ``end`` is **exclusive**. The
dataset therefore ended one session before its own trading calendar. Nothing in
the backtest noticed, because the loss is one row at the tail; the daily forward
paper trade could not work at all, since every freeze would have been one session
stale and therefore refused as a backfill.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingml import dataset as ds
from swingml.config import load_config
from swingml.data.prices import MockPriceProvider


def test_inclusive_end_moves_past_the_session_it_must_include():
    assert ds.inclusive_end(dt.date(2026, 9, 28)) == "2026-09-29"
    assert ds.inclusive_end("2026-09-28") == "2026-09-29"
    assert ds.inclusive_end(pd.Timestamp("2026-12-31")) == "2027-01-01"


def test_inclusive_end_defaults_to_today_inclusive(monkeypatch):
    real = dt.date
    monkeypatch.setattr(ds.dt, "date", _FrozenDate)
    assert ds.inclusive_end(None) == (pd.Timestamp(real(2026, 9, 29)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


class _FrozenDate(dt.date):
    @classmethod
    def today(cls):  # type: ignore[override]
        return cls(2026, 9, 29)


def test_build_dataset_asks_for_the_day_after_the_last_session(tmp_path, monkeypatch):
    """End-to-end: the request window must not truncate the calendar's last session."""
    seen: list[str] = []
    original = MockPriceProvider.get_many

    def spy(self, symbols, start=None, end=None, **kwargs):
        seen.append(end)
        return original(self, symbols, start, end, **kwargs)

    monkeypatch.setattr(MockPriceProvider, "get_many", spy)

    cfg = load_config()
    cfg.data.price_provider = "mock"
    cfg.data.delivery_provider = "mock"
    cfg.data.start = "2020-01-01"
    cfg.data.end = "2021-12-31"
    cfg.data.mock.n_days = 3000
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.dataset_dir = str(tmp_path / "dataset")

    bundle = ds.build_dataset(cfg)

    assert seen, "the symbol fetch never happened"
    last_session = pd.Timestamp(bundle.features["date"].max())
    requested = pd.Timestamp(seen[0])
    assert requested > last_session, (
        f"asked for {requested.date()} but the panel's last session is "
        f"{last_session.date()}: the fetch window drops the newest session"
    )


def test_dataset_ends_on_the_calendar_last_session(tmp_path, monkeypatch):
    """The panel's last session must be the calendar's, not one before it."""
    cfg = load_config()
    cfg.data.price_provider = "mock"
    cfg.data.delivery_provider = "mock"
    cfg.data.start = "2020-01-01"
    cfg.data.end = "2021-12-31"
    cfg.data.mock.n_days = 3000
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.dataset_dir = str(tmp_path / "dataset")

    bundle = ds.build_dataset(cfg)
    calendar_end = dt.date.fromisoformat(bundle.meta["end"])
    assert pd.Timestamp(bundle.features["date"].max()).date() == calendar_end


@pytest.mark.parametrize("bad", ["2026-09-28", dt.date(2026, 9, 28)])
def test_inclusive_end_accepts_the_types_the_calendar_uses(bad):
    assert ds.inclusive_end(bad) == "2026-09-29"
