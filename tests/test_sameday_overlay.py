"""The same-day overlay: fill the newest session's OHLC from the bhavcopy.

Why it exists: the index series is current the same evening while its constituent
bars are not. Without this the freeze sees a truncated panel and the
thin-cross-section guard refuses -- correctly.

The tests that matter are the guard rails, not the happy path: a normal backfill
must be a no-op, and an ex-date must never be spliced onto an adjusted history.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swingml.dataset import (
    MAX_CALENDAR_TAIL_DAYS,
    TailSessionOverlay,
    extend_calendar_tail,
    overlay_tail_session,
)
from swingml.data.universe import trading_days_from_benchmark

SESSION = pd.Timestamp("2024-03-15")  # a Friday
PRIOR = pd.Timestamp("2024-03-14")


def _prices(closes: list[float], end: pd.Timestamp = PRIOR, symbol: str = "AAA") -> dict:
    idx = pd.bdate_range(end=end, periods=len(closes))
    return {
        symbol: pd.DataFrame(
            {
                "open": closes, "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes], "close": closes,
                "volume": [1000.0] * len(closes),
            },
            index=idx,
        )
    }


def _panel(symbols=("AAA",), session: pd.Timestamp = SESSION, prev_close=100.0,
           close=102.0, prior_close=100.0) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        rows.append({
            "symbol": sym, "date": PRIOR, "series": "EQ",
            "prev_close": prior_close, "open": 100.5, "high": 103.0,
            "low": 100.0, "close": prior_close, "ttl_trd_qnty": 900.0,
        })
        rows.append({
            "symbol": sym, "date": session, "series": "EQ",
            "prev_close": prev_close, "open": 101.0, "high": 103.5,
            "low": 100.5, "close": close, "ttl_trd_qnty": 1234.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# calendar tail
# ---------------------------------------------------------------------------

def test_calendar_tail_is_appended_up_to_the_end_date():
    days = [dt.date(2024, 3, 13), dt.date(2024, 3, 14)]
    out = extend_calendar_tail(days, dt.date(2024, 3, 18))
    assert out[:2] == days                      # nothing existing is reordered
    assert out[2:] == [dt.date(2024, 3, 15), dt.date(2024, 3, 18)]  # weekdays only


def test_calendar_tail_is_empty_when_the_benchmark_is_current():
    days = [dt.date(2024, 3, 15)]
    assert extend_calendar_tail(days, dt.date(2024, 3, 15)) == days
    assert extend_calendar_tail(days, None) == days


def test_calendar_tail_is_bounded():
    """A badly stale benchmark must not become hundreds of bhavcopy requests."""
    days = [dt.date(2020, 1, 1)]
    out = extend_calendar_tail(days, dt.date(2026, 1, 1))
    assert len(out) == 1 + MAX_CALENDAR_TAIL_DAYS


def test_benchmark_calendar_roundtrip_still_works():
    idx = pd.bdate_range("2024-03-11", periods=3)
    bench = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    assert trading_days_from_benchmark(bench) == [
        dt.date(2024, 3, 11), dt.date(2024, 3, 12), dt.date(2024, 3, 13),
    ]


# ---------------------------------------------------------------------------
# the overlay
# ---------------------------------------------------------------------------

def test_newest_session_is_filled_from_the_panel():
    prices = _prices([98.0, 99.0, 100.0])
    report = overlay_tail_session(prices, _panel())
    assert report.session == SESSION
    assert report.filled == ["AAA"]
    assert SESSION in prices["AAA"].index
    assert prices["AAA"].loc[SESSION, "close"] == 102.0
    assert prices["AAA"].loc[SESSION, "volume"] == 1234.0
    # History is untouched and still sorted.
    assert prices["AAA"].loc[PRIOR, "close"] == 100.0
    assert prices["AAA"].index.is_monotonic_increasing


def test_backfill_is_a_no_op_when_the_bar_is_already_there():
    """A session the provider already has is never replaced, whatever the source."""
    prices = _prices([98.0, 99.0, 100.0], end=SESSION)
    prices["AAA"].loc[SESSION, "close"] = 555.0  # provider's own (adjusted) value
    report = overlay_tail_session(prices, _panel())
    assert report.already_present == ["AAA"] and report.filled == []
    assert prices["AAA"].loc[SESSION, "close"] == 555.0


def test_ex_date_name_is_excluded_not_spliced():
    """A 1:2 split restates PREV_CLOSE to ~50; the raw bar is a different basis."""
    prices = _prices([98.0, 99.0, 100.0])
    report = overlay_tail_session(prices, _panel(prev_close=50.0, close=51.0))
    assert report.skipped_corporate_action == ["AAA"]
    assert report.filled == []
    assert SESSION not in prices["AAA"].index


def test_ordinary_restatement_within_tolerance_still_fills():
    """PREV_CLOSE never exactly equals the prior close; 1% is not a split."""
    prices = _prices([98.0, 99.0, 100.0])
    report = overlay_tail_session(prices, _panel(prev_close=100.5, close=101.0))
    assert report.filled == ["AAA"]


def test_symbol_missing_from_the_panel_is_reported_not_invented():
    prices = _prices([98.0, 99.0, 100.0])
    prices["ZZZ"] = prices["AAA"].copy()
    report = overlay_tail_session(prices, _panel())
    assert report.filled == ["AAA"]
    assert report.absent_from_panel == ["ZZZ"]
    assert SESSION not in prices["ZZZ"].index


def test_session_absent_from_the_panel_leaves_prices_alone():
    prices = _prices([98.0, 99.0, 100.0])
    report = overlay_tail_session(prices, _panel(session=SESSION), session=pd.Timestamp("2024-03-19"))
    assert report.filled == [] and report.already_present == []
    assert SESSION not in prices["AAA"].index


def test_garbage_bar_is_refused():
    """A NaN close must not enter the panel as a real bar."""
    prices = _prices([98.0, 99.0, 100.0])
    panel = _panel()
    panel.loc[panel["date"] == SESSION, "close"] = np.nan
    report = overlay_tail_session(prices, panel)
    assert report.skipped_invalid_bar == ["AAA"]
    assert SESSION not in prices["AAA"].index


def test_empty_inputs_are_safe():
    assert overlay_tail_session({}, _panel()).filled == []
    assert overlay_tail_session(_prices([1.0, 2.0]), pd.DataFrame()).filled == []


def test_defaults_to_the_panels_own_newest_session():
    prices = _prices([98.0, 99.0, 100.0])
    report = overlay_tail_session(prices, _panel())
    assert report.session == SESSION


def test_report_serialises_for_the_dataset_metadata():
    report = TailSessionOverlay(session=SESSION, filled=["A", "B"], already_present=["C"],
                                skipped_corporate_action=["D"], absent_from_panel=["E"])
    d = report.as_dict()
    assert d["session"] == "2024-03-15"
    assert (d["n_filled"], d["n_already_present"], d["n_absent_from_panel"]) == (2, 1, 1)
    assert d["skipped_corporate_action"] == ["D"]


@pytest.mark.parametrize("symbols", [("AAA",), ("AAA", "BBB")])
def test_multiple_symbols_fill_independently(symbols):
    prices = {s: _prices([98.0, 99.0, 100.0])["AAA"] for s in symbols}
    report = overlay_tail_session(prices, _panel(symbols=symbols))
    assert sorted(report.filled) == list(symbols)
    for s in symbols:
        assert prices[s].loc[SESSION, "close"] == 102.0
