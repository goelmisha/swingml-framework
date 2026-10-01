"""Tests for the point-in-time (liquidity) universe.

The liquidity path is the project's answer to survivorship bias, and it cannot
be executed on the dev machine (it needs the full-symbol price download), so
its correctness must be provable from tests: membership at date t may use ONLY
data up to t, and downstream rows must be filtered to actual members.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.config import UniverseConfig
from swingml.data.universe import (
    Universe,
    apply_membership,
    liquidity_screen,
    point_in_time_membership,
)


def _panel(n_days: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for s, base in (("AAA", 500.0), ("BBB", 50.0)):
        for i, d in enumerate(dates):
            rows.append({
                "date": d.normalize(),
                "symbol": s,
                "close": base,
                "turnover_lacs": 200.0 if s == "AAA" else 100.0,
            })
    return pd.DataFrame(rows)


def test_membership_ranks_by_trailing_turnover():
    panel = _panel()
    members = point_in_time_membership(panel, top_n=1)
    # AAA has the higher turnover on every date -> it is the only member.
    assert set(members["symbol"]) == {"AAA"}
    assert (members["rank"] == 1).all()


def test_membership_is_strictly_trailing():
    """A symbol that only becomes liquid LATE must not be a member EARLY."""
    panel = _panel()
    n = panel["date"].nunique()
    # BBB gets a massive turnover burst only on the last 10 sessions.
    burst = panel["symbol"] == "BBB"
    panel.loc[burst & (panel.groupby("symbol", observed=True).cumcount() >= n - 10), "turnover_lacs"] = 10_000.0

    members = point_in_time_membership(panel, top_n=1)
    early_dates = sorted(panel["date"].unique())[: n - 10]
    late_dates = sorted(panel["date"].unique())[-10:]

    early_syms = set(members[members["date"].isin(early_dates)]["symbol"])
    late_syms = set(members[members["date"].isin(late_dates)]["symbol"])
    # With min_periods = lookback/4 = 15, the burst cannot reach back before
    # its trailing window fills -- BBB must not be a member on the early dates.
    assert early_syms == {"AAA"}
    assert late_syms == {"BBB"}  # once the window fills, BBB dominates


def test_membership_min_periods_blocks_fresh_symbols():
    """A brand-new symbol with huge turnover is excluded until it has history."""
    panel = _panel()
    n = panel["date"].nunique()
    fresh = []
    dates = sorted(panel["date"].unique())
    for i, d in enumerate(dates[-5:]):  # only 5 sessions of history
        fresh.append({"date": d, "symbol": "NEW", "close": 100.0, "turnover_lacs": 50_000.0})
    panel = pd.concat([panel, pd.DataFrame(fresh)], ignore_index=True)
    members = point_in_time_membership(panel, top_n=2)
    assert "NEW" not in set(members["symbol"])  # 5 < min_periods 15


def test_apply_membership_filters_rows():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"),
                 pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    membership = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
        "symbol": ["AAA", "AAA", "BBB"],
        "rank": [1, 1, 2],
    })
    out = apply_membership(df, membership)
    assert len(out) == 3
    assert set(out["symbol"]) == {"AAA", "BBB"}
    # The (Jan 1, BBB) row is gone: BBB was not a member that day.
    assert not ((out["date"] == pd.Timestamp("2024-01-01")) & (out["symbol"] == "BBB")).any()


def test_apply_membership_noop_on_empty():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "symbol": ["AAA"], "x": [1.0]})
    assert apply_membership(df, None) is df
    assert len(apply_membership(df, pd.DataFrame(columns=["date", "symbol"]))) == 1


def test_liquidity_screen_thresholds():
    panel = _panel(n_days=80)
    panel["med_price"] = panel["close"]
    keep = liquidity_screen(panel, min_avg_turnover_lacs=150.0, min_price=20.0)
    assert keep == ["AAA"]  # BBB's 100 lacs median fails the 150 floor

    keep2 = liquidity_screen(panel, min_avg_turnover_lacs=50.0, min_price=60.0)
    assert keep2 == ["AAA"]  # BBB's 50.0 price fails the price floor

    keep3 = liquidity_screen(panel, min_avg_turnover_lacs=50.0, min_price=20.0)
    assert keep3 == ["AAA", "BBB"]


def test_liquidity_screen_is_trailing_only():
    """The screen uses the trailing window of the panel it is handed; it must
    not need to see the full sample to decide (verified by truncating dates)."""
    panel = _panel(n_days=80)
    full = liquidity_screen(panel, 150.0, 20.0)
    last_dates = sorted(panel["date"].unique())[-30:]
    truncated = liquidity_screen(panel[panel["date"].isin(last_dates)], 150.0, 20.0)
    assert full == truncated == ["AAA"]
