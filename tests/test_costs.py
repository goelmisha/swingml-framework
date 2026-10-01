"""Tests for the size-aware impact cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.costs import (
    LAKHS_PER_CRORE,
    ImpactModel,
    describe,
    impact_cost_per_trade,
    positions_by_date,
)
from swingml.scorecard import period_returns


def _block(n_dates: int = 2, n_names: int = 20, adv_lakhs: float = 10_000.0) -> pd.DataFrame:
    rows = []
    for d in range(n_dates):
        for s in range(n_names):
            rows.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "symbol": f"S{s:02d}",
                "score": float(s),
                "ret_net": 0.01,
                "adv_20": adv_lakhs,
            })
    return pd.DataFrame(rows)


def test_positions_mirror_the_decile_rule():
    block = _block(n_names=20)
    assert positions_by_date(block, decile=10).tolist() == [2, 2]
    assert positions_by_date(block, decile=2).tolist() == [10, 10]
    assert positions_by_date(block, decile=1).tolist() == [20, 20]
    with pytest.raises(ValueError):
        positions_by_date(block, decile=0)


def test_impact_is_the_square_root_law_on_both_legs():
    block = _block(adv_lakhs=10_000.0)
    model = ImpactModel(book_size_cr=10.0, impact_k=1.0)
    cost = impact_cost_per_trade(block, positions_by_date(block, 10), model)

    position_cr = 10.0 / 2
    adv_cr = 10_000.0 / LAKHS_PER_CRORE
    participation = position_cr / adv_cr
    assert cost[0] == pytest.approx(2.0 * np.sqrt(participation))


def test_concentrating_capital_costs_more_impact_not_less():
    """Selecting a tenth of the universe puts 10x the capital in each name."""
    block = _block(adv_lakhs=10_000.0)
    model = ImpactModel(book_size_cr=10.0, impact_k=0.1)
    concentrated = impact_cost_per_trade(block, positions_by_date(block, 10), model)[0]
    diversified = impact_cost_per_trade(block, positions_by_date(block, 1), model)[0]
    assert concentrated > diversified
    # 10x participation -> sqrt(10) more impact per name
    assert concentrated / diversified == pytest.approx(np.sqrt(10.0), rel=1e-9)


def test_thinner_names_cost_more_than_deep_ones():
    deep = _block(adv_lakhs=100_000.0)
    thin = _block(adv_lakhs=500.0)
    model = ImpactModel(book_size_cr=10.0, impact_k=0.1)
    assert (impact_cost_per_trade(thin, positions_by_date(thin, 10), model)[0]
            > impact_cost_per_trade(deep, positions_by_date(deep, 10), model)[0])


def test_missing_adv_falls_back_to_that_sessions_median():
    block = _block(adv_lakhs=10_000.0)
    block.loc[0, "adv_20"] = np.nan          # one warm-up row
    model = ImpactModel(book_size_cr=10.0, impact_k=0.1)
    cost = impact_cost_per_trade(block, positions_by_date(block, 10), model)
    assert np.isfinite(cost).all()
    assert cost[0] == pytest.approx(cost[1])   # median == the rest on that session


def test_zero_adv_does_not_produce_infinite_cost():
    block = _block(adv_lakhs=0.0)
    model = ImpactModel(book_size_cr=10.0, impact_k=0.1)
    assert np.isfinite(impact_cost_per_trade(block, positions_by_date(block, 10), model)).all()


def test_model_validates_its_parameters():
    with pytest.raises(ValueError):
        ImpactModel(book_size_cr=0.0)
    with pytest.raises(ValueError):
        ImpactModel(impact_k=-0.1)
    assert "crore" in describe(ImpactModel())
    with pytest.raises(ValueError):
        impact_cost_per_trade(_block().drop(columns=["adv_20"]), pd.Series([2, 2]),
                              ImpactModel())


def test_period_returns_accepts_a_per_row_charge():
    """An array charge is averaged over the selected names, like a scalar would be."""
    block = _block(n_names=20)
    charge = np.full(len(block), 0.0025)
    r = period_returns(block, block["score"].to_numpy(), decile=10, extra_friction=charge)
    assert np.allclose(r.to_numpy(), 0.01 - 0.0025)
    with pytest.raises(ValueError):
        period_returns(block, block["score"].to_numpy(), extra_friction=np.zeros(3))
