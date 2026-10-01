"""Tests for the shared evaluation module (definitions A and B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.evaluation import evaluate_selections, top_fraction_mask


@pytest.fixture
def block():
    """Two dates x 20 symbols with known outcomes.

    The top-10% selection (2 symbols per date) is the highest-scored rows.
    On date 0 the selected rows are all profit labels with positive ret_net;
    on date 1 they are all stops with negative ret_net. Base rates differ.
    """
    rows = []
    for d in (0, 1):
        for s in range(20):
            selected = s >= 18
            if d == 0:
                label, ret_net = (1, 0.04) if selected else (0, 0.001)
            else:
                label, ret_net = (-1, -0.03)
            rows.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "symbol": f"S{s:02d}",
                "label": label,
                "ret_net": ret_net,
                "target": 1 if label == 1 else 0,
            })
    df = pd.DataFrame(rows)
    scores = np.tile(np.arange(20, dtype=float), 2)  # symbol index = score
    return df, scores


def test_definition_a_scores_profit_label_rate(block):
    df, scores = block
    st = evaluate_selections(df, scores, decile=10)
    # Selection = 2 rows per date x 2 dates = 4 rows, all label==1 on date 0
    # and none on date 1 -> precision A = 2/4.
    assert st.n_selected == 4
    assert st.precision_a == pytest.approx(0.5)
    # Base rate over all 40 rows: 2 profit labels out of 40.
    assert st.base_precision_a == pytest.approx(0.05)


def test_definition_b_scores_money_made(block):
    df, scores = block
    st = evaluate_selections(df, scores, decile=10)
    # Selected: date 0 both ret_net > 0, date 1 both < 0 -> B = 0.5.
    assert st.precision_b == pytest.approx(0.5)
    # avg_net of selections = (0.04 + 0.04 - 0.03 - 0.03) / 4 = 0.005.
    assert st.avg_net == pytest.approx(0.005)
    assert st.base_precision_b == pytest.approx((df["ret_net"] > 0).mean())


def test_lifts_are_selected_minus_base(block):
    df, scores = block
    st = evaluate_selections(df, scores, decile=10)
    lifts = st.lifts()
    assert lifts["precision_a_lift"] == pytest.approx(st.precision_a - st.base_precision_a)
    assert lifts["precision_b_lift"] == pytest.approx(st.precision_b - st.base_precision_b)
    assert lifts["net_lift"] == pytest.approx(st.avg_net - st.base_avg_net)


def test_auc_is_computed_when_both_classes_present(block):
    df, scores = block
    st = evaluate_selections(df, scores, decile=10)
    assert 0.5 <= st.auc <= 1.0


def test_scores_must_align_with_rows(block):
    df, scores = block
    with pytest.raises(ValueError):
        evaluate_selections(df, scores[:5])


def test_top_fraction_mask_selects_within_date(block):
    df, scores = block
    mask = top_fraction_mask(pd.Series(scores), df["date"], 10)
    # Top 10% within each date = the two highest-scored rows per date.
    assert mask.sum() == 4
    assert mask[:18].sum() == 0 and mask[18:20].sum() == 2
    assert mask[38:].sum() == 2


def test_perfect_and_inverse_scores_bound_auc(block):
    df, scores = block
    perfect = np.where(df["label"] == 1, 10.0, 0.0)
    st = evaluate_selections(df, perfect, decile=10)
    assert st.auc == pytest.approx(1.0)
    st_inv = evaluate_selections(df, -perfect, decile=10)
    assert st_inv.auc == pytest.approx(0.0)
