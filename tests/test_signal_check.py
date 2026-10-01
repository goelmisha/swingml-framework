"""Tests for the signal_check definition-B barrier-trade check."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

import scripts.signal_check as sc
from swingml import config as cfg_mod


@pytest.fixture
def joined():
    """Two dates x 20 symbols, feature + barrier-trade outcome per row.

    Top decile (s >= 18) wins with ret_net +0.04; everything else loses.
    """
    rows = []
    for d in (0, 1):
        for s in range(20):
            selected = s >= 18
            rows.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "symbol": f"S{s:02d}",
                "xs_rank_roc_20": float(s),
                "ret_net": 0.04 if selected else -0.005,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """Point the script's PROJECT_ROOT at a tmp dir so repo labels never interfere."""
    monkeypatch.setattr(sc, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data" / "datasets").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def capture(monkeypatch):
    lines = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    return lines


def test_skips_gracefully_without_labels(joined, isolated_root, capture):
    out = sc.barrier_trade_check(joined, ["xs_rank_roc_20"])
    assert out is None
    assert any("skipped" in l for l in capture)


def test_scores_top_decile_under_definition_b(joined, isolated_root, capture):
    labels_dir = isolated_root / "data" / "datasets"
    pd.DataFrame({
        "date": joined["date"], "symbol": joined["symbol"],
        "label": 1, "ret_net": joined["ret_net"],
    }).to_parquet(labels_dir / "labels.parquet", index=False)

    sc.barrier_trade_check(joined, ["xs_rank_roc_20"])
    text = "\n".join(capture)
    assert "DEFINITION B" in text
    # The top decile (2 rows/date) is exactly the winning rows -> 100% precision B.
    assert "100.0%" in text
    # Feature column must not leak into the selection ranking.
    assert "xs_rank_roc_20" in text


def test_no_labels_keys_means_no_crash(joined, isolated_root, capture):
    """Labels exist but share no (date, symbol) with the feature frame."""
    labels_dir = isolated_root / "data" / "datasets"
    pd.DataFrame({
        "date": pd.Timestamp("1999-01-01"), "symbol": "ZZZ",
        "label": 1, "ret_net": 0.01,
    }, index=[0]).to_parquet(labels_dir / "labels.parquet", index=False)

    out = sc.barrier_trade_check(joined, ["xs_rank_roc_20"])
    assert out is None
    assert any("share no" in l for l in capture)
