"""Tests for regime_experiment's per-fold export and gate-0 verdict.

The per-fold per-config JSON is the input to the config-averaged existence test,
so the export shape and the verdict rule are pinned here rather than left to a
run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.regime_experiment as rx

ENGINES = ("sklearn-hgb", "lightgbm", "xgboost")


def _record(fold: int, nets: dict[tuple[str, str], float], base_net: float = -0.001) -> dict:
    """One fold's export record, from {(engine, arm): avg_net}."""
    return {
        "fold": fold,
        "test_start": "2024-01-01",
        "test_end": "2024-06-30",
        "n_train": 504,
        "n_test": 1200,
        "base": {"precision_a": 0.285, "precision_b": 0.389, "avg_net": base_net},
        "configs": [
            {"engine": e, "arm": a, "precision_a": 0.314, "precision_b": 0.40,
             "avg_net": n, "auc": 0.51, "n_selected": 120,
             "base_precision_a": 0.285, "base_precision_b": 0.389,
             "base_avg_net": base_net}
            for (e, a), n in nets.items()
        ],
    }


def _all_engines(net_by_arm: dict[str, float]) -> dict[tuple[str, str], float]:
    return {(e, a): net for e in ENGINES for a, net in net_by_arm.items()}


# ---------------------------------------------------------------------------
# Export path and payload
# ---------------------------------------------------------------------------

def test_default_export_path_is_inside_the_dataset_dir():
    assert rx.default_export_path("data/datasets_liquidity") == \
        rx.Path("data/datasets_liquidity") / rx.EXPORT_FILENAME


def test_export_writes_folds_meta_and_verdict(tmp_path):
    records = [_record(0, _all_engines({"A": 0.002, "B": -0.004}))]
    verdict = rx.config_averaged_existence(records, arm="A")
    p = rx.export_fold_metrics(
        tmp_path / "nested" / "regime_folds.json", records,
        meta={"dataset_dir": "data/datasets_liquidity", "engines": list(ENGINES)},
        existence=verdict,
    )

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["meta"]["dataset_dir"] == "data/datasets_liquidity"
    assert payload["folds"][0]["configs"]  # per-config rows survive the round trip
    assert payload["config_averaged_existence"]["passed"] is True  # nested dir created
    assert p.parent.is_dir()


# ---------------------------------------------------------------------------
# Config-averaged existence test
# ---------------------------------------------------------------------------

def test_engine_average_beats_base_in_majority_of_folds():
    records = [
        _record(0, _all_engines({"A": 0.004, "B": 0.0})),
        _record(1, _all_engines({"A": 0.003, "B": 0.0})),
        _record(2, _all_engines({"A": 0.002, "B": 0.0})),
        _record(3, _all_engines({"A": -0.001, "B": 0.0})),  # base is -0.001 -> flat
    ]
    out = rx.config_averaged_existence(records, arm="A")
    assert out["n_folds"] == 4
    assert out["folds_improved"] == 3
    assert out["mean_diff"] > 0
    assert out["passed"] is True


def test_engine_luck_is_averaged_out_rather_than_selected():
    """The whole point of item 1a: one lucky engine must not carry the verdict."""
    records = [
        _record(0, {("sklearn-hgb", "A"): 0.05, ("lightgbm", "A"): -0.05,
                    ("xgboost", "A"): -0.05, ("sklearn-hgb", "B"): 0.0}),
        _record(1, {("sklearn-hgb", "A"): 0.05, ("lightgbm", "A"): -0.05,
                    ("xgboost", "A"): -0.05, ("sklearn-hgb", "B"): 0.0}),
    ]
    out = rx.config_averaged_existence(records, arm="A")
    # Best single config is strongly positive; the 3-engine average is not.
    assert out["mean_diff"] < 0
    assert out["folds_improved"] == 0
    assert out["passed"] is False
    assert out["per_fold"][0]["n_configs"] == 3  # arm B excluded


def test_arm_b_is_evaluated_separately():
    records = [
        _record(0, _all_engines({"A": -0.002, "B": 0.004})),
        _record(1, _all_engines({"A": -0.002, "B": 0.004})),
    ]
    assert rx.config_averaged_existence(records, arm="A")["passed"] is False
    assert rx.config_averaged_existence(records, arm="B")["passed"] is True


def test_flat_folds_do_not_count_as_improvement():
    records = [_record(0, _all_engines({"A": 0.0}), base_net=0.0)]  # mean net == base net
    out = rx.config_averaged_existence(records, arm="A")
    assert out["folds_improved"] == 0
    assert out["passed"] is False


def test_missing_finite_nets_are_skipped_not_counted_as_zero():
    records = [
        _record(0, _all_engines({"A": float("nan")})),
        _record(1, _all_engines({"A": 0.003})),
    ]
    out = rx.config_averaged_existence(records, arm="A")
    assert out["n_folds"] == 1
    assert out["folds_improved"] == 1


def test_no_folds_reports_an_unpassed_verdict():
    out = rx.config_averaged_existence([], arm="A")
    assert out["n_folds"] == 0
    assert out["passed"] is False
    assert np.isnan(out["mean_diff"])
