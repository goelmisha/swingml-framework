"""Tests for the splitter-geometry sweep.

Seeds are inert in the configured engines (STATUS section 8 item 8, VOID), so
geometry is the only validation axis left to vary. These tests pin the two
properties that make the sweep a fair robustness test rather than six arbitrary
runs: every variant moves off the *configured* geometry, and each one stays a
valid leak-free walk-forward cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.splitter_robustness as sr
from swingml.config import ValidationConfig
from swingml.validation import walk_forward_splits


BASE = ValidationConfig(train_days=504, test_days=126, purge_gap_days=5, embargo_days=5)


def _frame(n_sessions: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_sessions)
    return pd.DataFrame({"date": dates})


# ---------------------------------------------------------------------------
# geometry_variants
# ---------------------------------------------------------------------------

def test_first_variant_is_the_configured_geometry():
    geo = sr.geometry_variants(BASE)[0]
    assert (geo.train_days, geo.test_days) == (BASE.train_days, BASE.test_days)
    assert (geo.purge_gap_days, geo.embargo_days) == (BASE.purge_gap_days, BASE.embargo_days)


def test_variants_are_uniquely_named_and_non_degenerate():
    variants = sr.geometry_variants(BASE)
    assert len({g.name for g in variants}) == len(variants)
    for g in variants:
        assert g.train_days >= 1 and g.test_days >= 1


def test_variants_track_the_configured_geometry():
    """Editing `validation:` in the config must move the whole sweep."""
    rescaled = sr.geometry_variants(ValidationConfig(train_days=1000, test_days=200))
    by_name = {g.name: g for g in rescaled}
    assert by_name["longer train (1.5x)"].train_days == 1500
    assert by_name["shorter train (0.75x)"].train_days == 750
    assert by_name["shorter test (0.5x)"].test_days == 100
    assert by_name["longer test (1.5x)"].test_days == 300


def test_each_axis_moves_alone_and_the_purge_variant_only_widens():
    by_name = {g.name: g for g in sr.geometry_variants(BASE)}
    assert by_name["longer train (1.5x)"].test_days == BASE.test_days
    assert by_name["shorter test (0.5x)"].train_days == BASE.train_days
    wide = by_name["wide purge+embargo (20/20)"]
    assert (wide.train_days, wide.test_days) == (BASE.train_days, BASE.test_days)
    assert wide.purge_gap_days >= BASE.purge_gap_days
    assert wide.embargo_days >= BASE.embargo_days


def test_to_config_keeps_untouched_fields_and_yields_folds():
    """A variant is a ValidationConfig copy, and it still splits real data."""
    geo = sr.geometry_variants(BASE)[1]
    cfg = geo.to_config(BASE)
    assert isinstance(cfg, ValidationConfig)
    assert (cfg.train_days, cfg.test_days) == (geo.train_days, geo.test_days)

    df = _frame(1500)
    folds = walk_forward_splits(df, cfg, label_horizon=10)
    assert folds, "variant geometry produced no folds"
    # The horizon must floor the purge: a 10-session label cannot be purged 5 days.
    assert all(f.gap_sessions >= 10 for f in folds)


def test_shortest_variant_still_produces_folds_for_the_real_span():
    """1,465 sessions is the liquidity panel; the smallest variant must fit it."""
    df = _frame(1465)
    for geo in sr.geometry_variants(BASE):
        folds = walk_forward_splits(df, geo.to_config(BASE), label_horizon=10)
        assert folds, f"{geo.name} produced no folds over 1,465 sessions"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_label_mode_is_read_from_the_dataset_not_the_config(tmp_path):
    """The labels on disk decide the target; a stale config must not relabel them."""
    assert sr.dataset_label_mode(tmp_path) is None
    (tmp_path / "label_diagnostics.json").write_text('{"mode": "fixed_hold"}')
    assert sr.dataset_label_mode(tmp_path) == "fixed_hold"
    (tmp_path / "label_diagnostics.json").write_text("not json")
    assert sr.dataset_label_mode(tmp_path) is None


def test_cum_compounds_and_tolerates_nan():
    s = pd.Series([0.10, -0.05, np.nan])
    assert sr._cum(s) == np.float64(1.10 * 0.95 - 1.0)
    assert np.isnan(sr._cum(pd.Series(dtype=float)))
