"""Config tests: the friction cost and the purge/horizon invariant must be enforced."""

from __future__ import annotations

import pytest
import yaml

from swingml.config import PROJECT_ROOT, AppConfig, CostsConfig, load_config


def test_mock_section_is_nested_under_data():
    """Regression: a root-level `mock:` block is silently ignored by the loader."""
    cfg = load_config()
    assert cfg.data.mock.n_symbols == 40
    assert cfg.data.mock.seed == 7


def test_default_config_loads_and_validates():
    """The YAML file is the single source of truth.

    Asserted by round-tripping against the file rather than against literals, so
    this test keeps working whichever values a deployment chooses to ship.
    """
    cfg = load_config()
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

    assert cfg.costs.round_trip_cost_pct == pytest.approx(raw["costs"]["round_trip_cost_pct"])
    assert cfg.label.horizon_days == raw["label"]["horizon_days"]
    assert cfg.label.pt_atr_mult == pytest.approx(raw["label"]["pt_atr_mult"])
    assert cfg.label.sl_atr_mult == pytest.approx(raw["label"]["sl_atr_mult"])
    assert cfg.features.provider == raw["features"]["provider"]

    # Invariants that must hold for ANY configuration.
    assert cfg.costs.round_trip_cost_pct > 0
    # The requested purge is only a floor; the effective value must cover the
    # label horizon, otherwise training outcomes overlap the test block.
    assert cfg.validation.effective_purge_days(cfg.label.horizon_days) >= cfg.label.horizon_days
    # Paths must be anchored to the project root, not the CWD.
    assert cfg.paths.dataset_dir.endswith("datasets")


def test_friction_is_applied_to_returns():
    c = CostsConfig(round_trip_cost_pct=0.0025)
    assert c.net_return(0.05) == pytest.approx(0.0475)
    # A gross 0.2% move is a net LOSS once friction is paid -- this is the whole
    # reason a naive "up/down" label looks profitable and a real one does not.
    assert c.net_return(0.002) < 0


def test_zero_friction_is_rejected():
    cfg = AppConfig()
    cfg.costs.round_trip_cost_pct = 0.0
    with pytest.raises(ValueError, match="friction"):
        cfg.validate()


def test_short_purge_is_widened_not_obeyed():
    """A 10-day label with a 5-day purge would overlap train and test outcomes.

    The requested value is treated as a floor and silently widened, so a user
    asking for 5 bars of separation always gets at least 5 -- and at least 10
    when the label horizon demands it.
    """
    cfg = AppConfig()
    cfg.label.horizon_days = 10
    cfg.validation.purge_gap_days = 5
    cfg.validate()  # must NOT raise: widening is the resolution
    assert cfg.validation.effective_purge_days(cfg.label.horizon_days) == 10
    assert cfg.validation.effective_embargo_days(cfg.label.horizon_days) == 10


def test_generous_purge_is_respected_verbatim():
    cfg = AppConfig()
    cfg.label.horizon_days = 10
    cfg.validation.purge_gap_days = 25
    cfg.validate()
    assert cfg.validation.effective_purge_days(cfg.label.horizon_days) == 25


def test_negative_purge_is_rejected():
    cfg = AppConfig()
    cfg.validation.purge_gap_days = -1
    with pytest.raises(ValueError, match="purge_gap_days"):
        cfg.validate()


def test_start_after_end_is_rejected():
    cfg = AppConfig()
    cfg.data.start = "2026-01-01"
    cfg.data.end = "2025-01-01"
    with pytest.raises(ValueError, match="start must be before"):
        cfg.validate()


def test_invalid_model_hyperparameters_are_rejected():
    """A non-positive learning rate or depth is a silent no-op, not an error."""
    cfg = AppConfig()
    cfg.models.learning_rate = 0.0
    with pytest.raises(ValueError, match="models.learning_rate"):
        cfg.validate()

    cfg = AppConfig()
    cfg.models.max_depth = 0
    with pytest.raises(ValueError, match="models.max_depth"):
        cfg.validate()


def test_unknown_source_is_rejected():
    cfg = AppConfig()
    cfg.universe.source = "dow30"
    with pytest.raises(ValueError, match="universe.source"):
        cfg.validate()


def test_index_source_requires_an_index_name():
    """The traded index must be stated explicitly, never inherited from a default."""
    cfg = AppConfig()
    cfg.universe.source = "nifty200"
    cfg.universe.index_name = ""
    with pytest.raises(ValueError, match="index_name"):
        cfg.validate()

    cfg.universe.index_name = "NIFTY 500"
    cfg.validate()  # must not raise


def test_unknown_yaml_keys_warn_not_crash(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("costs:\n  round_trip_cost_pct: 0.0025\n  typo_key: 1\n", encoding="utf-8")
    cfg = load_config(p, root=tmp_path)
    assert cfg.costs.round_trip_cost_pct == pytest.approx(0.0025)
