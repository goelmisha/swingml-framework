"""Typed configuration layer.

Every downstream step (labelling, validation, model, backtest) imports its
settings from here. The point of this module is to make accidental divergence
impossible: there is exactly one definition of the friction cost, one label
horizon, one walk-forward geometry.

Usage
-----
    from swingml.config import load_config
    cfg = load_config("config/config.yaml")
    cfg.costs.round_trip_cost_pct   # 0.0025 -- mandatory 0.25% round trip
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Project root = parent of the `swingml` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    dataset_dir: str = "data/datasets"

    def resolve(self, root: Path = PROJECT_ROOT) -> "PathsConfig":
        """Return a copy with every path anchored to the project root."""
        return PathsConfig(
            data_dir=str(root / self.data_dir),
            raw_dir=str(root / self.raw_dir),
            cache_dir=str(root / self.cache_dir),
            dataset_dir=str(root / self.dataset_dir),
        )


@dataclass
class MockConfig:
    n_symbols: int = 40
    n_days: int = 1200
    seed: int = 7


@dataclass
class DataConfig:
    start: str = "2020-01-01"
    end: str | None = None
    price_provider: str = "yfinance"
    delivery_provider: str = "nse_bhavcopy"
    bench_symbol: str = "^NSEI"
    #: Breeze Connect (same-day complete prices). The App Key and Secret Key are
    #: read from BREEZE_API_KEY / BREEZE_SECRET_KEY in the environment and are
    #: never stored; only the daily session token is persisted, under data/.
    breeze_session_path: str = "data/breeze_session.json"
    #: Cached snapshot of ICICI's security master, the NSE-symbol -> Breeze
    #: ``stock_code`` map. Breeze does not accept the NSE symbol for most names
    #: (RELIANCE is ``RELIND``), so this is what makes the provider usable.
    breeze_master_path: str = "data/breeze_master.json"
    breeze_bench_symbol: str = "NIFTY"
    #: Fill the newest session's OHLC from the same-evening bhavcopy when the
    #: price provider does not have it yet (Yahoo's constituent bars arrive over
    #: hours; its index does not). A no-op whenever the provider already has the
    #: bar, so it is inert in a backfill -- see ``dataset.overlay_tail_session``.
    sameday_overlay: bool = True
    #: Resolve retired tickers to the symbol that replaced them, using the
    #: ISIN-keyed point-in-time symbol history. Without it the feature matrix
    #: silently drops every renamed universe member -- a survivorship bias in a
    #: panel that is meant to be survivorship-free. See ``data/symbol_history.py``.
    symbol_history: bool = True
    symbol_history_path: str = "data/symbol_history.json"
    max_workers: int = 6
    request_delay_sec: float = 0.35
    request_timeout_sec: int = 30
    max_retries: int = 4
    mock: MockConfig = field(default_factory=MockConfig)

    @property
    def start_date(self) -> dt.date:
        return dt.date.fromisoformat(self.start)

    @property
    def end_date(self) -> dt.date:
        """Inclusive end date; today when unspecified."""
        return dt.date.fromisoformat(self.end) if self.end else dt.date.today()


@dataclass
class UniverseConfig:
    """Tradable-universe selection.

    The defaults here are deliberately the *mechanism*, not a policy: the
    point-in-time liquidity screen is survivorship-free, whereas an exchange
    index list is convenient but biased by construction. Which universe you
    actually trade belongs in the YAML config, not in a schema default.
    """

    source: str = "liquidity"
    index_name: str = ""  # required only when source is an index list
    universe_size: int = 200
    symbols: list[str] | None = None
    min_price: float = 20.0
    min_avg_turnover_lacs: float = 100.0


@dataclass
class FeatureConfig:
    """Feature-set selection plus the window parameters providers read.

    ``provider`` names the :class:`~swingml.features.base.FeatureProvider` that
    builds the matrix. It may be a registered/entry-point name (``demo``) or a
    ``"module:Class"`` dotted path, which is how a feature set kept in a private
    package is wired in without the framework needing to know about it.
    """

    provider: str = "demo"
    ema_windows: list[int] = field(default_factory=lambda: [20, 50, 200])
    rsi_window: int = 14
    roc_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    atr_window: int = 14
    vol_windows: list[int] = field(default_factory=lambda: [5, 20, 60])
    delivery_windows: list[int] = field(default_factory=lambda: [5, 20])
    zscore_window: int = 20
    realized_vol_windows: list[int] = field(default_factory=lambda: [10, 20])
    range_window: int = 252
    cross_sectional_ranks: bool = True
    min_history_days: int = 260


@dataclass
class LabelConfig:
    """Triple-barrier geometry.

    The values here are ordinary placeholders that make the pipeline run; the
    geometry you actually trade belongs in the YAML config. They are not shipped
    as tuned numbers for the same reason no other score-affecting default is.

    ``entry_price``
        ``next_open`` (default) enters at the open of the bar after the signal.
        Entering at the signal bar's own close assumes you could trade at a price
        you had only just observed. ``close`` is available for comparison but is
        the optimistic convention.
    ``same_bar_resolution``
        When one daily bar spans both barriers the intrabar order is unknowable.
        ``pessimistic`` (default) books the stop loss; ``optimistic`` books the
        take profit and should only ever be used as a sensitivity check.
    ``mode``
        ``barrier`` (default) is the triple-barrier method: the trade exits at
        whichever horizontal barrier is reached first. ``fixed_hold`` exits at
        the vertical barrier only, and the label is whether that hold was
        profitable net of friction -- definition C as a training target.

The mode is not cosmetic: the exit rule is what the model learns, so it is a
first-class config choice, not a labelling detail.
    """

    mode: str = "barrier"
    horizon_days: int = 5
    pt_atr_mult: float = 1.5
    sl_atr_mult: float = 1.0
    atr_window: int = 20
    entry_price: str = "next_open"
    same_bar_resolution: str = "pessimistic"

    #: When a label is even emitted. ``every_bar`` (default) labels every bar;
    #: ``cusum`` labels only at symmetric-CUSUM events. This
    #: changes the training *distribution*, not the target, so it composes with
    #: both exit rules. The ``h_mult`` choice is a trial -- record it.
    sampling: str = "every_bar"
    cusum_h_mult: float = 1.5
    cusum_vol_span: int = 20


@dataclass
class ValidationConfig:
    """Walk-forward geometry. Random K-Fold is banned (it leaks across time).

    On the purge/horizon interaction
    ---------------------------------
    A 10-day label at bar ``t`` is only fully resolved 10 bars later. A purge
    *shorter* than the label horizon therefore still leaves training labels whose
    outcome window reaches into the test block -- the classic overlap leak.

    Rather than reject the requested ``purge_gap_days`` outright, we honour it as
    the *floor* and widen it automatically via :meth:`effective_purge_days`. That
    way a user asking for a 5-day purge gets at least 5 bars of separation, and
    the split is silently upgraded to 10 to stay leak-free.
    """

    train_days: int = 252
    test_days: int = 63
    purge_gap_days: int = 5
    embargo_days: int = 5

    def effective_purge_days(self, horizon_days: int) -> int:
        """Purge actually used: the requested floor, widened to cover the label
        horizon so no training outcome window overlaps the test block."""
        return max(self.purge_gap_days, horizon_days)

    def effective_embargo_days(self, horizon_days: int) -> int:
        """Feature windows overlap too, so embargo gets the same treatment."""
        return max(self.embargo_days, horizon_days)


@dataclass
class CostsConfig:
    """Trading friction. The 0.25% round trip is mandatory and non-negotiable."""

    round_trip_cost_pct: float = 0.0025
    enforce_in_backtest: bool = True

    def net_return(self, gross_return: float) -> float:
        """Apply the mandatory round-trip haircut to a gross trade return."""
        return gross_return - self.round_trip_cost_pct

    def breach_threshold(self, gross_threshold: float) -> float:
        """Gross move required to net ``gross_threshold`` after friction."""
        return gross_threshold + self.round_trip_cost_pct


@dataclass
class MetaConfig:
    """Meta-labelling geometry.

    A primary model ranks candidates at ``primary_frac`` (recall-first); a
    secondary model is trained only on the primary's positive calls and learns
    **whether that call paid**, net of friction (definition B). The primary's
    score is a feature of the secondary, so the secondary can condition on what
    the primary believed rather than re-deriving it.

    ``primary_frac`` must be wider than the trading selection (1/decile), or
    the secondary would have nothing left to filter.
    """

    primary_frac: float = 0.30
    #: Number of CPCV groups used to generate the primary's out-of-fold scores
    #: on the training block. Without OOF scores the secondary would learn from
    #: the primary's in-sample fit, which is optimistic by construction.
    oof_groups: int = 5


@dataclass
class ModelsConfig:
    """Shared hyperparameters for every engine.

    One set across engines rather than per-engine tuning, on purpose: the value
    of the multi-engine protocol is measuring how much a conclusion depends on
    the implementation, which is impossible if each engine is tuned separately.

    These are ordinary starting values, not recommendations, and they are read
    from the YAML config so that no tuned number is published in the code.
    """

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 20
    l2: float = 1.0


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


@dataclass
class ExperimentConfig:
    """Column selections the experiment scripts need.

    Named here rather than inside the scripts so that no script hard-codes a
    column that only one feature provider emits: a script asks for "the regime
    column", and the config says which column that is for this deployment.
    """

    #: Feature column that splits sessions into regimes for the per-regime A/B
    #: test. Arm B trains one model per distinct value; arm A trains a single
    #: model with this column as an ordinary input.
    regime_col: str = ""

    #: Columns the signal check sweeps. ``None`` sweeps every cross-sectional
    #: rank column the provider emitted.
    signal_features: list[str] | None = None

    #: Columns each signal feature is correlated against. ``None`` uses the
    #: same default as :attr:`signal_features`.
    control_features: list[str] | None = None

    def require_regime_col(self) -> str:
        """The regime column, or a clear error naming the config key."""
        col = (self.regime_col or "").strip()
        if not col:
            raise ValueError(
                "experiment.regime_col is unset: name the feature column that "
                "splits sessions into regimes (see config/config.yaml)"
            )
        return col


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    root: Path = PROJECT_ROOT

    # -- convenience -------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create every directory this pipeline writes into."""
        for p in (
            self.paths.data_dir,
            self.paths.raw_dir,
            self.paths.cache_dir,
            self.paths.dataset_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Fail loudly on configurations that would silently produce garbage."""
        if self.costs.round_trip_cost_pct <= 0:
            raise ValueError("costs.round_trip_cost_pct must be > 0 (friction is mandatory)")
        if self.label.horizon_days < 1:
            raise ValueError("label.horizon_days must be >= 1")
        if self.label.pt_atr_mult <= 0 or self.label.sl_atr_mult <= 0:
            raise ValueError("barrier multiples must be > 0")
        if self.label.entry_price not in {"next_open", "close"}:
            raise ValueError("label.entry_price must be 'next_open' or 'close'")
        if self.label.same_bar_resolution not in {"pessimistic", "optimistic"}:
            raise ValueError("label.same_bar_resolution must be 'pessimistic' or 'optimistic'")
        if self.label.mode not in {"barrier", "fixed_hold"}:
            raise ValueError(
                f"label.mode must be 'barrier' or 'fixed_hold', got {self.label.mode!r}"
            )
        if self.label.sampling not in {"every_bar", "cusum"}:
            raise ValueError(
                f"label.sampling must be 'every_bar' or 'cusum', got {self.label.sampling!r}"
            )
        _positive("label.cusum_h_mult", self.label.cusum_h_mult)
        if self.label.cusum_vol_span < 2:
            raise ValueError("label.cusum_vol_span must be >= 2")
        if not 0.0 < self.meta.primary_frac < 1.0:
            raise ValueError("meta.primary_frac must be in (0, 1)")
        if self.meta.oof_groups < 2:
            raise ValueError("meta.oof_groups must be >= 2")
        if self.label.mode == "fixed_hold" and (
            abs(self.label.pt_atr_mult - self.label.sl_atr_mult) > 1e-9
        ):
            # Not an error: the multipliers are simply unused in this mode, and
            # silently ignoring a number the reader sees in the YAML is exactly
            # how a config stops being a single source of truth.
            logger.warning(
                "label.mode=fixed_hold ignores pt_atr_mult/sl_atr_mult (%s/%s); "
                "the trade exits at the vertical barrier only",
                self.label.pt_atr_mult, self.label.sl_atr_mult,
            )
        if self.label.atr_window not in self.features.ema_windows and self.label.atr_window != 20:
            # The labeller reads the `atr_20` context column by name.
            logger.warning(
                "label.atr_window=%s but the feature matrix only carries `atr_20`; "
                "barrier width will use the 20-day ATR", self.label.atr_window,
            )
        if self.validation.purge_gap_days < 0:
            raise ValueError("validation.purge_gap_days cannot be negative")
        # A purge shorter than the label horizon is widened automatically rather
        # than rejected; warn so the widening is visible in the logs.
        if self.validation.purge_gap_days < self.label.horizon_days:
            eff = self.validation.effective_purge_days(self.label.horizon_days)
            logger.warning(
                "purge_gap_days (%s) is shorter than label.horizon_days (%s); the purge will be "
                "widened to %s bars so no training label overlaps the test block",
                self.validation.purge_gap_days, self.label.horizon_days, eff,
            )
        if self.label.atr_window > self.features.min_history_days:
            raise ValueError("label.atr_window exceeds features.min_history_days warm-up")
        if self.data.start_date >= self.data.end_date:
            raise ValueError("data.start must be before data.end")
        if self.universe.source not in {"nifty200", "nifty500", "liquidity"}:
            raise ValueError(f"unknown universe.source: {self.universe.source}")
        if self.universe.source in {"nifty200", "nifty500"} and not self.universe.index_name:
            raise ValueError(
                f"universe.index_name is required when universe.source is "
                f"{self.universe.source!r}"
            )
        for field_name in ("n_estimators", "max_depth", "min_samples_leaf"):
            if getattr(self.models, field_name) < 1:
                raise ValueError(f"models.{field_name} must be >= 1")
        _positive("models.learning_rate", self.models.learning_rate)
        _positive("models.l2", self.models.l2)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _build(cls: type[T], raw: Any, path: str = "") -> T:
    """Recursively instantiate a nested dataclass tree from plain dicts."""
    if raw is None:
        return cls()
    if not dataclasses.is_dataclass(cls):
        return raw  # type: ignore[return-value]
    if not isinstance(raw, dict):
        raise TypeError(f"expected a mapping for '{path or cls.__name__}', got {type(raw).__name__}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        key = f.name
        if key not in raw:
            continue  # keep the dataclass default
        value = raw[key]
        ftype = hints.get(key, f.type)
        if dataclasses.is_dataclass(ftype):
            kwargs[key] = _build(ftype, value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        # Warn instead of crashing: a stale key is usually a typo worth surfacing.
        logger.warning("ignoring unknown config key(s) under '%s': %s", path or "<root>", sorted(unknown))
    return cls(**kwargs)


def load_config(path: str | Path | None = None, root: Path = PROJECT_ROOT) -> AppConfig:
    """Load, type-check and validate the YAML config into an :class:`AppConfig`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = _build(AppConfig, raw)
    # `root` is not a YAML section; set it explicitly then anchor all paths.
    cfg.root = root
    cfg.paths = cfg.paths.resolve(root)
    cfg.validate()
    return cfg


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent root-logger setup shared by every CLI entry point."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )
