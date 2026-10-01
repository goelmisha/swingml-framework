"""Forward paper trading -- freeze the prediction before the open, then grade it.

Why this module exists
----------------------
Every number in this repo is historical. Gate 1a (STATUS section 9) needs the one
measurement a backtest cannot manufacture: a prediction written down **before**
the market it predicts. That is the entire value of the exercise, and it is easy
to lose -- a "forward test" whose log can be regenerated later is just another
backtest with extra steps.

Four rules make the log evidence rather than a note:

1. **Append-only and hash-chained.** Each record carries the hash of the previous
   one, so :func:`verify_chain` detects a later edit, insertion or deletion.
   Back-dating a winner is then a detectable act, not an invisible one.
2. **Full candidate score vector.** The record stores every candidate's score,
   not just the winners, so the benchmark -- the same trade on the whole universe
   -- can be recomputed at scoring time without refitting anything.
3. **No ledger writes.** Paper runs never append to ``data/trials.jsonl``. The
   ledger counts configurations *searched*; one entry per forward day would
   inflate the DSR's search penalty for a decision that involved no search at
   all (the configuration was counted once, by the backtest that chose it).
4. **No broker.** There is no order path, no Breeze import and no network call in
   the freeze path. This module cannot move money even if misused.

Why the scoring grid is pre-registered
--------------------------------------
A 10-session hold rebalanced *every* session produces overlapping periods, and a
Sharpe computed on overlapping periods is meaningless -- it counts one trade many
times. So the headline forward series is taken on a **pre-registered
non-overlapping grid**: every ``horizon``-th session counted from the dataset's
first session. Grid membership is decided on the signal date and stored in the
record, so it can never be chosen after seeing which periods won.

The anchor derives from the dataset's first session, which the config pins at
2020-01-01, so it does not drift as sessions accumulate. Off-grid records are
still frozen (they cost nothing and preserve the option of a tranched variant)
but are labelled ``off_grid`` and excluded from the headline.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from swingml.scorecard import SESSIONS_PER_YEAR, beta_analysis, performance_metrics

logger = logging.getLogger(__name__)

#: Bumped when the record layout changes; old records stay readable but are
#: flagged by :func:`verify_chain` so a silent format drift cannot happen.
PAPER_SCHEMA_VERSION = 1

#: IST is UTC+5:30 with no daylight saving, ever, so a fixed offset is exact and
#: avoids a tzdata dependency on a machine that may not have one.
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

#: NSE regular session opens at 09:15 IST. The entry price is the open of the
#: first session *after* the signal date, so this is the moment a prediction
#: stops being forward evidence.
MARKET_OPEN = dt.time(9, 15)

#: How far the signal date's candidate count may fall below its recent median
#: before the freeze is refused as an incomplete cross-section.
CROSS_SECTION_TOLERANCE = 0.9

#: Sessions of history used to establish that median.
CROSS_SECTION_LOOKBACK = 20

#: Default log location, relative to the repo root. Deliberately NOT under
#: ``data/`` -- that directory is gitignored, and this file is the audit trail.
DEFAULT_LOG_PATH = "paper/predictions.jsonl"

#: Scoring statuses. Only ``closed`` records enter the summary series.
STATUS_CLOSED = "closed"
STATUS_OPEN = "open"
STATUS_OFF_GRID = "off_grid"
STATUS_BACKFILL = "backfill"
STATUS_INCOMPLETE = "incomplete"
STATUS_NO_SESSION = "no_session"
STATUS_NO_ENTRY = "no_entry"


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def code_fingerprint(repo_root: str | Path = ".") -> str:
    """Git revision (plus ``-dirty``) so a record names the code that made it.

    Best effort: a tarball checkout or a machine without git yields ``unknown``
    rather than an exception, because losing the fingerprint must not lose the
    prediction.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}"
    except Exception as exc:  # pragma: no cover - depends on the checkout
        logger.warning("could not fingerprint code: %s", exc)
        return "unknown"


def config_fingerprint(
    cfg,
    engine: str,
    decile: int,
    selector: str = "decile",
    dataset_facts: dict | None = None,
) -> tuple[str, dict]:
    """The settings that determine a prediction, hashed, plus the settings.

    Anything that can move a score belongs here: the target mode and horizon,
    the sampling rule, the engine and its hyperparameters, the selector (single
    model vs meta-labelled), the universe screen, the feature provider and the
    friction. The dict is stored alongside the hash so a record stays
    interpretable even after ``config/config.yaml`` changes.

    ``dataset_facts`` supplies the two settings that belong to the *dataset*
    rather than to the config: the target mode its labels were built with, and
    the universe it was built over. Both are read back from the artifact, and
    the same rule already applies elsewhere
    (``test_label_mode_is_read_from_the_dataset_not_the_config``). The config
    default is not evidence about a dataset -- the default universe is
    ``nifty200`` for fast iteration while every validated result is over
    ``liquidity``, so trusting the config here would stamp the genesis record of
    the forward log with a universe it was never fitted to.
    """
    facts = dataset_facts or {}
    payload = {
        "engine": engine,
        "decile": int(decile),
        "selector": str(selector),
        "meta": dataclasses.asdict(cfg.meta) if hasattr(cfg, "meta") else None,
        "label_mode": facts.get("label_mode") or getattr(cfg.label, "mode", None),
        "label_sampling": getattr(cfg.label, "sampling", None),
        "cusum_h_mult": float(getattr(cfg.label, "cusum_h_mult", 0.0) or 0.0),
        "cusum_vol_span": int(getattr(cfg.label, "cusum_vol_span", 0) or 0),
        "horizon_days": int(cfg.label.horizon_days),
        "pt_atr_mult": float(cfg.label.pt_atr_mult),
        "sl_atr_mult": float(cfg.label.sl_atr_mult),
        "atr_window": int(cfg.label.atr_window),
        "models": dataclasses.asdict(cfg.models),
        "universe_source": facts.get("universe_source") or getattr(cfg.universe, "source", None),
        "universe_size": getattr(cfg.universe, "universe_size", None),
        "min_price": getattr(cfg.universe, "min_price", None),
        "min_avg_turnover_lacs": getattr(cfg.universe, "min_avg_turnover_lacs", None),
        "feature_provider": getattr(cfg.features, "provider", None),
        "regime_col": getattr(cfg.experiment, "regime_col", None),
        "round_trip_cost_pct": float(cfg.costs.round_trip_cost_pct),
        "validation": dataclasses.asdict(cfg.validation),
    }
    return _hash(payload), payload


def _hash(payload) -> str:
    """Canonical SHA-256 of a JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# forward vs backfill -- the test is about the entry PRICE, not the calendar
# ---------------------------------------------------------------------------

def next_weekday(day: dt.date) -> dt.date:
    """The next Mon-Fri after ``day``. Weekends need no calendar to skip."""
    nxt = day + dt.timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += dt.timedelta(days=1)
    return nxt


def entry_session_open(signal_date) -> dt.datetime:
    """When the entry session could first have traded: 09:15 IST next weekday.

    An **approximation** of the NSE calendar that is deliberately wrong in the
    safe direction. A hidden holiday moves the real open *later* than this, so
    the only error it can make is calling a still-forward record a backfill --
    never the reverse. Losing one observation is recoverable; admitting a
    backdated one is not.
    """
    day = pd.Timestamp(signal_date).date()
    return dt.datetime.combine(next_weekday(day), MARKET_OPEN, tzinfo=IST)


def cross_section_ok(
    n_candidates: int,
    recent_counts,
    tolerance: float = CROSS_SECTION_TOLERANCE,
) -> tuple[bool, int]:
    """Is the signal date's cross-section complete enough to rank?

    The newest session's per-symbol prices arrive from the provider over the
    hours after the close, so a freeze run too early ranks a **truncated
    universe** skewed to the large caps that update first.

    That is a different experiment from the backtested rule, and unlike a
    missing session it would bias every record in the log rather than one, so it
    is refused instead of recorded quietly. Returns ``(ok, recent_median)``.
    """
    counts = [int(c) for c in recent_counts if c is not None and int(c) > 0]
    if len(counts) < 5:
        # Too little history to judge; a brand-new panel must still be usable.
        return True, 0
    median = int(np.median(counts))
    if median <= 0:
        return True, median
    return n_candidates >= tolerance * median, median


def is_forward_record(signal_date, now: dt.datetime | None = None) -> bool:
    """Is this prediction written before the entry session could trade?

    The question is about the **entry price**, not the date arithmetic. Running
    the evening of the signal session (staleness 0) is forward, and so is
    running the next morning before 09:15 -- which is the workflow the daily
    routine actually uses, because yesterday's bhavcopy is what is published
    overnight. Running days later is not: the entry has been observed, and the
    model is deterministic, so the record is indistinguishable from one fitted
    to an open that already happened.
    """
    clock = now or dt.datetime.now(IST)
    if clock.tzinfo is None:  # a naive argument is taken to be IST
        clock = clock.replace(tzinfo=IST)
    return clock < entry_session_open(signal_date)


# ---------------------------------------------------------------------------
# selection -- must agree with the backtest, or the forward test measures
# something else
# ---------------------------------------------------------------------------

def selection_size(n_candidates: int, decile: int) -> int:
    """How many names `top_fraction_mask` would keep out of ``n_candidates``.

    ``top_fraction_mask`` keeps the rows whose within-date percentile rank
    exceeds ``1 - 1/decile``, which is ``n - floor(n * (decile-1) / decile)``
    names (46 of 460 at decile 10). Reproduced here so the live selection can be
    built from a plain score vector without a DataFrame.
    """
    if decile < 1:
        raise ValueError("decile must be >= 1")
    if n_candidates <= 0:
        return 0
    return int(n_candidates - int(n_candidates * (decile - 1) / decile))


def select_top(scores: Mapping[str, float], decile: int = 10) -> list[str]:
    """The top ``1/decile`` symbols by score, ties broken by symbol name.

    Deterministic on purpose: the same score vector must always produce the same
    list, or the frozen record is not a commitment. Sorting by ``(-score,
    symbol)`` makes the tie-break a property of the data rather than of dict
    iteration order.

    Verified against :func:`swingml.evaluation.top_fraction_mask` in
    ``tests/test_paper.py`` -- if the two ever disagree, the forward test would
    be measuring a different rule than the backtest.
    """
    items = [(sym, float(sc)) for sym, sc in scores.items() if np.isfinite(float(sc))]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    k = selection_size(len(items), decile)
    return sorted(sym for sym, _ in items[:k])


# ---------------------------------------------------------------------------
# the record and its chain
# ---------------------------------------------------------------------------

@dataclass
class PaperRecord:
    """One frozen prediction, for one signal date.

    ``is_grid`` is decided and stored at freeze time, never at scoring time --
    that is what stops the non-overlapping sample being chosen after seeing the
    results.
    """

    schema_version: int
    timestamp: str
    signal_date: str
    session_index: int
    is_grid: bool
    #: True only when the freeze happened on or before the signal date itself,
    #: i.e. before the entry session could have traded. A record written the
    #: next morning is a *backfill*: the entry price already exists, the model
    #: could have been chosen to fit it, and it is therefore not evidence. It is
    #: kept in the log and excluded from the headline series.
    is_forward: bool
    horizon_days: int
    decile: int
    engine: str
    label_mode: str | None
    dataset_dir: str
    n_candidates: int
    selected: list[str]
    scores: dict[str, float]
    config_hash: str
    config: dict
    code_hash: str
    n_train_rows: int = 0
    #: ``decile`` (single model) or ``meta`` (primary + meta-labelled filter).
    #: Defaults to ``decile`` so a record written before this field existed is
    #: read as the selector it actually used.
    selector: str = "decile"
    train_cutoff: str | None = None
    #: ``meta.json``'s ``built_at`` for the panel this prediction read, so the
    #: exact build can be identified even after the directory is rebuilt.
    dataset_built_at: str | None = None
    #: The candidate count the signal date offered against its recent median, and
    #: whether that gap was small enough to rank on. Recorded so a truncated
    #: cross-section is visible in the log rather than inferred later.
    n_candidates_recent: int = 0
    #: Defaults to False so a record that predates this flag -- or one written by
    #: something that does not check -- is treated as ineligible rather than
    #: admitted by omission.
    cross_section_ok: bool = False
    prev_hash: str = ""
    supersedes: str | None = None
    record_hash: str = ""
    #: The exact JSON object this record was read from, when it came off disk.
    #: Verification re-hashes *this*, not a reconstruction: otherwise adding a
    #: field to this dataclass would silently invalidate every record written
    #: before it existed -- which is how a five-minute-old log broke once.
    raw: dict | None = dataclasses.field(default=None, repr=False, compare=False)

    #: Fields excluded from the hashed payload.
    _UNHASHED = ("record_hash", "raw")

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    def body(self) -> dict:
        """Every hashable field, in a canonical order. Used when writing."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in self._UNHASHED
        }

    def stored_payload(self) -> dict:
        """What to hash: exactly what was written, when that is known."""
        if self.raw is None:
            return self.body()
        return {k: v for k, v in self.raw.items() if k != "record_hash"}


def _record_from_dict(raw: dict) -> PaperRecord:
    known = {f.name for f in dataclasses.fields(PaperRecord)}
    unknown = set(raw) - known
    if unknown:
        # Refuse rather than ignore: a field this reader does not understand
        # could be one the writer considered load-bearing.
        raise ValueError(f"paper record has unknown field(s): {sorted(unknown)}")
    return PaperRecord(raw=raw, **raw)


def new_record(**fields) -> PaperRecord:
    """Build a record with the schema version and UTC timestamp stamped on."""
    return PaperRecord(
        schema_version=PAPER_SCHEMA_VERSION,
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **fields,
    )


def load_records(log_path: str | Path) -> list[PaperRecord]:
    """Read the JSONL log in file order. A missing file is an empty log."""
    path = Path(log_path)
    if not path.exists():
        return []
    out: list[PaperRecord] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(_record_from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: unreadable paper record ({exc})") from exc
    return out


def append_record(log_path: str | Path, rec: PaperRecord) -> PaperRecord:
    """Chain, hash and append one record. Returns the stamped record.

    Re-freezing a signal date is allowed only when it explicitly names the
    record it replaces, so a correction is visible in the log instead of
    quietly overwriting history.
    """
    path = Path(log_path)
    records = load_records(path)

    existing = [r for r in records if r.signal_date == rec.signal_date]
    if existing:
        if rec.supersedes is None:
            raise ValueError(
                f"a prediction for {rec.signal_date} is already frozen "
                f"({existing[-1].record_hash[:12]}); pass supersedes= to replace it"
            )
        if rec.supersedes not in {r.record_hash for r in records}:
            raise ValueError("supersedes must reference an existing record_hash")

    rec.prev_hash = records[-1].record_hash if records else ""
    payload = rec.body()
    rec.record_hash = _hash(payload)
    written = payload | {"record_hash": rec.record_hash}
    rec.raw = written

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(written, sort_keys=True) + "\n")
    return rec


def verify_chain(records: list[PaperRecord]) -> list[str]:
    """Return the integrity problems in a log; empty means the chain is intact.

    Checks the hash of every record and that each links to its predecessor, so
    an edited score, a deleted loser or an inserted winner all surface.
    """
    problems: list[str] = []
    seen: set[str] = set()
    previous = ""
    for i, rec in enumerate(records):
        tag = f"record {i} ({rec.signal_date})"
        if rec.schema_version != PAPER_SCHEMA_VERSION:
            problems.append(f"{tag}: schema_version {rec.schema_version} != {PAPER_SCHEMA_VERSION}")
        if rec.record_hash in seen:
            problems.append(f"{tag}: duplicate record_hash {rec.record_hash[:12]}")
        seen.add(rec.record_hash)
        expected = _hash(rec.stored_payload())
        if expected != rec.record_hash:
            problems.append(f"{tag}: content hash mismatch (log has {rec.record_hash[:12]}, "
                            f"content hashes to {expected[:12]})")
        if rec.prev_hash != previous:
            problems.append(f"{tag}: prev_hash {rec.prev_hash[:12] or '(empty)'} does not "
                            f"link to {previous[:12] or '(empty)'}")
        previous = rec.record_hash
    return problems


def latest_per_signal_date(records: list[PaperRecord]) -> dict[str, PaperRecord]:
    """One record per signal date: the last one written wins.

    A superseded record stays in the log (the chain needs it) but must not be
    scored twice, or the same period would enter the series more than once.
    """
    out: dict[str, PaperRecord] = {}
    for rec in records:
        out[rec.signal_date] = rec
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _price_matrices(
    prices: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align every symbol's open and close onto one session calendar.

    Two dense frames (session x symbol) let a period's return be one vectorised
    division instead of a per-name loop, which matters at ~460 names a day.
    """
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for sym, px in prices.items():
        if px is None or px.empty:
            continue
        frame = px.reindex(calendar)
        opens[str(sym)] = frame["open"].astype(float)
        closes[str(sym)] = frame["close"].astype(float)
    return pd.DataFrame(opens, index=calendar), pd.DataFrame(closes, index=calendar)


def _calendar(prices: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None) -> pd.DatetimeIndex:
    """The session calendar: the benchmark's sessions, else every symbol's union."""
    if benchmark is not None and not benchmark.empty:
        return pd.DatetimeIndex(pd.to_datetime(benchmark.index).normalize()).sort_values()
    stamps: list[pd.Timestamp] = []
    for px in prices.values():
        if px is not None and not px.empty:
            stamps.append(pd.DatetimeIndex(pd.to_datetime(px.index).normalize()))
    if not stamps:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(stamps[0].append(stamps[1:])).unique().sort_values()


def score_records(
    records: list[PaperRecord],
    prices: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame | None = None,
    friction: float = 0.0,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Grade every frozen record whose holding period has closed.

    The trade reproduced here is the one the labels encode: enter at the **next
    session's open** after the signal, exit at the close ``horizon`` sessions
    after the signal, and pay the round trip once. Benchmarks are charged the
    same friction, because that is the whole discipline of the kill test.

    Only ``is_grid`` records are scored (see the module docstring). Everything
    else is returned with a non-``closed`` status so the log stays complete
    while the headline series stays non-overlapping.
    """
    calendar = _calendar(prices, benchmark)
    if len(calendar) == 0:
        raise ValueError("no price data: cannot score any record")

    position = {d: i for i, d in enumerate(calendar)}
    opens, closes = _price_matrices(prices, calendar)
    bench_open = bench_close = None
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(calendar)
        bench_open = b["open"].astype(float)
        bench_close = b["close"].astype(float)

    rows: list[dict] = []
    for rec in records:
        h = int(horizon or rec.horizon_days)
        row: dict = {
            "signal_date": rec.signal_date,
            "is_grid": bool(rec.is_grid),
            "n_selected": rec.n_selected,
            "n_candidates": int(rec.n_candidates),
        }
        i = position.get(pd.Timestamp(rec.signal_date))
        if i is None:
            row["status"] = STATUS_NO_SESSION
            rows.append(row)
            continue
        if i + 1 >= len(calendar):
            row["status"] = STATUS_NO_ENTRY
            rows.append(row)
            continue
        # The exit session is only known once it exists: looking it up before
        # the open check would raise on exactly the records that are still
        # running, which is the normal state of a live forward test.
        row["entry_date"] = str(calendar[i + 1].date())
        closed = i + h < len(calendar)
        row["exit_date"] = str(calendar[i + h].date()) if closed else None
        if not rec.is_forward:
            row["status"] = STATUS_BACKFILL
            rows.append(row)
            continue
        if not rec.cross_section_ok:
            row["status"] = STATUS_INCOMPLETE
            rows.append(row)
            continue
        if not rec.is_grid:
            row["status"] = STATUS_OFF_GRID
            rows.append(row)
            continue
        if not closed:
            row["status"] = STATUS_OPEN
            rows.append(row)
            continue

        rets = (closes.iloc[i + h] / opens.iloc[i + 1]) - 1.0 - friction
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        selected = rets.reindex([s for s in rec.selected if s in rets.index])
        row["status"] = STATUS_CLOSED
        row["n_priced_selected"] = int(len(selected))
        row["strategy"] = float(selected.mean()) if len(selected) else float("nan")
        row["universe"] = float(rets.mean()) if len(rets) else float("nan")
        if bench_open is not None:
            o, c = bench_open.iloc[i + 1], bench_close.iloc[i + h]
            row["market"] = float(c / o - 1.0 - friction) if np.isfinite(o) and o > 0 else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("signal_date").reset_index(drop=True)
    return out


def grid_problems(
    scored: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
) -> list[str]:
    """Confirm the scored periods really are non-overlapping.

    A grid record is a commitment, not a guarantee: if two signal dates entered
    the log fewer than ``horizon`` sessions apart, their holding periods share
    price path, and a Sharpe computed over both counts one move twice. Verified
    rather than assumed, the same way ``assert_no_overlap`` guards the backtest.
    """
    if scored is None or scored.empty:
        return []
    closed = scored[scored["status"] == STATUS_CLOSED]
    if closed.empty:
        return []

    position = {d: i for i, d in enumerate(calendar)}
    problems: list[str] = []
    positions: list[int] = []
    for raw in closed["signal_date"]:
        i = position.get(pd.Timestamp(raw))
        if i is None:
            problems.append(f"scored signal date {raw} is not on the price calendar")
        else:
            positions.append(i)
    for prev, cur in zip(positions, positions[1:]):
        gap = cur - prev
        if gap < horizon:
            problems.append(
                f"scored periods overlap: signal positions {prev} and {cur} are {gap} "
                f"sessions apart, but the hold is {horizon}"
            )
    return problems


def summarise(scored: pd.DataFrame, horizon: int) -> dict:
    """Portfolio statistics for the closed forward periods, plus both benchmarks.

    Reports an **undeflated** Sharpe with a t-statistic, and deliberately no DSR:
    the forward run searches nothing, so no new multiple-testing penalty applies
    (the configuration was charged once, by the backtest that chose it). With a
    handful of periods nothing here is conclusive, and the t-stat says so.
    """
    closed = scored[scored["status"] == STATUS_CLOSED] if not scored.empty else scored
    empty = {
        "n_periods": 0, "strategy": None, "universe": None, "market": None,
        "beta": {"n_periods": 0, "beta": float("nan"), "residual_sharpe_annualised": float("nan"),
                 "alpha_per_period": float("nan"), "r_squared": float("nan"),
                 "residual_series": pd.Series(dtype=float)},
        "t_stat": float("nan"), "periods_per_year": SESSIONS_PER_YEAR / horizon,
    }
    if closed is None or closed.empty:
        return empty

    ppy = SESSIONS_PER_YEAR / horizon
    perf = lambda s: performance_metrics(s, periods_per_year=ppy, period_sessions=horizon)
    strategy = perf(closed["strategy"])
    universe = perf(closed["universe"])
    market = perf(closed["market"]) if "market" in closed.columns else None
    beta = beta_analysis(closed["strategy"], closed["universe"], periods_per_year=ppy)

    r = closed["strategy"].dropna().to_numpy(dtype=float)
    t_stat = float("nan")
    if r.size >= 2:
        sd = float(r.std(ddof=1))
        if sd > 0:
            t_stat = float(r.mean() / (sd / np.sqrt(r.size)))

    return {
        "n_periods": int(len(closed)),
        "strategy": strategy,
        "universe": universe,
        "market": market,
        "beta": beta,
        "t_stat": t_stat,
        "periods_per_year": ppy,
        "first_signal": str(closed["signal_date"].iloc[0]),
        "last_signal": str(closed["signal_date"].iloc[-1]),
    }


def format_report(
    scored: pd.DataFrame,
    summary: dict,
    horizon: int,
    friction: float,
    problems: list[str] | None = None,
) -> str:
    """The forward scorecard a human reads. Forward in time, so also log it."""
    lines: list[str] = []
    if scored.empty:
        n_open = n_off = n_back = n_thin = 0
    else:
        n_open = int((scored["status"] == STATUS_OPEN).sum())
        n_off = int((scored["status"] == STATUS_OFF_GRID).sum())
        n_back = int((scored["status"] == STATUS_BACKFILL).sum())
        n_thin = int((scored["status"] == STATUS_INCOMPLETE).sum())
    lines.append(f"frozen records {len(scored):4d} | closed {summary['n_periods']:4d} | "
                 f"open {n_open:4d} | off-grid {n_off:4d} | thin cross-section {n_thin:4d} | "
                 f"backfill (not evidence) {n_back:4d}")
    if problems:
        lines.append("")
        for p in problems:
            lines.append(f"  !! {p}")

    if summary["n_periods"] == 0:
        lines.append("")
        lines.append("  no closed forward period yet -- nothing to conclude.")
        lines.append("  A 10-session hold needs ~10 sessions after the signal date, plus a "
                     "rebuild of")
        lines.append("  the dataset so the prices after the signal date exist. "
                     "Re-run once they do.")
        return "\n".join(lines)

    st, uni, mkt = summary["strategy"], summary["universe"], summary["market"]
    lines.append("")
    lines.append(f"FORWARD (undeflated), {summary['first_signal']} .. {summary['last_signal']} "
                 f"| {horizon}-session hold | friction {friction:.2%} per period")
    lines.append(f"  strategy   {st.n_periods:3d} periods | mean {st.mean_return:+.3%} | "
                 f"Sharpe {st.sharpe_annualised:6.2f} | Sortino {st.sortino_annualised:6.2f} | "
                 f"MDD {st.max_drawdown:6.2%} | PF {st.profit_factor:5.2f} | "
                 f"hit {st.hit_rate:5.1%}")
    lines.append(f"  universe   {uni.n_periods:3d} periods | mean {uni.mean_return:+.3%} | "
                 f"Sharpe {uni.sharpe_annualised:6.2f}   (same trade, every candidate, "
                 f"friction charged)")
    if mkt is not None:
        lines.append(f"  Nifty      {mkt.n_periods:3d} periods | mean {mkt.mean_return:+.3%} | "
                     f"Sharpe {mkt.sharpe_annualised:6.2f}   (same windows)")
    lines.append(f"  t-stat on the mean {summary['t_stat']:+.2f} "
                 f"({summary['n_periods']} independent periods)")
    beta = summary["beta"]
    if beta["n_periods"] >= 3 and np.isfinite(beta["beta"]):
        lines.append(f"  beta {beta['beta']:.2f} | R2 {beta['r_squared']:.2f} | "
                     f"alpha {beta['alpha_per_period']:+.3%}/period | beta-hedged residual "
                     f"Sharpe {beta['residual_sharpe_annualised']:.2f}")
    lines.append("")
    n = summary["n_periods"]
    if n < 30:
        lines.append(f"  NOT CONCLUSIVE: {n} independent periods. This strategy yields ~"
                     f"{252 // horizon}/year,")
        lines.append("  so a Sharpe claim needs years, not weeks. What this catches early is a")
        lines.append("  *break* -- if the edge is gone, the mean turns negative long before the")
        lines.append("  Sharpe is estimable.")
    return "\n".join(lines)


__all__ = [
    "PAPER_SCHEMA_VERSION",
    "DEFAULT_LOG_PATH",
    "PaperRecord",
    "append_record",
    "code_fingerprint",
    "config_fingerprint",
    "cross_section_ok",
    "entry_session_open",
    "format_report",
    "grid_problems",
    "is_forward_record",
    "latest_per_signal_date",
    "load_records",
    "next_weekday",
    "new_record",
    "score_records",
    "select_top",
    "selection_size",
    "summarise",
    "verify_chain",
]
