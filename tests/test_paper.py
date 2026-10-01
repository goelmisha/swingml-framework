"""Forward paper-trading tests.

The log's whole value is that it is *evidence*, so most of these tests attack it:
can a record be edited, deleted or back-dated without the chain noticing? The
rest pin the two places where the live path could silently diverge from the
backtest -- the selection (must match ``top_fraction_mask``) and the trade itself
(entry at the next session's open, exit at the close ``horizon`` sessions later,
friction charged once).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from swingml.config import load_config
from swingml.evaluation import top_fraction_mask
from swingml.paper import (
    PAPER_SCHEMA_VERSION,
    PaperRecord,
    STATUS_BACKFILL,
    STATUS_CLOSED,
    STATUS_INCOMPLETE,
    STATUS_NO_SESSION,
    STATUS_OFF_GRID,
    STATUS_OPEN,
    append_record,
    config_fingerprint,
    format_report,
    grid_problems,
    latest_per_signal_date,
    load_records,
    new_record,
    score_records,
    select_top,
    selection_size,
    summarise,
    verify_chain,
)
from swingml.paper import (
    cross_section_ok,
    entry_session_open,
    is_forward_record,
    next_weekday,
)
from scripts.paper_trade import dataset_label_mode, grid_flag, training_cutoff

HORIZON = 10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _record(signal_date: str, selected, scores, *, is_grid=True, is_forward=True,
            cross_section_ok=True, horizon=HORIZON, session_index=0) -> PaperRecord:
    return new_record(
        signal_date=signal_date,
        session_index=session_index,
        is_grid=is_grid,
        is_forward=is_forward,
        cross_section_ok=cross_section_ok,
        horizon_days=horizon,
        decile=10,
        engine="sklearn-hgb",
        label_mode="fixed_hold",
        dataset_dir="data/datasets_liquidity_fh",
        n_candidates=len(scores),
        selected=list(selected),
        scores=dict(scores),
        config_hash="cafe",
        config={"engine": "sklearn-hgb"},
        code_hash="deadbeef",
    )


def _panel(slopes: dict[str, float], dates: pd.DatetimeIndex, base: float = 100.0):
    """One OHLC frame per symbol: flat open, linearly rising close."""
    out = {}
    for sym, slope in slopes.items():
        closes = [base * (1.0 + slope * i) for i in range(len(dates))]
        out[sym] = pd.DataFrame({"open": [base] * len(dates), "close": closes}, index=dates)
    return out


def _dates(n: int = 16) -> pd.DatetimeIndex:
    idx = pd.bdate_range("2024-01-01", periods=n)
    idx.name = "date"
    return idx


# ---------------------------------------------------------------------------
# selection must be the backtest's selection
# ---------------------------------------------------------------------------

def test_selection_size_matches_top_fraction_mask():
    """46 of 460 at decile 10 is the rule the backtest actually applies."""
    for n, decile in [(460, 10), (100, 10), (37, 5), (9, 10), (1, 10)]:
        df = pd.DataFrame({
            "date": pd.Timestamp("2024-01-02"),
            "score": np.arange(n, dtype=float),
        })
        mask = top_fraction_mask(df["score"], df["date"], decile)
        assert int(mask.sum()) == selection_size(n, decile), (n, decile)


def test_select_top_agrees_with_top_fraction_mask():
    """If these ever diverge, the live run measures a different rule."""
    rng = np.random.default_rng(11)
    symbols = [f"S{i:03d}" for i in range(240)]
    scores = {s: float(v) for s, v in zip(symbols, rng.normal(size=len(symbols)))}

    df = pd.DataFrame({
        "date": pd.Timestamp("2024-03-01"),
        "symbol": symbols,
        "score": [scores[s] for s in symbols],
    })
    mask = top_fraction_mask(df["score"], df["date"], 10)
    from_mask = sorted(df.loc[mask, "symbol"])

    assert select_top(scores, decile=10) == from_mask


def test_select_top_is_deterministic_on_ties_and_drops_nan():
    scores = {"BBB": 1.0, "AAA": 1.0, "CCC": 0.5, "DDD": float("nan")}
    # decile 1 keeps everything finite; ties break by symbol, not dict order
    assert select_top(scores, decile=1) == ["AAA", "BBB", "CCC"]
    assert "DDD" not in select_top(scores, decile=1)
    # 3 candidates at decile 10 -> 3 - floor(2.7) = 1 name, and the tie is the
    # higher-relevance one either way: both are 1.0 here, so the symbol decides.
    assert select_top(scores, decile=10) == ["AAA"]


def test_selection_size_rejects_bad_decile():
    with pytest.raises(ValueError):
        selection_size(10, 0)


# ---------------------------------------------------------------------------
# the chain: the log has to be evidence
# ---------------------------------------------------------------------------

def test_append_chains_records_and_round_trips(tmp_path):
    log = tmp_path / "predictions.jsonl"
    first = append_record(log, _record("2024-01-02", ["AAA"], {"AAA": 0.9, "BBB": 0.1}))
    second = append_record(log, _record("2024-01-03", ["BBB"], {"AAA": 0.2, "BBB": 0.8}))

    assert first.prev_hash == ""            # genesis
    assert second.prev_hash == first.record_hash
    assert first.schema_version == PAPER_SCHEMA_VERSION

    loaded = load_records(log)
    assert [r.signal_date for r in loaded] == ["2024-01-02", "2024-01-03"]
    assert verify_chain(loaded) == []
    assert loaded[1].record_hash == second.record_hash


def test_chain_detects_an_edited_score(tmp_path):
    log = tmp_path / "predictions.jsonl"
    append_record(log, _record("2024-01-02", ["AAA"], {"AAA": 0.9, "BBB": 0.1}))
    append_record(log, _record("2024-01-03", ["BBB"], {"AAA": 0.2, "BBB": 0.8}))
    records = load_records(log)

    records[0].scores["BBB"] = 0.99          # quietly promote a name after the fact
    problems = verify_chain(records)
    assert any("content hash mismatch" in p for p in problems)


def test_chain_detects_a_deleted_record(tmp_path):
    log = tmp_path / "predictions.jsonl"
    append_record(log, _record("2024-01-02", ["AAA"], {"AAA": 0.9}))
    append_record(log, _record("2024-01-03", ["BBB"], {"BBB": 0.8}))
    records = load_records(log)

    problems = verify_chain(records[1:])     # drop the losing first day
    assert any("does not link" in p for p in problems)


def test_chain_detects_an_inserted_record(tmp_path):
    log = tmp_path / "predictions.jsonl"
    append_record(log, _record("2024-01-02", ["AAA"], {"AAA": 0.9}))
    records = load_records(log)

    forged = _record("2023-12-01", ["ZZZ"], {"ZZZ": 1.0})
    forged.prev_hash = ""
    from swingml.paper import _hash
    forged.record_hash = _hash(forged.body())

    problems = verify_chain([forged] + records)
    assert any("does not link" in p for p in problems)


def test_refreezing_a_date_requires_supersedes(tmp_path):
    log = tmp_path / "predictions.jsonl"
    first = append_record(log, _record("2024-01-02", ["AAA"], {"AAA": 0.9}))

    with pytest.raises(ValueError, match="already frozen"):
        append_record(log, _record("2024-01-02", ["BBB"], {"BBB": 0.9}))

    bad = _record("2024-01-02", ["BBB"], {"BBB": 0.9})
    bad.supersedes = "not-a-real-hash"
    with pytest.raises(ValueError, match="supersedes must reference"):
        append_record(log, bad)

    corrected = _record("2024-01-02", ["BBB"], {"BBB": 0.9})
    corrected.supersedes = first.record_hash
    second = append_record(log, corrected)

    assert second.prev_hash == first.record_hash
    assert verify_chain(load_records(log)) == []
    # the superseded record stays in the log but the latest one is scored
    latest = latest_per_signal_date(load_records(log))
    assert latest["2024-01-02"].selected == ["BBB"]


def test_verify_chain_flags_an_unknown_schema_version(tmp_path):
    log = tmp_path / "predictions.jsonl"
    append_record(log, _record("2024-01-02", ["AAA"], {"AAA": 0.9}))
    records = load_records(log)
    records[0].schema_version = 99
    assert any("schema_version" in p for p in verify_chain(records))


def test_adding_a_field_does_not_invalidate_earlier_records(tmp_path):
    """The log must stay verifiable across schema growth.

    Verification hashes the stored object, not a reconstruction, so a record
    written before a field existed still checks out -- and, because the new
    flags default to ineligible, it is excluded from the series rather than
    admitted by omission.
    """
    from swingml.paper import _hash

    log = tmp_path / "predictions.jsonl"
    legacy = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "timestamp": "2026-09-28T20:00:00+00:00",
        "signal_date": "2026-09-28",
        "session_index": 1466,
        "is_grid": False,
        "is_forward": True,
        "horizon_days": HORIZON,
        "decile": 10,
        "engine": "sklearn-hgb",
        "label_mode": "fixed_hold",
        "dataset_dir": "data/datasets_liquidity_fh",
        "n_candidates": 139,
        "selected": ["AAA"],
        "scores": {"AAA": 0.9},
        "config_hash": "cafe",
        "config": {},
        "code_hash": "deadbeef",
        "prev_hash": "",
    }
    legacy["record_hash"] = _hash(legacy)
    log.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    records = load_records(log)
    assert verify_chain(records) == [], "a stored record must verify as stored"
    assert records[0].cross_section_ok is False, "an unflagged record is not eligible"


def test_load_records_rejects_unknown_fields(tmp_path):
    log = tmp_path / "predictions.jsonl"
    log.write_text('{"signal_date": "2024-01-02", "surprise": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        load_records(log)


# ---------------------------------------------------------------------------
# training cut and the pre-registered grid
# ---------------------------------------------------------------------------

def test_training_cutoff_sits_one_purge_behind_the_signal():
    sessions = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=40))
    cutoff = training_cutoff(sessions, sessions[30], purge_sessions=10)
    assert cutoff == sessions[20]


def test_training_cutoff_refuses_a_window_too_short_to_place_the_purge():
    sessions = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5))
    with pytest.raises(ValueError, match="need at least"):
        training_cutoff(sessions, sessions[3], purge_sessions=10)


# ---------------------------------------------------------------------------
# forward vs backfill: the entry price is what decides
# ---------------------------------------------------------------------------

def test_next_weekday_skips_the_weekend():
    import datetime as dt
    assert next_weekday(dt.date(2024, 1, 5)) == dt.date(2024, 1, 8)   # Fri -> Mon
    assert next_weekday(dt.date(2024, 1, 8)) == dt.date(2024, 1, 9)   # Mon -> Tue


def test_entry_opens_at_0915_ist_the_next_weekday():
    import datetime as dt
    opened = entry_session_open("2024-01-05")                          # a Friday
    assert (opened.year, opened.month, opened.day) == (2024, 1, 8)
    assert opened.time() == dt.time(9, 15)
    assert opened.utcoffset() == dt.timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("signal,now_ist,expected", [
    # the evening of the signal session: the normal cron slot
    ("2024-01-08", "2024-01-08 20:00", True),
    # the next morning before the open: also genuinely forward
    ("2024-01-08", "2024-01-09 07:00", True),
    ("2024-01-08", "2024-01-09 09:14", True),
    # the entry session has opened
    ("2024-01-08", "2024-01-09 09:15", False),
    ("2024-01-08", "2024-01-09 12:00", False),
    # days later
    ("2024-01-08", "2024-01-11 10:00", False),
    # Friday signal: the entry is Monday, so a weekend run is still forward
    ("2024-01-05", "2024-01-06 11:00", True),
    ("2024-01-05", "2024-01-08 08:00", True),
    ("2024-01-05", "2024-01-08 09:20", False),
])
def test_is_forward_record_turns_on_the_entry_price(signal, now_ist, expected):
    import datetime as dt
    now = dt.datetime.strptime(now_ist, "%Y-%m-%d %H:%M")
    assert is_forward_record(signal, now=now) is expected


# ---------------------------------------------------------------------------
# thin cross-section: the newest session is still filling in
# ---------------------------------------------------------------------------

def test_cross_section_guard_catches_a_truncated_newest_session():
    """A truncated newest session must be refused."""
    recent = [172, 175, 174, 172, 173, 172, 174, 176, 172, 175]
    ok, median = cross_section_ok(139, recent)
    assert median == 173
    assert not ok
    # a normal session passes
    assert cross_section_ok(170, recent)[0]
    # the tolerance is a floor (0.9 x 173 = 155.7), not an equality test -- 155 is
    # still refused, 158 is not
    assert not cross_section_ok(155, recent)[0]
    assert cross_section_ok(158, recent)[0]


def test_cross_section_guard_defers_when_there_is_no_history():
    assert cross_section_ok(3, [])[0]
    assert cross_section_ok(3, [10, 10, 10])[0]      # fewer than 5 sessions: cannot judge


def test_a_thin_cross_section_is_excluded_from_the_forward_series():
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0})
    rec.cross_section_ok = False

    scored = score_records([rec], prices, friction=0.0, horizon=HORIZON)
    assert scored.iloc[0]["status"] == STATUS_INCOMPLETE
    assert summarise(scored, HORIZON)["n_periods"] == 0


def test_a_stale_dataset_is_a_backfill():
    """The real case this guard exists for: freezing last week's session today."""
    import datetime as dt
    assert not is_forward_record("2026-09-24", now=dt.datetime(2026, 9, 29, 7, 0))


def test_grid_flag_marks_every_horizonth_session():
    flags = [grid_flag(i, 10) for i in range(25)]
    assert flags[0] and flags[10] and flags[20]
    assert not any(flags[i] for i in range(1, 10))


# ---------------------------------------------------------------------------
# scoring: the trade has to be the labelled trade
# ---------------------------------------------------------------------------

def test_score_reproduces_next_open_to_exit_close_minus_friction():
    dates = _dates()
    slopes = {"AAA": 0.01, "BBB": 0.02, "CCC": 0.005}
    prices = _panel(slopes, dates)
    bench = _panel({"NSEI": 0.004}, dates)["NSEI"]
    friction = 0.0025

    rec = _record(str(dates[0].date()), ["AAA", "BBB"], {s: 0.5 for s in slopes})
    scored = score_records([rec], prices, benchmark=bench, friction=friction, horizon=HORIZON)

    row = scored.iloc[0]
    assert row["status"] == STATUS_CLOSED
    assert row["entry_date"] == str(dates[1].date())
    assert row["exit_date"] == str(dates[HORIZON].date())
    assert row["strategy"] == pytest.approx(0.15 - friction)      # mean(10%, 20%)
    assert row["universe"] == pytest.approx((0.10 + 0.20 + 0.05) / 3 - friction)
    assert row["market"] == pytest.approx(0.04 - friction)
    assert row["n_priced_selected"] == 2


def test_score_charges_friction_to_every_series():
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0})

    free = score_records([rec], prices, friction=0.0, horizon=HORIZON).iloc[0]
    paid = score_records([rec], prices, friction=0.02, horizon=HORIZON).iloc[0]
    assert paid["strategy"] == pytest.approx(free["strategy"] - 0.02)


def test_score_reports_an_open_period_instead_of_guessing():
    dates = _dates(6)                                  # fewer sessions than the hold
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0})

    row = score_records([rec], prices, friction=0.0, horizon=HORIZON).iloc[0]
    assert row["status"] == STATUS_OPEN
    assert row["exit_date"] is None
    assert row["entry_date"] == str(dates[1].date())
    assert "strategy" not in row or pd.isna(row.get("strategy"))


def test_a_backfilled_record_is_excluded_from_the_forward_series():
    """A prediction written after its entry session traded is not evidence.

    It may have been fitted to a price that already exists, so it is kept in the
    log (deleting it would break the chain) and kept out of the headline.
    """
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0}, is_forward=False)

    scored = score_records([rec], prices, friction=0.0, horizon=HORIZON)
    assert scored.iloc[0]["status"] == STATUS_BACKFILL
    assert summarise(scored, HORIZON)["n_periods"] == 0
    text = format_report(scored, summarise(scored, HORIZON), HORIZON, 0.0)
    assert "backfill (not evidence)    1" in text


@pytest.mark.parametrize("is_forward,is_grid,status", [
    (False, True, STATUS_BACKFILL),      # backfill wins: it is not evidence either way
    (True, False, STATUS_OFF_GRID),
    (True, True, STATUS_CLOSED),
])
def test_scoring_gates_on_forward_before_grid(is_forward, is_grid, status):
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0},
                  is_forward=is_forward, is_grid=is_grid)
    assert score_records([rec], prices, friction=0.0, horizon=HORIZON).iloc[0]["status"] == status


def test_off_grid_records_are_recorded_but_never_scored():
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0}, is_grid=False)

    scored = score_records([rec], prices, friction=0.0, horizon=HORIZON)
    assert scored.iloc[0]["status"] == STATUS_OFF_GRID
    assert summarise(scored, HORIZON)["n_periods"] == 0


def test_signal_date_outside_the_price_window_is_flagged_not_scored():
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record("2019-06-03", ["AAA"], {"AAA": 1.0})

    row = score_records([rec], prices, friction=0.0, horizon=HORIZON).iloc[0]
    assert row["status"] == STATUS_NO_SESSION


def test_missing_prices_shrink_the_selection_but_do_not_crash():
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)             # BBB was delisted / unpriced
    rec = _record(str(dates[0].date()), ["AAA", "ZZZ"], {"AAA": 1.0, "ZZZ": 0.9})

    row = score_records([rec], prices, friction=0.0, horizon=HORIZON).iloc[0]
    assert row["status"] == STATUS_CLOSED
    assert row["n_priced_selected"] == 1


def test_summarise_reports_sharpe_and_both_benchmarks():
    dates = _dates(60)
    rng = np.random.default_rng(3)
    slopes = {f"S{i}": float(v) for i, v in enumerate(rng.normal(0.012, 0.01, 30))}
    prices = _panel(slopes, dates)
    bench = _panel({"NSEI": 0.006}, dates)["NSEI"]

    records = []
    for start in range(0, 40, HORIZON):               # a pre-registered non-overlapping grid
        signal = dates[start]
        ordered = sorted(slopes, key=lambda s: -slopes[s])[:3]
        records.append(_record(str(signal.date()), ordered, dict(slopes), session_index=start))

    scored = score_records(records, prices, benchmark=bench, friction=0.0025, horizon=HORIZON)
    summary = summarise(scored, HORIZON)

    assert summary["n_periods"] == len(records)
    assert summary["strategy"].mean_return == pytest.approx(
        np.mean([r["strategy"] for _, r in scored.iterrows() if r["status"] == STATUS_CLOSED]))
    assert np.isfinite(summary["t_stat"])
    assert summary["periods_per_year"] == pytest.approx(252 / HORIZON)
    assert grid_problems(scored, dates, HORIZON) == []


def test_summarise_of_nothing_is_empty_not_a_flattering_zero():
    summary = summarise(pd.DataFrame(), HORIZON)
    assert summary["n_periods"] == 0
    assert summary["strategy"] is None
    assert "no closed forward period" in format_report(pd.DataFrame(), summary, HORIZON, 0.0025)


def test_grid_problems_flags_overlapping_scored_periods():
    dates = _dates(40)
    prices = _panel({"AAA": 0.01}, dates)
    # two "grid" records only 5 sessions apart: the hold would be counted twice
    records = [
        _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0}, session_index=0),
        _record(str(dates[5].date()), ["AAA"], {"AAA": 1.0}, session_index=5),
    ]
    scored = score_records(records, prices, friction=0.0, horizon=HORIZON)
    problems = grid_problems(scored, dates, HORIZON)
    assert problems and "overlap" in problems[0]


def test_format_report_says_a_handful_of_periods_is_not_conclusive():
    dates = _dates()
    prices = _panel({"AAA": 0.01}, dates)
    rec = _record(str(dates[0].date()), ["AAA"], {"AAA": 1.0})
    scored = score_records([rec], prices, friction=0.0, horizon=HORIZON)
    text = format_report(scored, summarise(scored, HORIZON), HORIZON, 0.0)
    assert "NOT CONCLUSIVE" in text
    assert "t-stat" in text


# ---------------------------------------------------------------------------
# provenance and the dataset's own target mode
# ---------------------------------------------------------------------------

def test_config_fingerprint_prefers_the_datasets_own_facts():
    """A record must name the dataset it was fitted to, not the YAML defaults.

    The config default universe is ``nifty200`` while every validated result is
    over ``liquidity``; stamping the genesis record with the default would make
    its provenance describe a panel it was never fitted to.
    """
    cfg = load_config()
    _, payload = config_fingerprint(
        cfg, "sklearn-hgb", 10,
        dataset_facts={"label_mode": "fixed_hold", "universe_source": "liquidity"},
    )
    assert payload["label_mode"] == "fixed_hold"
    assert payload["universe_source"] == "liquidity"
    # No facts -> fall back to the config, so the argument stays optional.
    _, bare = config_fingerprint(cfg, "sklearn-hgb", 10)
    assert bare["label_mode"] == cfg.label.mode
    assert bare["universe_source"] == cfg.universe.source
    # A missing fact falls back individually rather than nulling the payload.
    _, partial = config_fingerprint(cfg, "sklearn-hgb", 10, dataset_facts={"label_mode": "fixed_hold"})
    assert partial["label_mode"] == "fixed_hold"
    assert partial["universe_source"] == cfg.universe.source


def test_config_fingerprint_moves_when_a_scoring_setting_moves():
    cfg = load_config()
    before, payload = config_fingerprint(cfg, "sklearn-hgb", 10)
    assert payload["engine"] == "sklearn-hgb" and payload["horizon_days"] == cfg.label.horizon_days

    assert config_fingerprint(cfg, "sklearn-hgb", 10)[0] == before
    assert config_fingerprint(cfg, "lightgbm", 10)[0] != before
    assert config_fingerprint(cfg, "sklearn-hgb", 5)[0] != before

    cfg.label.horizon_days = cfg.label.horizon_days + 1
    assert config_fingerprint(cfg, "sklearn-hgb", 10)[0] != before


def test_dataset_label_mode_reads_disk_and_tolerates_junk(tmp_path):
    assert dataset_label_mode(tmp_path) is None

    (tmp_path / "label_diagnostics.json").write_text('{"mode": "fixed_hold"}', encoding="utf-8")
    assert dataset_label_mode(tmp_path) == "fixed_hold"

    (tmp_path / "label_diagnostics.json").write_text("not json", encoding="utf-8")
    assert dataset_label_mode(tmp_path) is None
