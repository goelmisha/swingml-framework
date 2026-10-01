"""The point-in-time symbol history and the rename recovery it enables.

Two things are being defended here, and both matter more than coverage:

1. **A wrong alias fabricates price history.** An alias is only admitted when the
   retired symbol's last observation strictly precedes the replacement's first --
   an overlap means they were both listed, so they are different securities.
2. **A recovered name must not double-count.** The successor is itself in the
   universe, so the recovered series is sliced to the retired symbol's own
   listing span.

Everything runs offline: the parsers are exercised on synthetic payloads of both
real bhavcopy formats, and no test touches the archive.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from swingml.dataset import apply_symbol_aliases
from swingml.data.symbol_history import (
    SymbolHistory,
    build_symbol_history,
    parse_isin_frame,
)

OLD_HEADER = "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,"
NEW_HEADER = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
              "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
              "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
              "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,RSV1,RSV2,RSV3,RSV4")


def _old_bhav(*rows: tuple[str, str, str]) -> str:
    lines = [OLD_HEADER] + [f"{s},{ser},1,2,0.5,1.5,1.5,1.4,10,100,01-JAN-2020,5,{isin}," for s, ser, isin in rows]
    return "\n".join(lines) + "\n"


def _new_bhav(*rows: tuple[str, str, str]) -> str:
    lines = [NEW_HEADER] + [
        f"2026-09-30,2026-09-30,CM,NSE,STK,1,{isin},{s},{ser},,,,,,1,2,0.5,1.5,1.5,1.4,,1.5,,,10,100,5,,1,,,,,"
        for s, ser, isin in rows
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# parsing both archive formats
# ---------------------------------------------------------------------------

def test_parses_the_old_format():
    df = parse_isin_frame(_old_bhav(("CADILAHC", "EQ", "INE010B01027"), ("TCS", "EQ", "INE467B01029")))
    assert dict(zip(df["symbol"], df["isin"])) == {
        "CADILAHC": "INE010B01027", "TCS": "INE467B01029",
    }


def test_parses_the_udiff_format():
    df = parse_isin_frame(_new_bhav(("ZYDUSLIFE", "EQ", "INE010B01027")))
    assert list(df["symbol"]) == ["ZYDUSLIFE"]
    assert list(df["isin"]) == ["INE010B01027"]


def test_non_cash_series_are_excluded_and_symbols_are_normalised():
    df = parse_isin_frame(_old_bhav(
        ("reliance", "EQ", "ine002a01018"),
        ("SOMEBOND", "N1", "IN0020010081"),
        ("SOMEGS", "GS", "IN0020010081"),
    ))
    assert list(df["symbol"]) == ["RELIANCE"]
    assert list(df["isin"]) == ["INE002A01018"]


def test_unknown_schema_raises_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="schema"):
        parse_isin_frame("A,B,C\n1,2,3\n")


def test_empty_payload_gives_an_empty_frame():
    assert parse_isin_frame(_old_bhav()).empty


# ---------------------------------------------------------------------------
# alias derivation
# ---------------------------------------------------------------------------

def _history_from_samples(monkeypatch, samples: dict[dt.date, list[tuple[str, str]]]) -> SymbolHistory:
    calls = []

    def fake_fetch(day, timeout_sec=45):
        calls.append(day)
        rows = samples.get(day)
        if not rows:
            return pd.DataFrame(columns=["symbol", "isin"])
        return pd.DataFrame(rows, columns=["symbol", "isin"])

    monkeypatch.setattr("swingml.data.symbol_history.fetch_isin_frame", fake_fetch)
    days = sorted(samples)
    return build_symbol_history(days, sample_every=1)


def test_rename_is_detected_from_a_shared_isin(monkeypatch):
    """CADILAHC trades, then stops; ZYDUSLIFE picks the ISIN up in the next sample."""
    same = "INE010B01027"
    h = _history_from_samples(monkeypatch, {
        dt.date(2024, 1, 1): [("CADILAHC", same)],
        dt.date(2024, 2, 1): [("ZYDUSLIFE", same)],
        dt.date(2024, 3, 1): [("ZYDUSLIFE", same)],
    })
    assert h.isin_of["CADILAHC"] == same
    assert h.resolve("CADILAHC") == "ZYDUSLIFE"
    assert h.is_alias("CADILAHC")
    assert h.resolve("ZYDUSLIFE") == "ZYDUSLIFE"  # the current name is not an alias
    assert not h.is_alias("ZYDUSLIFE")
    assert len(h.aliases) == 1


def test_overlapping_symbols_on_one_isin_are_not_a_rename(monkeypatch):
    """Both listed at once -- different securities, and a fabricated series risk."""
    same = "INE155A01022"
    h = _history_from_samples(monkeypatch, {
        dt.date(2024, 1, 1): [("TATAMOTORS", same), ("TATAMTRDVR", same)],
        dt.date(2024, 2, 1): [("TATAMOTORS", same), ("TATAMTRDVR", same)],
    })
    assert h.aliases == {}


def test_chains_resolve_to_the_latest_name(monkeypatch):
    """LTI -> LTIM -> LTIMINDTREE, all one ISIN, resolves to the newest."""
    same = "INE214T01019"
    h = _history_from_samples(monkeypatch, {
        dt.date(2022, 1, 1): [("LTI", same)],
        dt.date(2023, 1, 1): [("LTIM", same)],
        dt.date(2024, 1, 1): [("LTIMINDTREE", same)],
    })
    assert h.resolve("LTI") == "LTIMINDTREE"
    assert h.resolve("LTIM") == "LTIMINDTREE"
    assert h.resolve("LTIMINDTREE") == "LTIMINDTREE"


def test_distinct_isins_are_never_aliased(monkeypatch):
    h = _history_from_samples(monkeypatch, {
        dt.date(2024, 1, 1): [("AAA", "INE000A01001")],
        dt.date(2024, 2, 1): [("BBB", "INE000B01002")],
    })
    assert h.aliases == {}


def test_json_roundtrip_preserves_the_map(monkeypatch):
    same = "INE010B01027"
    h = _history_from_samples(monkeypatch, {
        dt.date(2024, 1, 1): [("CADILAHC", same)],
        dt.date(2024, 2, 1): [("ZYDUSLIFE", same)],
    })
    back = SymbolHistory.from_json(h.to_json())
    assert back.aliases == h.aliases
    assert back.isin_of == h.isin_of
    assert back.resolve("CADILAHC") == "ZYDUSLIFE"


def test_resolve_is_identity_for_unknown_symbols():
    assert SymbolHistory().resolve("WHATEVER") == "WHATEVER"


# ---------------------------------------------------------------------------
# price recovery
# ---------------------------------------------------------------------------

def _series(start: str, periods: int) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}, index=idx
    )


def _panel(*pairs) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "date": pd.Timestamp(d)} for s, d in pairs])


def test_retired_symbol_gets_its_successors_history_over_its_own_span():
    successor = _series("2022-01-03", 500)
    fetched = {"ZYDUSLIFE": successor}
    history = SymbolHistory(aliases={"CADILAHC": "ZYDUSLIFE"})
    panel = _panel(("CADILAHC", "2022-01-03"), ("CADILAHC", "2022-06-30"))

    out, recovered = apply_symbol_aliases(fetched, history, ["CADILAHC", "ZYDUSLIFE"], panel)
    assert recovered == {"CADILAHC": "ZYDUSLIFE"}
    assert "CADILAHC" in out
    assert out["CADILAHC"].index.min() == pd.Timestamp("2022-01-03")
    assert out["CADILAHC"].index.max() == pd.Timestamp("2022-06-30")
    # The successor keeps its full series -- no truncation of the live name.
    assert len(out["ZYDUSLIFE"]) == len(successor)


def test_the_two_spans_of_one_security_do_not_overlap():
    """The invariant that keeps a rename from double-counting the same bars."""
    successor = _series("2022-01-03", 400)
    fetched = {"NEW": successor}
    history = SymbolHistory(aliases={"OLD": "NEW"})
    panel = _panel(("OLD", "2022-01-03"), ("OLD", "2022-03-31"), ("NEW", "2022-04-01"), ("NEW", "2022-12-30"))

    out, _ = apply_symbol_aliases(fetched, history, ["OLD", "NEW"], panel)
    old_end, new_start = out["OLD"].index.max(), out["NEW"].index.min()
    assert old_end < new_start


def test_recovery_never_reaches_past_the_successors_first_session():
    """A demerger is not a rename: TATAMOTORS kept trading after TMPV took the ISIN.

    Without the clamp, TMPV's post-split (passenger-vehicle) series would be
    spliced in as TATAMOTORS (commercial-vehicle) prices.
    """
    successor = _series("2022-01-03", 800)
    fetched = {"TMPV": successor}
    history = SymbolHistory(
        aliases={"TATAMOTORS": "TMPV"},
        first_seen={"TMPV": "2022-06-01"},
    )
    # The retired symbol's own span runs well past the successor's first session.
    panel = _panel(("TATAMOTORS", "2022-01-03"), ("TATAMOTORS", "2023-06-30"),
                   ("TMPV", "2022-06-01"), ("TMPV", "2023-12-29"))

    out, recovered = apply_symbol_aliases(fetched, history, ["TATAMOTORS", "TMPV"], panel)
    assert recovered == {"TATAMOTORS": "TMPV"}
    assert out["TATAMOTORS"].index.max() == pd.Timestamp("2022-05-31")
    assert out["TATAMOTORS"].index.max() < out["TMPV"].index.min()


def test_symbol_already_priced_is_left_alone():
    fetched = {"AAA": _series("2022-01-03", 10)}
    history = SymbolHistory(aliases={"AAA": "BBB"})
    out, recovered = apply_symbol_aliases(fetched, history, ["AAA"], _panel(("AAA", "2022-01-03")))
    assert recovered == {}
    assert len(out["AAA"]) == 10


def test_alias_with_no_successor_data_is_skipped_not_invented():
    history = SymbolHistory(aliases={"OLD": "NEW"})
    out, recovered = apply_symbol_aliases({}, history, ["OLD"], _panel(("OLD", "2022-01-03")))
    assert recovered == {} and "OLD" not in out


def test_alias_outside_the_successors_history_is_skipped():
    """Nothing is fabricated when the successor's series does not cover the span."""
    fetched = {"NEW": _series("2024-01-01", 10)}
    history = SymbolHistory(aliases={"OLD": "NEW"})
    panel = _panel(("OLD", "2022-01-03"), ("OLD", "2022-06-30"))
    out, recovered = apply_symbol_aliases(fetched, history, ["OLD"], panel)
    assert recovered == {} and "OLD" not in out


def test_symbols_without_an_alias_are_untouched():
    out, recovered = apply_symbol_aliases({}, SymbolHistory(), ["GONE"], _panel(("GONE", "2020-01-01")))
    assert out == {} and recovered == {}


def test_empty_panel_still_recovers_using_the_full_successor_series():
    fetched = {"NEW": _series("2022-01-03", 50)}
    history = SymbolHistory(aliases={"OLD": "NEW"})
    out, recovered = apply_symbol_aliases(fetched, history, ["OLD"], pd.DataFrame())
    assert recovered == {"OLD": "NEW"}
    assert len(out["OLD"]) == 50
