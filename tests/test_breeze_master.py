"""The ICICI security master: the NSE symbol -> Breeze ``stock_code`` map.

The mapping is the difference between "Breeze has no data for RELIANCE" and
"Breeze calls RELIANCE ``RELIND``". Everything here runs offline -- the master is
a public zip, but the parser and the cache policy are exercised on synthetic
bytes so the suite needs no network and no credentials.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from swingml.data.breeze_master import (
    BreezeSymbolMap,
    build_symbol_map,
    load_symbol_map,
    parse_security_master,
)

#: The real header is quoted and space-padded; both must be tolerated.
HEADER = 'Token, "ShortName", "Series", "CompanyName", "ExchangeCode"'


def _master(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


def test_parses_nse_symbol_to_breeze_code():
    text = _master(
        '"2885","RELIND","EQ","RELIANCE INDUSTRIES","RELIANCE"',
        '"1333","HDFBAN","EQ","HDFC BANK LIMITED","HDFCBANK"',
        '"11536","TCS","EQ","TATA CONSULTANCY SERVICES LTD","TCS"',
    )
    m = parse_security_master(text)
    assert m["RELIANCE"] == "RELIND"
    assert m["HDFCBANK"] == "HDFBAN"
    assert m["TCS"] == "TCS"  # a code that happens to be identical still round-trips


def test_non_cash_series_are_excluded():
    """A bond's ``ExchangeCode`` is its underlying, not a tradable symbol."""
    text = _master(
        '"2885","RELIND","EQ","RELIANCE INDUSTRIES","RELIANCE"',
        '"0","INHN56","BQ","SOME BOND","IBULHSGFIN"',
        '"13188","RELCOM","BE","RELIANCE COMMUNICATIONS","RELCOM"',
        '"1","MFUND","MF","SOME FUND","FUNDX"',
    )
    m = parse_security_master(text)
    assert set(m) == {"RELIANCE", "RELCOM"}
    assert m["RELCOM"] == "RELCOM"


def test_first_cash_row_wins_and_short_rows_are_skipped():
    text = _master(
        '"1","FIRST","EQ","A LTD","DUP"',
        '"2","SECOND","EQ","A LTD","DUP"',
        '"3","SHORT"',
    )
    m = parse_security_master(text)
    assert m["DUP"] == "FIRST"


def test_unexpected_schema_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="schema"):
        parse_security_master("A,B,C\n1,2,3\n")


def test_empty_text_is_an_empty_map():
    assert parse_security_master("") == {}


def test_resolve_falls_back_to_the_symbol_itself():
    m = BreezeSymbolMap(codes={"RELIANCE": "RELIND"})
    assert m.resolve("RELIANCE") == "RELIND"
    assert m.resolve("reliance") == "RELIND"  # case-insensitive
    assert m.resolve("BRANDNEW") == "BRANDNEW"
    assert "RELIANCE" in m and "BRANDNEW" not in m


def test_json_roundtrip_preserves_codes_and_provenance():
    m = BreezeSymbolMap(codes={"RELIANCE": "RELIND"}, fetched_at="2026-09-30")
    back = BreezeSymbolMap.from_json(m.to_json())
    assert back.codes == m.codes
    assert back.fetched_at == "2026-09-30"
    assert back.age_days == (dt.date.today() - dt.date(2026, 9, 30)).days


def test_fresh_cache_is_used_without_downloading(tmp_path, monkeypatch):
    p = tmp_path / "master.json"
    BreezeSymbolMap(codes={"RELIANCE": "RELIND"}, fetched_at=dt.date.today().isoformat()).save(p)

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("download attempted despite a fresh cache")

    monkeypatch.setattr("swingml.data.breeze_master.build_symbol_map", _boom)
    m = load_symbol_map(p)
    assert m.resolve("RELIANCE") == "RELIND"


def test_stale_cache_is_refreshed_and_rewritten(tmp_path, monkeypatch):
    p = tmp_path / "master.json"
    old = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    BreezeSymbolMap(codes={"OLD": "OLD"}, fetched_at=old).save(p)
    monkeypatch.setattr(
        "swingml.data.breeze_master.build_symbol_map",
        lambda **k: BreezeSymbolMap(codes={"RELIANCE": "RELIND"}),
    )
    m = load_symbol_map(p)
    assert m.resolve("RELIANCE") == "RELIND"
    assert json.loads(p.read_text())["codes"] == {"RELIANCE": "RELIND"}


def test_refresh_failure_degrades_to_the_cached_snapshot(tmp_path, monkeypatch):
    """A dead network must not turn a working setup into an empty symbol map."""
    p = tmp_path / "master.json"
    old = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    BreezeSymbolMap(codes={"RELIANCE": "RELIND"}, fetched_at=old).save(p)

    def _boom(**k):
        raise ConnectionError("network down")

    monkeypatch.setattr("swingml.data.breeze_master.build_symbol_map", _boom)
    assert load_symbol_map(p).resolve("RELIANCE") == "RELIND"


def test_refresh_failure_with_no_cache_raises(tmp_path, monkeypatch):
    """Silence here would make every symbol look delisted -- fail instead."""
    monkeypatch.setattr(
        "swingml.data.breeze_master.build_symbol_map",
        lambda **k: (_ for _ in ()).throw(ConnectionError("network down")),
    )
    with pytest.raises(RuntimeError, match="security master"):
        load_symbol_map(tmp_path / "missing.json")


def test_build_symbol_map_stamps_todays_date(monkeypatch):
    monkeypatch.setattr(
        "swingml.data.breeze_master.download_security_master",
        lambda **k: _master('"2885","RELIND","EQ","RELIANCE INDUSTRIES","RELIANCE"'),
    )
    m = build_symbol_map()
    assert m.fetched_at == dt.date.today().isoformat()
    assert len(m) == 1
