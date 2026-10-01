"""Breeze Connect: token handling and the price provider.

The token rules are the security- and correctness-critical part: a token from a
previous day must be *refused*, not sent to the API where the failure would look
like a market-data outage. The provider is exercised against a fake SDK client,
so nothing here touches the network or needs credentials.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd
import pytest

from swingml.config import DataConfig
from swingml.data.breeze import (
    BreezeSessionError,
    ist_today,
    load_env_file,
    load_session_token,
    login_url,
    parse_session_token,
    save_session_token,
    session_status,
)
from swingml.data.cache import DiskCache
from swingml.data.prices import BreezeApiError, BreezePriceProvider, make_price_provider

TOKEN = "AbCdEf0123456789xyzXYZ"  # 24 chars, passes the shape check


# ---------------------------------------------------------------------------
# token parsing
# ---------------------------------------------------------------------------

def test_parse_full_redirect_url():
    url = f"https://127.0.0.1/?apisession={TOKEN}&other=1"
    assert parse_session_token(url) == TOKEN


def test_parse_query_fragment_and_bare_token():
    assert parse_session_token(f"apisession={TOKEN}") == TOKEN
    assert parse_session_token(f"  {TOKEN}  ") == TOKEN
    assert parse_session_token(f'"{TOKEN}"') == TOKEN


def test_parse_url_encoded_token_is_unquoted():
    # %41 -> 'A'; the decoded value still has to satisfy the token shape.
    assert parse_session_token("https://x.y/?apisession=AbCdEf0123456789xyz%41") == "AbCdEf0123456789xyzA"


def test_login_page_without_apisession_is_rejected():
    with pytest.raises(BreezeSessionError, match="apisession"):
        parse_session_token("https://api.icicidirect.com/apiuser/login?api_key=abc")


def test_empty_and_shapeless_inputs_are_rejected():
    with pytest.raises(BreezeSessionError):
        parse_session_token("")
    with pytest.raises(BreezeSessionError, match="look like"):
        parse_session_token("short")


def test_short_numeric_apisession_is_accepted(tmp_path):
    """Breeze has been observed to mint short numeric tokens (~8 chars).

    The shape check only guards against pasted garbage; the API (--check) is
    the real validator, so anything in the observed charset is let through.
    """
    assert parse_session_token("https://127.0.0.1/?apisession=12345678") == "12345678"
    assert parse_session_token("12345678") == "12345678"
    assert save_session_token("12345678", tmp_path / "sess.json")


def test_login_url_encodes_the_app_key():
    url = login_url("my key/with+chars")
    assert url.startswith("https://api.icicidirect.com/apiuser/login?api_key=")
    assert " " not in url and "+" not in url.split("=", 1)[1]
    with pytest.raises(ValueError):
        login_url("")


# ---------------------------------------------------------------------------
# token store
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "sess.json"
    save_session_token(TOKEN, p, api_key="appkey123456")
    assert load_session_token(p) == TOKEN
    payload = json.loads(p.read_text())
    assert payload["date"] == ist_today().isoformat()
    assert payload["api_key_suffix"] == "123456"


def test_stale_token_is_refused(tmp_path):
    """Yesterday's token must fail here, not as an opaque 401 later."""
    p = tmp_path / "sess.json"
    yesterday = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    p.write_text(json.dumps({"date": (ist_today(yesterday)).isoformat(), "token": TOKEN}))
    with pytest.raises(BreezeSessionError, match="midnight"):
        load_session_token(p)


def test_missing_and_corrupt_token_files_raise(tmp_path):
    with pytest.raises(BreezeSessionError, match="no Breeze session token"):
        load_session_token(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(BreezeSessionError, match="unreadable"):
        load_session_token(bad)


def test_save_refuses_a_shapeless_token(tmp_path):
    with pytest.raises(BreezeSessionError):
        save_session_token("short", tmp_path / "s.json")


def test_session_status_summarises_without_raising(tmp_path):
    p = tmp_path / "sess.json"
    assert session_status(p)["present"] is False
    save_session_token(TOKEN, p)
    st = session_status(p)
    assert st["present"] and st["current"] and st["age_days"] == 0


# ---------------------------------------------------------------------------
# provider (fake SDK client -- no network, no credentials)
# ---------------------------------------------------------------------------

class FakeBreeze:
    """Records calls and returns the REST-shaped payload the SDK also emits."""

    def __init__(self, series: dict[str, pd.Series]):
        self.series = series
        self.calls: list[dict] = []

    def get_historical_data_v2(self, **kw):
        self.calls.append(kw)
        sym = kw["stock_code"]
        close = self.series[sym]
        lo = pd.Timestamp(kw["from_date"][:10])
        hi = pd.Timestamp(kw["to_date"][:10])
        window = close.loc[(close.index >= lo) & (close.index <= hi)]
        records = [
            {
                "datetime": ts.strftime("%Y-%m-%dT00:00:00.000Z"),
                "open": float(v) * 0.99, "high": float(v) * 1.01,
                "low": float(v) * 0.98, "close": float(v), "volume": 1000.0,
            }
            for ts, v in window.items()
        ]
        return {"Success": records}


def _series() -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=10)
    return pd.Series([100.0 + i for i in range(10)], index=idx)


def _provider(tmp_path, fake, symbol_map=None, **cfg_kw):
    """A provider wired to a fake SDK client.

    ``symbol_map`` defaults to an *empty injected map* on purpose: with ``None``
    the provider would lazily fetch ICICI's security master, and this suite must
    not touch the network. Tests about the mapping pass one explicitly.
    """
    cfg = DataConfig(price_provider="breeze", start="2022-01-03", **cfg_kw)
    cache = DiskCache(tmp_path, "prices")
    return BreezePriceProvider(cfg, cache, client=fake, symbol_map={} if symbol_map is None else symbol_map), cache


def test_get_ohlcv_returns_normalised_frame(tmp_path):
    fake = FakeBreeze({"AAA": _series()})
    provider, _ = _provider(tmp_path, fake)
    df = provider.get_ohlcv("AAA", "2022-01-03", "2022-01-07")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index[0].date()) == "2022-01-03"
    assert str(df.index[-1].date()) == "2022-01-07"
    assert len(fake.calls) == 1


def test_second_call_fetches_only_the_missing_tail(tmp_path):
    """The whole point of incremental caching: a new session must not refetch history."""
    fake = FakeBreeze({"AAA": _series()})
    provider, _ = _provider(tmp_path, fake)

    provider.get_ohlcv("AAA", "2022-01-03", "2022-01-07")
    provider.get_ohlcv("AAA", "2022-01-03", "2022-01-17")

    assert len(fake.calls) == 2
    # Resumes the day after the cached tail (a Saturday here); the fetch window
    # is calendar-arithmetic, not a trading-calendar lookup.
    assert fake.calls[1]["from_date"].startswith("2022-01-08")
    assert fake.calls[1]["stock_code"] == "AAA"
    assert fake.calls[1]["interval"] == "1day"
    assert fake.calls[1]["exchange_code"] == "NSE"
    assert fake.calls[1]["product_type"] == "cash"


def test_empty_payload_is_not_an_error(tmp_path):
    """A weekend/holiday gap returns no records; that must not raise or retry four times."""
    fake = FakeBreeze({"AAA": _series()})
    provider, _ = _provider(tmp_path, fake)
    first = provider.get_ohlcv("AAA", "2022-01-03", "2022-01-14")
    n = len(fake.calls)
    out = provider.get_ohlcv("AAA", "2022-01-03", "2022-01-20")
    assert len(out) == len(first)          # nothing invented
    assert len(fake.calls) == n + 1        # one tail attempt, not max_retries


def test_cache_hit_does_not_refetch(tmp_path):
    fake = FakeBreeze({"AAA": _series()})
    provider, _ = _provider(tmp_path, fake)
    provider.get_ohlcv("AAA", "2022-01-03", "2022-01-14")
    n = len(fake.calls)
    provider.get_ohlcv("AAA", "2022-01-03", "2022-01-14")
    assert len(fake.calls) == n


def test_weekend_gap_is_not_refetched(tmp_path):
    """Cached through Friday, asked for Saturday: no weekday in the gap, no call."""
    fake = FakeBreeze({"AAA": _series()})
    provider, _ = _provider(tmp_path, fake)
    provider.get_ohlcv("AAA", "2022-01-03", "2022-01-14")
    n = len(fake.calls)
    out = provider.get_ohlcv("AAA", "2022-01-03", "2022-01-15")
    assert len(fake.calls) == n
    assert str(out.index[-1].date()) == "2022-01-14"


def test_unadjusted_jump_is_flagged(tmp_path):
    """A raw split shows up as a >60% move; the provider must say so once."""
    idx = pd.bdate_range("2022-01-03", periods=5)
    close = pd.Series([100.0, 101.0, 50.0, 50.5, 49.0], index=idx)  # a 1:2 split
    fake = FakeBreeze({"SPLIT": close})
    provider, _ = _provider(tmp_path, fake)
    provider.get_ohlcv("SPLIT", "2022-01-03", "2022-01-07")
    assert "SPLIT" in provider._warned_unadjusted


def test_get_many_reports_what_arrived(tmp_path):
    fake = FakeBreeze({"AAA": _series(), "BBB": _series()})
    provider, _ = _provider(tmp_path, fake, request_delay_sec=0.0)
    out = provider.get_many(["AAA", "BBB", "CCC"], "2022-01-03", "2022-01-07")
    assert set(out) == {"AAA", "BBB"}


def test_factory_resolves_breeze():
    cfg = DataConfig(price_provider="breeze")
    assert isinstance(make_price_provider(cfg), BreezePriceProvider)


def test_missing_sdk_or_credentials_fails_loudly(tmp_path, monkeypatch):
    # Neutralise any real .env on the developer's machine: this test is about the
    # guard firing when nothing is configured.
    monkeypatch.setattr("swingml.data.prices.load_env_file", lambda *a, **k: 0)
    monkeypatch.delenv("BREEZE_API_KEY", raising=False)
    monkeypatch.delenv("BREEZE_SECRET_KEY", raising=False)
    cfg = DataConfig(price_provider="breeze")
    provider = BreezePriceProvider(cfg, DiskCache(tmp_path, "prices"))
    with pytest.raises(RuntimeError):
        provider._breeze()


def test_load_env_file_sets_values_and_shell_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("BREEZE_TEST_B", "from-shell")
    monkeypatch.delenv("BREEZE_TEST_A", raising=False)
    monkeypatch.delenv("BREEZE_TEST_C", raising=False)
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "BREEZE_TEST_A=from-file\n"
        "BREEZE_TEST_B=from-file\n"
        'BREEZE_TEST_C="quoted"\n'
        "\n"
        "not a valid line\n",
        encoding="utf-8",
    )
    n = load_env_file(p)
    assert os.environ["BREEZE_TEST_A"] == "from-file"
    assert os.environ["BREEZE_TEST_B"] == "from-shell"  # existing env is not clobbered
    assert os.environ["BREEZE_TEST_C"] == "quoted"
    assert n == 2  # B is skipped, the malformed line ignored


def test_load_env_file_missing_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == 0


# ---------------------------------------------------------------------------
# symbol mapping (the "per-symbol data gap" was a code mismatch)
# ---------------------------------------------------------------------------

def test_nse_symbol_is_translated_to_the_breeze_code(tmp_path):
    """The request must carry Breeze's own code, not the NSE trading symbol."""
    fake = FakeBreeze({"RELIND": _series()})
    provider, cache = _provider(tmp_path, fake, symbol_map={"RELIANCE": "RELIND"})
    df = provider.get_ohlcv("RELIANCE", "2022-01-03", "2022-01-07")
    assert fake.calls[0]["stock_code"] == "RELIND"
    assert len(df) == 5
    # The cache key stays the NSE symbol, so the snapshot survives a remap.
    assert cache.exists("breeze_RELIANCE")


def test_identical_code_is_left_alone(tmp_path):
    fake = FakeBreeze({"TCS": _series()})
    provider, _ = _provider(tmp_path, fake, symbol_map={"TCS": "TCS"})
    provider.get_ohlcv("TCS", "2022-01-03", "2022-01-07")
    assert fake.calls[0]["stock_code"] == "TCS"


def test_symbol_absent_from_the_master_is_requested_as_is_and_recorded(tmp_path):
    fake = FakeBreeze({"NEWLIST": _series()})
    provider, _ = _provider(tmp_path, fake, symbol_map={"AAA": "AAA"})
    provider.get_ohlcv("NEWLIST", "2022-01-03", "2022-01-07")
    assert fake.calls[0]["stock_code"] == "NEWLIST"
    assert "NEWLIST" in provider._unmapped


def test_benchmark_index_is_not_run_through_the_equity_map(tmp_path):
    fake = FakeBreeze({"NIFTY": _series()})
    provider, _ = _provider(tmp_path, fake, symbol_map={"NIFTY": "WRONGCODE"})
    assert provider._breeze_code("NIFTY") == "NIFTY"
    assert provider._breeze_code("nifty") == "NIFTY"


def test_unavailable_master_degrades_to_identity_not_an_exception(tmp_path, monkeypatch):
    """A dead master must not kill a 460-symbol refresh -- but it must not be silent."""
    monkeypatch.setattr(
        "swingml.data.prices.load_symbol_map",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")),
    )
    fake = FakeBreeze({"AAA": _series()})
    provider = BreezePriceProvider(
        DataConfig(price_provider="breeze", start="2022-01-03"),
        DiskCache(tmp_path, "prices"),
        client=fake,
        symbol_map=None,
    )
    df = provider.get_ohlcv("AAA", "2022-01-03", "2022-01-07")
    assert len(df) == 5
    assert fake.calls[0]["stock_code"] == "AAA"


def test_get_many_resolves_each_symbol(tmp_path):
    fake = FakeBreeze({"RELIND": _series(), "HDFBAN": _series()})
    provider, _ = _provider(
        tmp_path, fake,
        symbol_map={"RELIANCE": "RELIND", "HDFCBANK": "HDFBAN"},
        request_delay_sec=0.0,
    )
    out = provider.get_many(["RELIANCE", "HDFCBANK"], "2022-01-03", "2022-01-07")
    assert set(out) == {"RELIANCE", "HDFCBANK"}
    assert [c["stock_code"] for c in fake.calls] == ["RELIND", "HDFBAN"]


def test_unknown_payload_shape_raises(tmp_path):
    class Bad:
        def get_historical_data_v2(self, **kw):
            return {"nonsense": []}

    provider, _ = _provider(tmp_path, Bad())
    with pytest.raises(BreezeApiError, match="no 'Success' payload"):
        provider.get_ohlcv("AAA", "2022-01-03", "2022-01-07")


def test_error_payload_is_raised_not_swallowed(tmp_path):
    """The SDK reports bad params/symbols as DATA, not exceptions.

    ``{"Success": "", "Status": 500, "Error": ...}`` must surface the reason,
    not degrade into an empty frame that reads as "delisted symbol".
    """
    class Rejected:
        def get_historical_data_v2(self, **kw):
            return {"Success": "", "Status": 500, "Error": "Invalid stock code"}

    provider, _ = _provider(tmp_path, Rejected())
    with pytest.raises(BreezeApiError, match="Invalid stock code"):
        provider.get_ohlcv("ZZZZ", "2022-01-03", "2022-01-07")


def test_definitive_error_is_not_retried(tmp_path):
    """A rejected symbol is answered once, not four times against the rate limit."""
    class Rejected:
        def __init__(self):
            self.calls = 0

        def get_historical_data_v2(self, **kw):
            self.calls += 1
            return {"Success": "", "Status": 500, "Error": "Invalid stock code"}

    fake = Rejected()
    provider, _ = _provider(tmp_path, fake)
    with pytest.raises(BreezeApiError):
        provider.get_ohlcv("ZZZZ", "2022-01-03", "2022-01-07")
    assert fake.calls == 1
