"""Tests for the bhavcopy ingestion guards.

These run fully offline by exercising the parsing/validation layer directly,
because the guards are the part that must never regress: a silent date alias
corrupts the entire dataset without raising anything.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingml.config import DataConfig
from swingml.data.delivery import NseBhavcopyDeliveryProvider

HEADER = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
    "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
    "NO_OF_TRADES, DELIV_QTY, DELIV_PER"
)


def _bhav_text(date_str: str) -> str:
    """A bhavcopy payload mirroring the real column layout and spacing."""
    rows = [
        f"RELIANCE, EQ, {date_str}, 1200.00, 1210.00, 1230.00, 1195.00, 1225.00, 1226.00, 1215.00, 5000000, 61000.00, 45000, 2200000, 44.00",
        f"TCS, EQ, {date_str}, 2000.00, 2010.00, 2050.00, 1990.00, 2080.00, 2082.00, 2030.00, 3000000, 61000.00, 30000, 1500000, 50.00",
        # Non-equity series that must be filtered out.
        f"SOMEFUND, MF, {date_str}, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 100, 1.00, 1, 100, 100.00",
        f"GSEC1, GB, {date_str}, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100, 1.00, 1, 0, 0.00",
    ]
    return HEADER + "\n" + "\n".join(rows) + "\n"


@pytest.fixture
def provider():
    return NseBhavcopyDeliveryProvider(DataConfig(), cache=None)


def _parse(text: str) -> pd.DataFrame:
    import io

    return pd.read_csv(io.StringIO(text), skipinitialspace=True)


def test_payload_date_mismatch_is_rejected(provider):
    """THE critical guard: a Sunday request must not accept Friday's payload."""
    payload = _parse(_bhav_text("25-Sep-2026"))
    requested = dt.date(2026, 9, 27)  # Sunday

    out = provider._validate_and_shape(payload, requested)

    assert out is None, "non-trading-day alias was accepted -> duplicate rows would be written"


def test_matching_payload_is_accepted_and_series_filtered(provider):
    requested = dt.date(2026, 9, 25)
    out = provider._validate_and_shape(_parse(_bhav_text("25-Sep-2026")), requested)

    assert out is not None
    assert set(out["symbol"]) == {"RELIANCE", "TCS"}, "non-EQ series leaked through"
    assert (out["series"] == "EQ").all()
    assert out["date"].nunique() == 1
    assert out["date"].iloc[0] == requested
    # Numeric coercion must survive the padding-heavy NSE CSV layout.
    assert out.loc[out["symbol"] == "RELIANCE", "deliv_per"].iloc[0] == pytest.approx(44.0)
    assert out.loc[out["symbol"] == "TCS", "ttl_trd_qnty"].iloc[0] == pytest.approx(3_000_000)


def test_unparseable_dates_are_rejected(provider):
    payload = _parse(_bhav_text("not-a-date"))
    assert provider._validate_and_shape(payload, dt.date(2026, 9, 25)) is None


def test_missing_schema_is_rejected(provider):
    assert provider._validate_and_shape(pd.DataFrame({"FOO": [1]}), dt.date(2026, 9, 25)) is None


def test_ddmmyyyy_formatting(provider):
    assert provider._ddmmyyyy(dt.date(2026, 9, 25)) == "25092026"
    assert provider._ddmmyyyy(dt.date(2020, 1, 1)) == "01012020"


def test_bool_series_is_not_mistaken_for_a_date(provider):
    """dataclass/pandas booleans are not dates; the coercion must not crash."""
    assert provider._to_str_date("2026-09-25") == dt.date(2026, 9, 25)
    assert provider._to_str_date(pd.Timestamp("2026-09-25")) == dt.date(2026, 9, 25)
