"""Tests for clean_budget_records() — five cleaning steps."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.cleaning.budget_records import clean_budget_records


def _base_record(**overrides) -> dict:
    record = {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme S.L.",
        "client_code": "CLI-0042",
        "currency": "EUR",
        "total_amount": 48000,
        "signed_at": "2024-03-15",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Paso 1 — nulos disfrazados
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "placeholder", ["TBD", "N/A", "n/a", "tbd", "", "null", "None", "-"]
)
def test_null_placeholder_in_client_name_becomes_na(placeholder):
    df = clean_budget_records([_base_record(client_name=placeholder)])
    assert pd.isna(df.loc[0, "client_name"])


def test_real_client_name_is_preserved():
    df = clean_budget_records([_base_record(client_name="Acme S.L.")])
    assert df.loc[0, "client_name"] == "Acme S.L."


def test_null_placeholder_only_affects_optional_text_columns():
    """budget_id with a placeholder string is NOT converted to NA."""
    df = clean_budget_records([_base_record(budget_id="TBD")])
    # budget_id is not in the nullable columns list — it must survive as-is
    assert df.loc[0, "budget_id"] == "TBD"


# ---------------------------------------------------------------------------
# Paso 2 — currency casing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("eur", "EUR"),
        ("Eur", "EUR"),
        ("USD", "USD"),
        ("usd", "USD"),
    ],
)
def test_currency_normalised_to_uppercase(raw, expected):
    df = clean_budget_records([_base_record(currency=raw)])
    assert df.loc[0, "currency"] == expected


def test_unknown_currency_preserved_as_uppercase():
    """BTC is not valid per schema, but cleaning just uppercases — doesn't discard."""
    df = clean_budget_records([_base_record(currency="btc")])
    assert df.loc[0, "currency"] == "BTC"


# ---------------------------------------------------------------------------
# Paso 3 — date coercion
# ---------------------------------------------------------------------------


def test_iso_date_parsed_correctly():
    df = clean_budget_records([_base_record(signed_at="2024-03-15")])
    ts = df.loc[0, "signed_at"]
    assert ts.year == 2024
    assert ts.month == 3
    assert ts.day == 15


def test_spanish_date_format_parsed_with_dayfirst():
    df = clean_budget_records([_base_record(signed_at="12/03/2024")])
    ts = df.loc[0, "signed_at"]
    assert ts.year == 2024
    assert ts.month == 3
    assert ts.day == 12


def test_unparseable_date_becomes_nat():
    df = clean_budget_records([_base_record(signed_at="not-a-date")])
    assert pd.isna(df.loc[0, "signed_at"])


# ---------------------------------------------------------------------------
# Paso 4 — numeric coercion
# ---------------------------------------------------------------------------


def test_string_amount_coerced_to_float():
    df = clean_budget_records([_base_record(total_amount="80000")])
    assert df.loc[0, "total_amount"] == 80000.0


def test_non_numeric_amount_becomes_nan():
    df = clean_budget_records([_base_record(total_amount="not-a-number")])
    assert pd.isna(df.loc[0, "total_amount"])


# ---------------------------------------------------------------------------
# Paso 5 — dedup por budget_id con regla "keep latest signed_at"
# ---------------------------------------------------------------------------


def test_exact_duplicate_is_removed():
    records = [_base_record(), _base_record()]
    df = clean_budget_records(records)
    assert len(df) == 1


def test_divergent_duplicate_keeps_most_recent():
    records = [
        _base_record(total_amount=80000, signed_at="2024-01-01"),
        _base_record(total_amount=82500, signed_at="2024-06-01"),  # ← más reciente
    ]
    df = clean_budget_records(records)
    assert len(df) == 1
    assert df.loc[0, "total_amount"] == 82500.0


def test_divergent_duplicate_older_row_not_kept():
    records = [
        _base_record(total_amount=80000, signed_at="2024-01-01"),  # ← más antigua
        _base_record(total_amount=82500, signed_at="2024-06-01"),
    ]
    df = clean_budget_records(records)
    assert df.loc[0, "total_amount"] != 80000.0


# ---------------------------------------------------------------------------
# Comportamiento general
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_dataframe():
    df = clean_budget_records([])
    assert df.empty


def test_function_never_raises_on_dirty_data():
    """Cleaning must always return a DataFrame, never raise."""
    dirty = [
        _base_record(total_amount="??", signed_at="bad-date", currency=None),
    ]
    df = clean_budget_records(dirty)
    assert isinstance(df, pd.DataFrame)
