"""Tests for validate_with_policy() — routing reparar / cuarentena / descartar."""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.cleaning.policy import ValidationResult, validate_with_policy


def _valid_df(**overrides) -> pd.DataFrame:
    """Minimal fully-valid budget row after cleaning."""
    row = {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme S.L.",
        "client_code": "CLI-0042",
        "currency": "EUR",
        "total_amount": 48000.0,
        "signed_at": pd.Timestamp("2024-03-15"),
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# DataFrame completamente válido
# ---------------------------------------------------------------------------

def test_fully_valid_row_goes_to_valid():
    result = validate_with_policy(_valid_df())
    assert len(result.valid) == 1
    assert result.quarantined.empty
    assert result.discarded.empty


def test_fully_valid_report_has_expected_keys():
    result = validate_with_policy(_valid_df())
    for key in ("input_rows", "valid_rows", "quarantined_rows", "discarded_rows"):
        assert key in result.report


def test_fully_valid_report_counts_match():
    result = validate_with_policy(_valid_df())
    assert result.report["input_rows"] == 1
    assert result.report["valid_rows"] == 1
    assert result.report["quarantined_rows"] == 0
    assert result.report["discarded_rows"] == 0


# ---------------------------------------------------------------------------
# Cuarentena — campo nullable=False con valor nulo (client_code)
# ---------------------------------------------------------------------------

def test_non_nullable_field_null_goes_to_quarantine():
    """client_code is nullable=False — NA triggers 'not_nullable' → quarantine."""
    result = validate_with_policy(_valid_df(client_code=pd.NA))
    assert len(result.quarantined) == 1
    assert result.discarded.empty
    assert result.valid.empty


def test_quarantine_report_count():
    result = validate_with_policy(_valid_df(client_code=pd.NA))
    assert result.report["quarantined_rows"] == 1
    assert result.report["discarded_rows"] == 0


# ---------------------------------------------------------------------------
# Descarte — valor fuera de rango (total_amount negativo)
# ---------------------------------------------------------------------------

def test_negative_amount_goes_to_discard():
    result = validate_with_policy(_valid_df(total_amount=-1.0))
    assert len(result.discarded) == 1
    assert result.quarantined.empty
    assert result.valid.empty


def test_discard_report_count_negative_amount():
    result = validate_with_policy(_valid_df(total_amount=-1.0))
    assert result.report["discarded_rows"] == 1
    assert result.report["quarantined_rows"] == 0


def test_amount_above_maximum_goes_to_discard():
    result = validate_with_policy(_valid_df(total_amount=10_000_001.0))
    assert len(result.discarded) == 1


# ---------------------------------------------------------------------------
# Descarte — budget_id malformado (str_matches)
# ---------------------------------------------------------------------------

def test_malformed_budget_id_goes_to_discard():
    result = validate_with_policy(_valid_df(budget_id="PRESUP-001"))
    assert len(result.discarded) == 1
    assert result.quarantined.empty


# ---------------------------------------------------------------------------
# Descarte — currency fuera del catálogo (isin)
# ---------------------------------------------------------------------------

def test_unknown_currency_goes_to_discard():
    result = validate_with_policy(_valid_df(currency="BTC"))
    assert len(result.discarded) == 1


# ---------------------------------------------------------------------------
# Discard gana sobre quarantine cuando una fila falla ambos checks
# ---------------------------------------------------------------------------

def test_discard_beats_quarantine_on_same_row():
    """A row with both client_code=NA (quarantine) and total_amount=-1 (discard)
    must end up in discard, not quarantine."""
    result = validate_with_policy(_valid_df(client_code=pd.NA, total_amount=-1.0))
    assert len(result.discarded) == 1
    assert result.quarantined.empty


# ---------------------------------------------------------------------------
# Múltiples filas — routing independiente
# ---------------------------------------------------------------------------

def test_mixed_rows_routed_independently():
    df = pd.concat([
        _valid_df(),                          # → valid
        _valid_df(budget_id="BUDGET-2024-0002", client_code=pd.NA),   # → quarantine
        _valid_df(budget_id="BUDGET-2024-0003", total_amount=-500.0), # → discard
    ], ignore_index=True)
    result = validate_with_policy(df)
    assert len(result.valid) == 1
    assert len(result.quarantined) == 1
    assert len(result.discarded) == 1


def test_report_failures_by_check_populated_on_errors():
    result = validate_with_policy(_valid_df(total_amount=-1.0))
    assert "failures_by_check" in result.report
    assert len(result.report["failures_by_check"]) > 0


# ---------------------------------------------------------------------------
# DataFrame vacío
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_result():
    empty = pd.DataFrame(columns=["budget_id", "client_name", "client_code",
                                   "currency", "total_amount", "signed_at"])
    result = validate_with_policy(empty)
    assert result.valid.empty
    assert result.quarantined.empty
    assert result.discarded.empty
    assert result.report.get("input_rows") == 0


# ---------------------------------------------------------------------------
# Tipo de retorno siempre es ValidationResult
# ---------------------------------------------------------------------------

def test_return_type_is_always_validation_result():
    result = validate_with_policy(_valid_df())
    assert isinstance(result, ValidationResult)
