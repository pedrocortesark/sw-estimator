"""Tests for domain-specific Presidio recognizers.

PatternRecognizers operate purely on regex — no spaCy model needed.
We call `.analyze()` directly with `nlp_artifacts=None` to keep tests fast.
"""
from __future__ import annotations

import pytest

from src.ingest.pii.recognizers import BudgetIdRecognizer, ClientCodeRecognizer

# ---------------------------------------------------------------------------
# BudgetIdRecognizer
# ---------------------------------------------------------------------------


def test_budget_id_detected_in_sentence():
    recognizer = BudgetIdRecognizer()
    results = recognizer.analyze(
        text="Revisando BUDGET-2024-0001 con el equipo.",
        entities=["BUDGET_ID"],
        nlp_artifacts=None,
    )
    assert len(results) == 1
    assert results[0].entity_type == "BUDGET_ID"


def test_budget_id_score_high():
    recognizer = BudgetIdRecognizer()
    results = recognizer.analyze(
        text="BUDGET-2024-0315",
        entities=["BUDGET_ID"],
        nlp_artifacts=None,
    )
    assert results[0].score >= 0.9


def test_budget_id_span_correct():
    text = "El presupuesto BUDGET-2023-0042 está firmado."
    recognizer = BudgetIdRecognizer()
    results = recognizer.analyze(text=text, entities=["BUDGET_ID"], nlp_artifacts=None)
    assert len(results) == 1
    assert text[results[0].start : results[0].end] == "BUDGET-2023-0042"


def test_budget_id_incomplete_pattern_not_matched():
    """BUDGET-2024 without the serial part must not match."""
    recognizer = BudgetIdRecognizer()
    results = recognizer.analyze(
        text="Ver BUDGET-2024 para más info.",
        entities=["BUDGET_ID"],
        nlp_artifacts=None,
    )
    assert len(results) == 0


def test_budget_id_wrong_separator_not_matched():
    recognizer = BudgetIdRecognizer()
    results = recognizer.analyze(
        text="BUDGET_2024_0001",
        entities=["BUDGET_ID"],
        nlp_artifacts=None,
    )
    assert len(results) == 0


def test_budget_id_multiple_occurrences():
    recognizer = BudgetIdRecognizer()
    results = recognizer.analyze(
        text="BUDGET-2024-0001 y BUDGET-2025-0099 en el mismo documento.",
        entities=["BUDGET_ID"],
        nlp_artifacts=None,
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# ClientCodeRecognizer
# ---------------------------------------------------------------------------


def test_client_code_detected():
    recognizer = ClientCodeRecognizer()
    results = recognizer.analyze(
        text="Cliente CLI-0042 activo.",
        entities=["CLIENT_CODE"],
        nlp_artifacts=None,
    )
    assert len(results) == 1
    assert results[0].entity_type == "CLIENT_CODE"


def test_client_code_score_high():
    recognizer = ClientCodeRecognizer()
    results = recognizer.analyze(
        text="CLI-1234",
        entities=["CLIENT_CODE"],
        nlp_artifacts=None,
    )
    assert results[0].score >= 0.9


def test_client_code_span_correct():
    text = "Código: CLI-0099 registrado."
    recognizer = ClientCodeRecognizer()
    results = recognizer.analyze(text=text, entities=["CLIENT_CODE"], nlp_artifacts=None)
    assert text[results[0].start : results[0].end] == "CLI-0099"


def test_client_code_too_few_digits_not_matched():
    """CLI-42 has only 2 digits — must NOT match."""
    recognizer = ClientCodeRecognizer()
    results = recognizer.analyze(
        text="Ver CLI-42 en el sistema.",
        entities=["CLIENT_CODE"],
        nlp_artifacts=None,
    )
    assert len(results) == 0


def test_client_code_too_many_digits_not_matched():
    """CLI-12345 has 5 digits — must NOT match."""
    recognizer = ClientCodeRecognizer()
    results = recognizer.analyze(
        text="Código CLI-12345.",
        entities=["CLIENT_CODE"],
        nlp_artifacts=None,
    )
    assert len(results) == 0


def test_client_code_multiple_occurrences():
    recognizer = ClientCodeRecognizer()
    results = recognizer.analyze(
        text="CLI-0001 y CLI-9999 son clientes distintos.",
        entities=["CLIENT_CODE"],
        nlp_artifacts=None,
    )
    assert len(results) == 2
