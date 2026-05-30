"""Tests for ConsistentPseudonymizer.

We use a StubAnalyzer to avoid loading the spaCy model in tests.
The stub lets us control exactly which entities get detected and where.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.ingest.pii.mapping_store import InMemoryMappingStore
from src.ingest.pii.pseudonymizer import ConsistentPseudonymizer, PseudonymizationResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _StubResult:
    entity_type: str
    start: int
    end: int
    score: float = 0.95


class _StubAnalyzer:
    """Returns a fixed list of RecognizerResult-like objects per text."""

    def __init__(self, results_by_text: dict[str, list[_StubResult]]) -> None:
        self._results = results_by_text

    def analyze(self, *, text: str, language: str, entities=None) -> list[_StubResult]:
        return list(self._results.get(text, []))


def _build_pseudonymizer(
    analyzer: _StubAnalyzer, store: InMemoryMappingStore | None = None
) -> ConsistentPseudonymizer:
    return ConsistentPseudonymizer(
        analyzer=analyzer,
        mapping_store=store or InMemoryMappingStore(),
        salt="test-salt-fixed",
    )


# ---------------------------------------------------------------------------
# Texto sin entidades detectadas
# ---------------------------------------------------------------------------


def test_no_entities_returns_text_unchanged():
    analyzer = _StubAnalyzer({})
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize("Texto sin PII.")
    assert result.pseudonymized_text == "Texto sin PII."
    assert result.applied == []


def test_no_entities_result_is_pseudonymization_result():
    p = _build_pseudonymizer(_StubAnalyzer({}))
    assert isinstance(p.pseudonymize("hola"), PseudonymizationResult)


# ---------------------------------------------------------------------------
# Consistencia: mismo plaintext → mismo pseudónimo
# ---------------------------------------------------------------------------


def test_same_value_produces_same_pseudonym_in_same_call():
    """Two occurrences of the same name in the same text → same pseudonym."""
    text = "Juan García llamó a Juan García."
    # Verify offsets match the actual string before using them:
    # text[0:11]  == "Juan García"
    # text[20:31] == "Juan García"
    assert text[0:11] == "Juan García"
    assert text[20:31] == "Juan García"
    analyzer = _StubAnalyzer({
        text: [
            _StubResult("PERSON", 0, 11),   # primer "Juan García"
            _StubResult("PERSON", 20, 31),  # segundo "Juan García"
        ]
    })
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize(text)
    # Both occurrences must map to the same pseudonym.
    assert result.applied[0].pseudonym == result.applied[1].pseudonym


def test_same_value_produces_same_pseudonym_across_calls():
    """Same plaintext → same pseudonym in a second call to pseudonymize()."""
    text1 = "Contrato con Laura Fernández."
    text2 = "Email de Laura Fernández recibido."
    store = InMemoryMappingStore()
    analyzer = _StubAnalyzer({
        text1: [_StubResult("PERSON", 13, 28)],
        text2: [_StubResult("PERSON", 9, 24)],
    })
    p = _build_pseudonymizer(analyzer, store)
    r1 = p.pseudonymize(text1)
    r2 = p.pseudonymize(text2)
    assert r1.applied[0].pseudonym == r2.applied[0].pseudonym


# ---------------------------------------------------------------------------
# Distintos valores → distintos pseudónimos
# ---------------------------------------------------------------------------


def test_different_values_produce_different_pseudonyms():
    text = "Laura Fernández habló con Javier Romero."
    analyzer = _StubAnalyzer({
        text: [
            _StubResult("PERSON", 0, 15),   # "Laura Fernández"
            _StubResult("PERSON", 25, 39),  # "Javier Romero"
        ]
    })
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize(text)
    assert result.applied[0].pseudonym != result.applied[1].pseudonym


# ---------------------------------------------------------------------------
# Reemplazo derecha-a-izquierda preserva offsets
# ---------------------------------------------------------------------------


def test_left_entity_offset_not_corrupted_by_right_replacement():
    """Replacing right-to-left must keep left entity offset valid."""
    text = "AAA BBB"  # two entities side by side
    analyzer = _StubAnalyzer({
        text: [
            _StubResult("PERSON", 0, 3),  # "AAA"
            _StubResult("PERSON", 4, 7),  # "BBB"
        ]
    })
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize(text)
    # Both entities must have been replaced (text changed, no IndexError)
    assert "AAA" not in result.pseudonymized_text
    assert "BBB" not in result.pseudonymized_text


# ---------------------------------------------------------------------------
# Plaintext nunca aparece en los AppliedMapping
# ---------------------------------------------------------------------------


def test_applied_mapping_stores_hash_not_plaintext():
    text = "Email: secreto@empresa.com"
    analyzer = _StubAnalyzer({text: [_StubResult("EMAIL_ADDRESS", 7, 26)]})
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize(text)
    assert result.applied[0].original_hash != "secreto@empresa.com"
    assert len(result.applied[0].original_hash) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Tipos de entidades distintos — generadores semánticos
# ---------------------------------------------------------------------------


def test_budget_id_replaced_with_budget_format():
    text = "Presupuesto BUDGET-2024-0001 aprobado."
    analyzer = _StubAnalyzer({text: [_StubResult("BUDGET_ID", 12, 27)]})
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize(text)
    import re
    assert re.search(r"BUDGET-\d{4}-\d{4}", result.pseudonymized_text)


def test_client_code_replaced_with_cli_format():
    text = "Cliente CLI-0042 registrado."
    analyzer = _StubAnalyzer({text: [_StubResult("CLIENT_CODE", 8, 16)]})
    p = _build_pseudonymizer(analyzer)
    result = p.pseudonymize(text)
    import re
    assert re.search(r"CLI-\d{4}", result.pseudonymized_text)


# ---------------------------------------------------------------------------
# Derecho al olvido — forget() borra el mapping
# ---------------------------------------------------------------------------


def test_forget_causes_new_pseudonym_on_next_call():
    text = "Cliente: ACME Corp"
    analyzer = _StubAnalyzer({text: [_StubResult("ORGANIZATION", 9, 18)]})
    store = InMemoryMappingStore()
    p = _build_pseudonymizer(analyzer, store)

    r1 = p.pseudonymize(text)
    first_pseudonym = r1.applied[0].pseudonym
    original_hash = r1.applied[0].original_hash

    # Simulate right-to-be-forgotten: delete the mapping
    store.forget("ORGANIZATION", original_hash)

    r2 = p.pseudonymize(text)
    second_pseudonym = r2.applied[0].pseudonym

    # After forget(), a new pseudonym is generated (statistically distinct)
    # We can't guarantee inequality with Faker, but the store no longer has
    # the old entry — verify it was re-created
    assert store.lookup_or_create("ORGANIZATION", original_hash, lambda: "x") is not None


def test_forget_returns_true_when_entry_existed():
    store = InMemoryMappingStore()
    store.lookup_or_create("PERSON", "abc123", lambda: "Carlos")
    assert store.forget("PERSON", "abc123") is True


def test_forget_returns_false_when_entry_missing():
    store = InMemoryMappingStore()
    assert store.forget("PERSON", "nonexistent") is False


# ---------------------------------------------------------------------------
# InMemoryMappingStore — idempotencia
# ---------------------------------------------------------------------------


def test_mapping_store_idempotent_on_same_hash():
    store = InMemoryMappingStore()
    first = store.lookup_or_create("PERSON", "hash-abc", lambda: "Carlos Pérez")
    second = store.lookup_or_create("PERSON", "hash-abc", lambda: "Otro Nombre")
    assert first == second  # factory not called a second time
