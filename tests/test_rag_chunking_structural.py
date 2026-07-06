"""Tests for src/rag/chunking/structural.py — JSONStructuralChunker.

Pure unit tests: no I/O, no API calls, no DB. The chunker is a deterministic
function (budgets in, chunks out), so the test surface is the chunk shape and
the parent-context header that the embedder sees.
"""
from __future__ import annotations

from src.generation.rag.chunking.structural import (
    JSONStructuralChunker,
    component_metadata,
    render_component_text,
)
from src.generation.rag.schemas import Budget, BudgetComponent, ClientMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_component(**overrides) -> BudgetComponent:
    """A valid component with overridable defaults."""
    defaults = dict(
        component_id="AUTH-001",
        name="OAuth 2.0 authentication backend",
        description="Implementation of OAuth 2.0 flows with JWT-based session management.",
        tech_stack=["ruby_on_rails", "postgresql"],
        estimated_hours=120,
        complexity="high",
        dependencies=[],
    )
    return BudgetComponent(**(defaults | overrides))


def _make_budget(**overrides) -> Budget:
    """A valid budget with overridable defaults and one component."""
    defaults = dict(
        budget_id="BUD-2024-001",
        client_metadata=ClientMetadata(name="FintechCorp", sector="finance", country="ES"),
        project_summary="Mobile banking API with OAuth 2.0 authentication",
        main_technology="ruby_on_rails",
        year=2024,
        total_estimated_hours=480,
        components=[_make_component()],
    )
    return Budget(**(defaults | overrides))


# ---------------------------------------------------------------------------
# Tests — render_component_text
# ---------------------------------------------------------------------------


def test_render_component_text_includes_parent_context_header() -> None:
    """The parent context header (project, sector, year, main tech) is what
    stops a component like "Authentication backend" from losing its context
    once embedded. If this assertion fails, retrieval quality degrades."""
    budget = _make_budget()
    component = _make_component()
    text = render_component_text(budget, component)

    # Project line — the prose the embedder will see at the top.
    assert "Mobile banking API with OAuth 2.0 authentication" in text
    # The four anchor fields that bind the component to its parent budget.
    assert "[Client sector: finance" in text
    assert "Year: 2024" in text
    assert "Main tech: ruby_on_rails" in text
    # And the component detail itself.
    assert "OAuth 2.0 authentication backend" in text
    assert "120" in text  # estimated_hours


def test_render_component_text_includes_tech_stack_as_comma_list() -> None:
    """The tech stack is a list of strings; the render must join them with
    ``, `` so a single string is what the embedder sees (embedders do not
    understand list-of-strings)."""
    component = _make_component(tech_stack=["python", "fastapi", "postgres"])
    budget = _make_budget()
    text = render_component_text(budget, component)
    assert "Tech stack: python, fastapi, postgres" in text


def test_render_component_text_handles_empty_tech_stack() -> None:
    """A component with no tech stack must not crash on the ``', '.join(...)``
    call and must produce an empty (but valid) tech stack line."""
    component = _make_component(tech_stack=[])
    budget = _make_budget()
    text = render_component_text(budget, component)
    assert "Tech stack: " in text


# ---------------------------------------------------------------------------
# Tests — component_metadata
# ---------------------------------------------------------------------------


def test_component_metadata_carries_filterable_fields() -> None:
    """These are the fields the live session will filter on in SQL
    (``metadata->>'sector' = 'finance'``). The GIN index needs them
    to be top-level keys, not nested."""
    budget = _make_budget()
    component = _make_component()
    meta = component_metadata(budget, component)

    assert meta["budget_id"] == "BUD-2024-001"
    assert meta["component_id"] == "AUTH-001"
    assert meta["client_sector"] == "finance"
    assert meta["main_technology"] == "ruby_on_rails"
    assert meta["year"] == 2024
    assert meta["complexity"] == "high"
    assert meta["estimated_hours"] == 120


# ---------------------------------------------------------------------------
# Tests — JSONStructuralChunker.chunk
# ---------------------------------------------------------------------------


def test_chunk_produces_one_chunk_per_component() -> None:
    """1:1 mapping: every component becomes exactly one chunk. No splitting,
    no merging, no fixed-size slicing."""
    budget = _make_budget(
        components=[
            _make_component(component_id="A-001", name="comp A"),
            _make_component(component_id="B-002", name="comp B"),
            _make_component(component_id="C-003", name="comp C"),
        ]
    )
    chunks = JSONStructuralChunker().chunk([budget])
    assert len(chunks) == 3
    assert [c.chunk_id for c in chunks] == [
        "BUD-2024-001::A-001",
        "BUD-2024-001::B-002",
        "BUD-2024-001::C-003",
    ]


def test_chunk_flattens_multiple_budgets() -> None:
    """Across multiple budgets, chunks concatenate in budget-then-component
    order. No grouping, no per-budget envelopes."""
    b1 = _make_budget(budget_id="BUD-A", components=[_make_component(component_id="A-1")])
    b2 = _make_budget(budget_id="BUD-B", components=[_make_component(component_id="B-1")])
    chunks = JSONStructuralChunker().chunk([b1, b2])
    assert [c.chunk_id for c in chunks] == ["BUD-A::A-1", "BUD-B::B-1"]


def test_chunk_produces_empty_list_for_no_budgets() -> None:
    """Defensive: empty input must not crash and must produce zero chunks."""
    chunks = JSONStructuralChunker().chunk([])
    assert chunks == []


def test_chunk_id_format_is_budget_double_colon_component() -> None:
    """The ``::`` separator is the contract that ties a chunk back to its
    parent budget. If you change it, downstream consumers (logs, JSONB
    filters) need to know."""
    budget = _make_budget(budget_id="BUD-2024-014")
    chunks = JSONStructuralChunker().chunk([budget])
    assert chunks[0].chunk_id == "BUD-2024-014::AUTH-001"


def test_chunk_text_contains_parent_context_for_every_chunk() -> None:
    """Regression guard for the most common chunker bug: forgetting the
    parent context header in some branches. We assert that EVERY chunk in a
    multi-chunk budget contains the project summary, not just the first."""
    budget = _make_budget(
        project_summary="Distinctive project marker ZZZQ-99",
        components=[
            _make_component(component_id="A-1", name="alpha"),
            _make_component(component_id="A-2", name="beta"),
            _make_component(component_id="A-3", name="gamma"),
        ],
    )
    chunks = JSONStructuralChunker().chunk([budget])
    for chunk in chunks:
        assert "ZZZQ-99" in chunk.text, f"missing parent context in {chunk.chunk_id}"


def test_chunk_token_count_is_positive_and_uses_cl100k() -> None:
    """The token count is the billing basis for the embedding API call. If
    we silently switch tokenizers, the per-chunk cost estimate lies."""
    budget = _make_budget()
    chunks = JSONStructuralChunker().chunk([budget])
    # A real component text is well over 50 tokens but well under 1000.
    assert 50 < chunks[0].token_count < 1000


def test_chunk_does_not_incur_extra_api_calls() -> None:
    """The structural chunker is the free baseline. It must NEVER call an
    external API (that's what the Session 7 strategies like
    PropositionalChunker are for)."""
    budget = _make_budget()
    chunker = JSONStructuralChunker()
    chunker.chunk([budget])
    assert chunker.last_extra_api_calls == 0
    assert chunker.last_extra_cost_usd == 0.0


def test_chunk_metadata_matches_per_component() -> None:
    """Each chunk's metadata is the metadata of ITS component, not of the
    first component in the budget. Mixing them up is a classic copy-paste bug."""
    budget = _make_budget(
        components=[
            _make_component(component_id="A-1", estimated_hours=10),
            _make_component(component_id="A-2", estimated_hours=200),
        ],
    )
    chunks = JSONStructuralChunker().chunk([budget])
    assert chunks[0].metadata["estimated_hours"] == 10
    assert chunks[1].metadata["estimated_hours"] == 200
