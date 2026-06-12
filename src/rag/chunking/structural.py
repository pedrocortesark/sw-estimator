"""Structural chunker for budget JSON.

Strategy: trust the document structure. One :class:`BudgetComponent` becomes
exactly one :class:`Chunk`. No overlap, no fixed-size splitting of long
descriptions — if a description is abnormally large, that is a data point we
want to surface (via ``token_count``) and discuss, not paper over.

The parent budget context is prepended to every chunk (a *contextual chunk
header*): without it a component like "Authentication backend" would lose track
of which client and sector it belongs to.
"""
from __future__ import annotations

import time

from src.rag.chunking.base import Chunker, count_tokens, emit_chunking_done
from src.rag.schemas import Budget, BudgetComponent, Chunk


def render_component_text(budget: Budget, component: BudgetComponent) -> str:
    """Parent context header + component detail. This is what gets embedded."""
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.client_metadata.sector} | "
        f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
        f"\n"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


def component_metadata(budget: Budget, component: BudgetComponent) -> dict:
    """Filterable metadata that travels with a component-level chunk."""
    return {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
    }


class JSONStructuralChunker(Chunker):
    """Turns budgets into one chunk per component."""

    strategy_name = "structural"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                chunks.append(self._chunk_component(budget, component))
        self.last_extra_api_calls = 0
        self.last_extra_cost_usd = 0.0
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks

    def _chunk_component(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = render_component_text(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata=component_metadata(budget, component),
            token_count=count_tokens(text),
        )
