"""Structural chunker for budget JSON documents.

Strategy: one BudgetComponent = one Chunk.

Each chunk text prepends the parent budget context so that the embedding
captures *who* the component belongs to — without this, a chunk like
"Authentication backend" loses its sector and client signal entirely.
This is an instance of the *contextual chunk header* pattern.

Token counting uses tiktoken with the cl100k_base encoding, which is the
tokenizer shared by text-embedding-3-small and text-embedding-3-large.
"""

from __future__ import annotations

import tiktoken

from src.embedding_pipeline.schemas import Budget, Chunk

# text-embedding-3-small uses cl100k_base (same as GPT-4 / ada-002).
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _render_chunk_text(budget: Budget, component_idx: int) -> str:
    component = budget.phases[component_idx]
    tech = ", ".join(component.tech_stack) if component.tech_stack else "N/A"
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.sector} | Year: {budget.year} | Main tech: {budget.main_technology or 'N/A'}]\n"
        f"\n"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {tech}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


class JSONStructuralChunker:
    """Splits a list of Budget documents into Chunks (one per BudgetComponent)."""

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for budget in budgets:
            for i, component in enumerate(budget.phases):
                text = _render_chunk_text(budget, i)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata={
                            "budget_id": budget.budget_id,
                            "component_id": component.component_id,
                            "client_sector": budget.sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "complexity": component.complexity,
                            "estimated_hours": component.estimated_hours,
                        },
                        token_count=_count_tokens(text),
                    )
                )

        return chunks
