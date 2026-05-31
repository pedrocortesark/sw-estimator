"""Pydantic v2 schemas for the embedding pipeline."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sector universe — closed Literal derived from the seed data and domain model.
# Add new values here when the catalog expands.
# ---------------------------------------------------------------------------
Sector = Literal[
    "saas_b2b",
    "saas_b2c",
    "erp_integration",
    "ecommerce",
    "data_analytics",
    "fintech",
    "hr_tech",
    "iot",
    "other",
]

Currency = Literal["EUR", "USD", "GBP"]


# ---------------------------------------------------------------------------
# Budget building blocks
# ---------------------------------------------------------------------------


class BudgetComponent(BaseModel):
    """A single phase / work-package inside a budget."""

    name: str = Field(description="Phase or component name.")
    weeks: int | None = Field(default=None, ge=0, description="Duration in weeks.")
    amount: float = Field(description="Component cost in the budget currency.")


class Budget(BaseModel):
    """A complete, pre-cleaned budget document ready for chunking."""

    # --- Identity ---
    budget_id: str = Field(pattern=r"^BUDGET-\d{4}-\d{4}$")
    year: int = Field(ge=2000, le=2100)

    # --- Client metadata ---
    client_name: str
    client_code: str = Field(pattern=r"^CLI-\d{4}$")
    contact: str | None = None
    contact_email: str | None = None

    # --- Project summary ---
    project_summary: str = Field(
        description="1-2 sentence description of the project scope."
    )
    main_technology: str | None = Field(
        default=None,
        description="Primary tech stack or platform (e.g. 'Workday integration', 'React + FastAPI').",
    )
    sector: Sector = "other"

    # --- Financials ---
    currency: Currency = "EUR"
    total_amount: Annotated[float, Field(gt=0)]
    total_estimated_hours: int | None = Field(
        default=None,
        ge=0,
        description="Estimated effort in person-hours (optional at this stage).",
    )

    # --- Structure ---
    phases: list[BudgetComponent] = Field(min_length=1)
    notes: str | None = None

    @field_validator("currency", mode="before")
    @classmethod
    def normalise_currency(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Chunk models
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A text fragment ready for embedding."""

    chunk_id: str = Field(
        description="Unique identifier: '{budget_id}#{chunk_type}#{index}'."
    )
    text: str = Field(min_length=1, description="Human-readable text to embed.")
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Filterable fields stored alongside the vector: budget_id, "
            "chunk_type, client_code, sector, year, currency."
        ),
    )
    token_count: int = Field(ge=0, description="Token count for the text field.")


class EmbeddedChunk(Chunk):
    """A Chunk enriched with its dense vector representation."""

    embedding: list[float] = Field(
        description="Dense vector from text-embedding-3-small (1 536 dims)."
    )


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Body for POST /embeddings/ingest."""

    budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
    """Aggregate statistics returned alongside the embedded chunks."""

    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float = Field(
        description=(
            "Approximate cost at text-embedding-3-small pricing "
            "($0.02 / 1M tokens as of 2024)."
        )
    )


class IngestResponse(BaseModel):
    """Body returned by POST /embeddings/ingest."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
