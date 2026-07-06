"""Post-generation checks for grounded estimates (Session 9 + 11).

Three independent guards run after the LLM returns an :class:`Estimate`:

* :func:`verify_citations` — every cited ``chunk_id`` must correspond to a chunk
  that was actually retrieved. Fabricated ids are the classic grounding failure
  and trigger one corrective retry in the orchestrator. Returns a detailed
  :class:`CitationReport` with per-line status.
* :func:`validate_citations` — backward-compatible wrapper that extracts just the
  fabricated chunk_ids from the :class:`CitationReport`.
* :func:`check_coherence` — the ``insufficient`` confidence level has a strict
  shape (no numbers, an explanation present); a violation is a malformed
  response, not a valid estimate.
"""

from __future__ import annotations

import structlog

from src.generation.rag.schemas import (
    CitationLineStatus,
    CitationReport,
    Estimate,
    RetrievedChunk,
)

log = structlog.get_logger()


def verify_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> CitationReport:
    """Verify every task's cited chunk_ids against the retrieved context.

    For each task in the estimate, extracts the cited chunk_ids and checks them
    against the set of retrieved chunk ids. Builds a :class:`CitationReport`
    with per-line status: grounded (all citations valid), dangling_citation
    (at least one fabricated id), or ungrounded (task marked grounded=false).

    Parameters
    ----------
    estimate:
        The generated estimate to inspect.
    retrieved_chunks:
        The chunks the estimate was supposed to be grounded in.

    Returns
    -------
    CitationReport
        Per-line citation verification report with aggregate counts.
    """
    valid_ids = {chunk.id for chunk in retrieved_chunks}

    lines: list[CitationLineStatus] = []
    for module in estimate.modules:
        for task in module.tasks:
            cited = [source.chunk_id for source in task.sources]
            valid = [cid for cid in cited if cid in valid_ids]
            fabricated = [cid for cid in cited if cid not in valid_ids]

            if not task.grounded:
                status = "ungrounded"
            elif fabricated:
                status = "dangling_citation"
            else:
                status = "grounded"

            lines.append(
                CitationLineStatus(
                    module_name=module.name,
                    task_name=task.name,
                    grounded=task.grounded,
                    cited_chunk_ids=cited,
                    valid_chunk_ids=valid,
                    fabricated_chunk_ids=fabricated,
                    status=status,
                )
            )

    report = CitationReport(
        total_lines=len(lines),
        grounded_lines=sum(1 for line in lines if line.status == "grounded"),
        dangling_citation_lines=sum(
            1 for line in lines if line.status == "dangling_citation"
        ),
        ungrounded_lines=sum(1 for line in lines if line.status == "ungrounded"),
        lines=lines,
        all_valid=all(line.status != "dangling_citation" for line in lines),
    )

    log.info(
        "citation_verification",
        total=report.total_lines,
        grounded=report.grounded_lines,
        dangling=report.dangling_citation_lines,
        ungrounded=report.ungrounded_lines,
        all_valid=report.all_valid,
    )

    return report


def validate_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Return the cited chunk_ids that were never retrieved (fabricated).

    Backward-compatible wrapper around :func:`verify_citations` that extracts
    just the fabricated chunk_ids. An empty list means every citation is valid
    (including the edge case of an estimate that cites nothing at all).

    Parameters
    ----------
    estimate:
        The generated estimate to inspect.
    retrieved_chunks:
        The chunks the estimate was supposed to be grounded in.

    Returns
    -------
    list[int]
        Sorted, de-duplicated fabricated chunk_ids (empty if all valid).
    """
    report = verify_citations(estimate, retrieved_chunks)
    fabricated: set[int] = set()
    for line in report.lines:
        fabricated.update(line.fabricated_chunk_ids)
    return sorted(fabricated)


def check_coherence(estimate: Estimate) -> bool:
    """Return whether the estimate's confidence level matches its content.

    When ``confidence == "insufficient"``: both numeric totals must be ``None``,
    ``modules`` must be empty, and ``insufficient_context_explanation`` must be
    non-empty. Any other confidence level is always considered coherent here
    (the numeric checks belong to the schema/business rules, not to this guard).
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
