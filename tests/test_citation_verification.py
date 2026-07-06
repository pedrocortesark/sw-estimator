"""Unit tests for citation verification (Session 11).

Tests the per-line citation verification system:
- TaskItem grounded/sources consistency validator
- verify_citations() function
- CitationReport generation
"""

from __future__ import annotations

import pytest

from src.generation.rag.schemas import (
    CitationReport,
    Estimate,
    RetrievedChunk,
    SourceCitation,
    TaskItem,
    WorkModule,
)
from src.generation.rag.validation import verify_citations


def _make_chunk(id: int, budget_id: str = "BUD-2024-001") -> RetrievedChunk:
    """Helper to create a RetrievedChunk for testing."""
    return RetrievedChunk(
        id=id,
        content=f"Content for chunk {id}",
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.3,
        budget_id=budget_id,
    )


def _make_source(chunk_id: int, budget_id: str = "BUD-2024-001") -> SourceCitation:
    """Helper to create a SourceCitation for testing."""
    return SourceCitation(
        chunk_id=chunk_id,
        document_id=budget_id,
        evidence=f"Evidence from chunk {chunk_id}",
        relevance="primary",
        used_for="Task implementation",
    )


class TestTaskItemValidator:
    """Test TaskItem grounded/sources consistency."""

    def test_grounded_true_requires_sources(self):
        """grounded=True with empty sources should raise ValueError."""
        with pytest.raises(ValueError, match="grounded=True requires at least one source"):
            TaskItem(
                name="Test task",
                engineer_days=10,
                grounded=True,
                sources=[],
            )

    def test_grounded_false_requires_empty_sources(self):
        """grounded=False with non-empty sources should raise ValueError."""
        with pytest.raises(ValueError, match="grounded=False must have empty sources"):
            TaskItem(
                name="Test task",
                engineer_days=10,
                grounded=False,
                sources=[_make_source(1)],
            )

    def test_grounded_true_with_sources_valid(self):
        """grounded=True with sources should be valid."""
        task = TaskItem(
            name="Test task",
            engineer_days=10,
            grounded=True,
            sources=[_make_source(1)],
        )
        assert task.grounded is True
        assert len(task.sources) == 1

    def test_grounded_false_without_sources_valid(self):
        """grounded=False with empty sources should be valid."""
        task = TaskItem(
            name="Test task",
            engineer_days=10,
            grounded=False,
            sources=[],
        )
        assert task.grounded is False
        assert len(task.sources) == 0


class TestVerifyCitations:
    """Test verify_citations function."""

    def test_all_valid_citations(self):
        """All citations valid should return all_valid=True."""
        chunks = [_make_chunk(1), _make_chunk(2)]
        estimate = Estimate(
            total_engineer_days=20,
            modules=[
                WorkModule(
                    name="Module 1",
                    tasks=[
                        TaskItem(
                            name="Task 1",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(1)],
                        ),
                        TaskItem(
                            name="Task 2",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(2)],
                        ),
                    ],
                ),
            ],
            confidence="high",
            reasoning="All tasks grounded in sources.",
        )

        report = verify_citations(estimate, chunks)

        assert report.all_valid is True
        assert report.total_lines == 2
        assert report.grounded_lines == 2
        assert report.dangling_citation_lines == 0
        assert report.ungrounded_lines == 0

    def test_dangling_citation(self):
        """Citation with fabricated chunk_id should return all_valid=False."""
        chunks = [_make_chunk(1)]  # Only chunk 1 exists
        estimate = Estimate(
            total_engineer_days=20,
            modules=[
                WorkModule(
                    name="Module 1",
                    tasks=[
                        TaskItem(
                            name="Task 1",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(1)],  # Valid
                        ),
                        TaskItem(
                            name="Task 2",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(999)],  # Fabricated!
                        ),
                    ],
                ),
            ],
            confidence="medium",
            reasoning="One task has fabricated citation.",
        )

        report = verify_citations(estimate, chunks)

        assert report.all_valid is False
        assert report.total_lines == 2
        assert report.grounded_lines == 1
        assert report.dangling_citation_lines == 1
        assert report.ungrounded_lines == 0

        # Check the dangling line details
        dangling_line = next(line for line in report.lines if line.status == "dangling_citation")
        assert dangling_line.task_name == "Task 2"
        assert dangling_line.cited_chunk_ids == [999]
        assert dangling_line.valid_chunk_ids == []
        assert dangling_line.fabricated_chunk_ids == [999]

    def test_ungrounded_task(self):
        """Task with grounded=False should be counted as ungrounded."""
        chunks = [_make_chunk(1)]
        estimate = Estimate(
            total_engineer_days=10,
            modules=[
                WorkModule(
                    name="Module 1",
                    tasks=[
                        TaskItem(
                            name="Grounded task",
                            engineer_days=5,
                            grounded=True,
                            sources=[_make_source(1)],
                        ),
                        TaskItem(
                            name="Ungrounded task",
                            engineer_days=5,
                            grounded=False,
                            sources=[],
                        ),
                    ],
                ),
            ],
            confidence="low",
            reasoning="One task has no source support.",
        )

        report = verify_citations(estimate, chunks)

        assert report.all_valid is True  # No dangling citations
        assert report.total_lines == 2
        assert report.grounded_lines == 1
        assert report.dangling_citation_lines == 0
        assert report.ungrounded_lines == 1

    def test_mixed_statuses(self):
        """Mix of grounded, dangling, and ungrounded tasks."""
        chunks = [_make_chunk(1), _make_chunk(2)]
        estimate = Estimate(
            total_engineer_days=30,
            modules=[
                WorkModule(
                    name="Module 1",
                    tasks=[
                        TaskItem(
                            name="Grounded task",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(1)],
                        ),
                        TaskItem(
                            name="Dangling task",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(999)],  # Fabricated
                        ),
                        TaskItem(
                            name="Ungrounded task",
                            engineer_days=10,
                            grounded=False,
                            sources=[],
                        ),
                    ],
                ),
            ],
            confidence="low",
            reasoning="Mixed citation statuses.",
        )

        report = verify_citations(estimate, chunks)

        assert report.all_valid is False
        assert report.total_lines == 3
        assert report.grounded_lines == 1
        assert report.dangling_citation_lines == 1
        assert report.ungrounded_lines == 1


class TestCitationReport:
    """Test CitationReport structure."""

    def test_report_counts(self):
        """CitationReport should have correct aggregate counts."""
        chunks = [_make_chunk(1)]
        estimate = Estimate(
            total_engineer_days=20,
            modules=[
                WorkModule(
                    name="Module 1",
                    tasks=[
                        TaskItem(
                            name="Task 1",
                            engineer_days=10,
                            grounded=True,
                            sources=[_make_source(1)],
                        ),
                        TaskItem(
                            name="Task 2",
                            engineer_days=10,
                            grounded=False,
                            sources=[],
                        ),
                    ],
                ),
            ],
            confidence="medium",
            reasoning="Test report.",
        )

        report = verify_citations(estimate, chunks)

        assert isinstance(report, CitationReport)
        assert report.total_lines == 2
        assert report.grounded_lines == 1
        assert report.ungrounded_lines == 1
        assert report.dangling_citation_lines == 0
        assert len(report.lines) == 2

    def test_empty_estimate(self):
        """Empty estimate should return empty report."""
        chunks = [_make_chunk(1)]
        estimate = Estimate(
            total_engineer_days=None,
            modules=[],
            confidence="insufficient",
            reasoning="No context.",
            insufficient_context_explanation="No sources retrieved.",
        )

        report = verify_citations(estimate, chunks)

        assert report.all_valid is True
        assert report.total_lines == 0
        assert report.grounded_lines == 0
        assert report.dangling_citation_lines == 0
        assert report.ungrounded_lines == 0
        assert len(report.lines) == 0
