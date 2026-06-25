"""Unit tests for Reciprocal Rank Fusion (RRF).

RRF is a pure function with no I/O, so these tests are fast and deterministic.
They verify the fusion logic, not the database queries.
"""

from __future__ import annotations

import pytest

from src.generation.rag.retrieval.fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    """Test the RRF fusion algorithm."""

    def test_single_ranking(self):
        """Single ranking should preserve order with decreasing scores."""
        ranking = [10, 20, 30]
        result = reciprocal_rank_fusion([ranking])

        assert len(result) == 3
        assert result[0] == (10, 1.0 / (DEFAULT_RRF_K + 0))
        assert result[1] == (20, 1.0 / (DEFAULT_RRF_K + 1))
        assert result[2] == (30, 1.0 / (DEFAULT_RRF_K + 2))

    def test_two_rankings_with_overlap(self):
        """Documents appearing in both rankings should have higher scores."""
        ranking1 = [1, 2, 3]
        ranking2 = [2, 3, 4]
        result = reciprocal_rank_fusion([ranking1, ranking2])

        # Document 2 appears at position 1 in ranking1 and position 0 in ranking2
        # Score = 1/(60+1) + 1/(60+0) = 0.01639 + 0.01667 = 0.03306
        doc_2_score = next(score for doc_id, score in result if doc_id == 2)
        expected_score = 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 0)
        assert abs(doc_2_score - expected_score) < 1e-9

        # Document 2 should be first (highest score)
        assert result[0][0] == 2

    def test_empty_rankings(self):
        """Empty rankings should return empty result."""
        result = reciprocal_rank_fusion([[], []])
        assert result == []

    def test_custom_k_parameter(self):
        """Custom k should affect score calculation."""
        ranking = [1, 2]
        k = 10
        result = reciprocal_rank_fusion([ranking], k=k)

        assert result[0] == (1, 1.0 / (k + 0))
        assert result[1] == (2, 1.0 / (k + 1))

    def test_invalid_k_raises_error(self):
        """k <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="smoothing constant k must be positive"):
            reciprocal_rank_fusion([[1, 2]], k=0)

        with pytest.raises(ValueError, match="smoothing constant k must be positive"):
            reciprocal_rank_fusion([[1, 2]], k=-1)

    def test_duplicates_within_ranking_ignored(self):
        """Duplicate IDs within a single ranking should be ignored after first occurrence."""
        ranking = [1, 2, 1, 3, 2]  # 1 and 2 appear twice
        result = reciprocal_rank_fusion([ranking])

        # Should only have 3 unique documents
        assert len(result) == 3
        doc_ids = [doc_id for doc_id, _ in result]
        assert doc_ids == [1, 2, 3]

    def test_tie_breaking_by_id(self):
        """Documents with equal scores should be sorted by ID ascending."""
        # Create two rankings where some documents have the same score
        # Document 10 appears at position 0 in ranking1
        # Document 30 appears at position 0 in ranking2
        # Both have score 1/(60+0) = 0.01667
        ranking1 = [10, 20]
        ranking2 = [30, 40]
        result = reciprocal_rank_fusion([ranking1, ranking2])

        # Documents 10 and 30 have equal scores, so they should be sorted by ID
        # Documents 20 and 40 have equal scores, so they should be sorted by ID
        doc_ids = [doc_id for doc_id, _ in result]
        
        # Verify that within equal-score groups, IDs are sorted ascending
        # Group by score
        from collections import defaultdict
        score_groups = defaultdict(list)
        for doc_id, score in result:
            score_groups[score].append(doc_id)
        
        # Within each score group, IDs should be sorted ascending
        for score, ids in score_groups.items():
            assert ids == sorted(ids), f"IDs with score {score} not sorted: {ids}"

    def test_large_k_flattens_curve(self):
        """Large k should make scores more uniform."""
        ranking = [1, 2, 3, 4, 5]
        result_small_k = reciprocal_rank_fusion([ranking], k=1)
        result_large_k = reciprocal_rank_fusion([ranking], k=1000)

        # With small k, first position dominates
        score_diff_small = result_small_k[0][1] - result_small_k[-1][1]
        # With large k, scores are more uniform
        score_diff_large = result_large_k[0][1] - result_large_k[-1][1]

        assert score_diff_large < score_diff_small
