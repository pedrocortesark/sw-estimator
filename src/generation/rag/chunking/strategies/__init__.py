"""Chunking strategies for the Session 7 comparison.

Seven strategies behind the common :class:`~app.generation.rag.chunking.base.Chunker`
interface (the structural chunker lives in ``app.generation.rag.chunking.structural``).
Re-exported here so ``from src.generation.rag.chunking.strategies import *`` is a
one-line pre-flight check that every strategy imports cleanly.
"""

from src.generation.rag.chunking.strategies.contextual_retrieval import ContextualRetrievalChunker
from src.generation.rag.chunking.strategies.fixed_size import FixedSizeChunker
from src.generation.rag.chunking.strategies.hierarchical import HierarchicalChunker
from src.generation.rag.chunking.strategies.propositional import PropositionalChunker
from src.generation.rag.chunking.strategies.recursive import RecursiveChunker
from src.generation.rag.chunking.strategies.semantic import SemanticChunker
from src.generation.rag.chunking.strategies.sentence_window import SentenceWindowChunker

__all__ = [
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "SemanticChunker",
    "PropositionalChunker",
    "ContextualRetrievalChunker",
    "HierarchicalChunker",
]
