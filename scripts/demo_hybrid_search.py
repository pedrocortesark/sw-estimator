"""Demo script for hybrid search with 4 configurations.

This script demonstrates the 4 retrieval modes:
1. Vector search only (baseline)
2. Vector search + reranking
3. Hybrid search (vector + lexical + RRF)
4. Hybrid search + reranking

Run with: uv run python scripts/demo_hybrid_search.py
"""

from __future__ import annotations

import asyncio
import time

from openai import OpenAI

from src.core.config import get_settings
from src.generation.rag.retrieval.pipeline import retrieve
from src.rag.embedding.embedder import OpenAIEmbedder


async def demo_hybrid_search():
    """Demonstrate the 4 retrieval configurations."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(client=client)

    # Test queries
    queries = [
        "OAuth authentication banking API",
        "e-commerce checkout payment",
        "real-time payments fraud detection",
    ]

    print("=" * 80)
    print("HYBRID SEARCH DEMONSTRATION")
    print("=" * 80)
    print()

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)

        query_embedding = embedder.embed_one(query)

        # Configuration 1: Vector only
        print("\n1. Vector Search Only (baseline)")
        start = time.perf_counter()
        result = await retrieve(
            query_embedding=query_embedding,
            query_text=query,
            search_mode="vector",
            rerank=False,
            top_k=5,
        )
        elapsed = time.perf_counter() - start
        print(f"   Results: {len(result.chunks)} chunks in {elapsed*1000:.1f}ms")
        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}, sector={chunk.sector}")

        # Configuration 2: Vector + Reranking
        print("\n2. Vector Search + Reranking")
        start = time.perf_counter()
        result = await retrieve(
            query_embedding=query_embedding,
            query_text=query,
            search_mode="vector",
            rerank=True,
            recall_k=10,
            rerank_top_n=5,
        )
        elapsed = time.perf_counter() - start
        print(f"   Results: {len(result.chunks)} chunks in {elapsed*1000:.1f}ms")
        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}, sector={chunk.sector}")

        # Configuration 3: Hybrid (vector + lexical + RRF)
        print("\n3. Hybrid Search (Vector + Lexical + RRF)")
        start = time.perf_counter()
        result = await retrieve(
            query_embedding=query_embedding,
            query_text=query,
            search_mode="hybrid",
            rerank=False,
            top_k=5,
        )
        elapsed = time.perf_counter() - start
        print(f"   Results: {len(result.chunks)} chunks in {elapsed*1000:.1f}ms")
        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}, sector={chunk.sector}")

        # Configuration 4: Hybrid + Reranking
        print("\n4. Hybrid Search + Reranking")
        start = time.perf_counter()
        result = await retrieve(
            query_embedding=query_embedding,
            query_text=query,
            search_mode="hybrid",
            rerank=True,
            recall_k=10,
            rerank_top_n=5,
        )
        elapsed = time.perf_counter() - start
        print(f"   Results: {len(result.chunks)} chunks in {elapsed*1000:.1f}ms")
        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}, sector={chunk.sector}")

        print("\n" + "=" * 80)
        print()


if __name__ == "__main__":
    asyncio.run(demo_hybrid_search())
