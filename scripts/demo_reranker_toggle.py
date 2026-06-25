"""Demo script showing reranker can be enabled/disabled without code changes.

This demonstrates the 3 ways to control reranking:
1. Via API request parameter (rerank=true/false)
2. Via environment variable (RERANKER_ENABLED=true/false)
3. Via default settings (reranker_enabled=False)

Run with: uv run python scripts/demo_reranker_toggle.py
"""

from __future__ import annotations

import asyncio
import os

from openai import OpenAI

from src.core.config import get_settings
from src.generation.rag.retrieval.pipeline import retrieve
from src.rag.embedding.embedder import OpenAIEmbedder


async def demo_reranker_toggle():
    """Demonstrate reranker can be toggled without code changes."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(client=client)

    query = "OAuth authentication banking API"
    print("=" * 80)
    print("RERANKER TOGGLE DEMONSTRATION")
    print("=" * 80)
    print(f"\nQuery: {query}")
    print(f"Default reranker_enabled setting: {settings.reranker_enabled}")
    print()

    query_embedding = embedder.embed_one(query)

    # Method 1: Explicit rerank=False in request
    print("1. Explicit rerank=False in request")
    print("-" * 80)
    result = await retrieve(
        query_embedding=query_embedding,
        query_text=query,
        search_mode="vector",
        rerank=False,  # Explicitly disabled
        top_k=5,
    )
    print(f"   Results: {len(result.chunks)} chunks")
    print(f"   Rerank used: No (explicit parameter)")
    for i, chunk in enumerate(result.chunks[:3], 1):
        print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}")
    print()

    # Method 2: Explicit rerank=True in request
    print("2. Explicit rerank=True in request")
    print("-" * 80)
    result = await retrieve(
        query_embedding=query_embedding,
        query_text=query,
        search_mode="vector",
        rerank=True,  # Explicitly enabled
        recall_k=10,
        rerank_top_n=5,
    )
    print(f"   Results: {len(result.chunks)} chunks")
    print(f"   Rerank used: Yes (explicit parameter)")
    for i, chunk in enumerate(result.chunks[:3], 1):
        print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}")
    print()

    # Method 3: Use settings default (None in request)
    print("3. Use settings default (rerank=None in request)")
    print("-" * 80)
    print(f"   Current RERANKER_ENABLED env var: {os.getenv('RERANKER_ENABLED', 'not set (uses default False)')}")
    result = await retrieve(
        query_embedding=query_embedding,
        query_text=query,
        search_mode="vector",
        rerank=settings.reranker_enabled,  # Use settings default
        top_k=5,
    )
    print(f"   Results: {len(result.chunks)} chunks")
    print(f"   Rerank used: {settings.reranker_enabled} (from settings)")
    for i, chunk in enumerate(result.chunks[:3], 1):
        print(f"   {i}. id={chunk.id}, distance={chunk.distance:.4f}")
    print()

    print("=" * 80)
    print("\nTo enable reranking by default without code changes:")
    print("  export RERANKER_ENABLED=true")
    print("  # or add to .env file:")
    print("  RERANKER_ENABLED=true")
    print()
    print("To disable reranking by default:")
    print("  export RERANKER_ENABLED=false")
    print("  # or add to .env file:")
    print("  RERANKER_ENABLED=false")
    print()
    print("To override per-request (regardless of default):")
    print('  POST /v1/retrieval/search')
    print('  {"query_text": "...", "rerank": true, ...}')
    print('  {"query_text": "...", "rerank": false, ...}')
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_reranker_toggle())
