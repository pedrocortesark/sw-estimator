#!/usr/bin/env python3
"""Test script for the LangGraph estimation graph (Session 13).

Runs the graph on sample_transcript_complex.txt and prints the trace.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import logfire

from src.graph import build_graph


async def main():
    """Run the graph on sample_transcript_complex.txt."""
    # Configure Logfire
    logfire.configure(
        service_name="sw-estimator-test",
        send_to_logfire="if-token-present",
    )

    # Load transcript
    transcript_path = Path("exercises/session-12/sample_transcript_complex.txt")
    if not transcript_path.exists():
        print(f"ERROR: Transcript not found: {transcript_path}")
        return

    transcript = transcript_path.read_text()
    print(f"Loaded transcript: {len(transcript)} chars")

    # Build graph without checkpointer for now (will add in live session)
    print("\nBuilding graph...")
    graph = build_graph(checkpointer=None)

    # Generate estimation ID
    estimation_id = str(uuid.uuid4())
    print(f"Estimation ID: {estimation_id}")

    # Configure execution
    config = {"configurable": {"thread_id": estimation_id}}

    # Initial state
    initial_state = {
        "transcript": transcript,
        "requirements": [],
        "components": [],
        "budget_matches": [],
        "estimate": None,
        "status": None,
        "errors": [],
        "estimation_id": estimation_id,
    }

    # Execute graph
    print("\nExecuting graph...")
    with logfire.span("graph_execution", estimation_id=estimation_id):
        result = await graph.ainvoke(initial_state, config)

    # Print results
    print("\n" + "=" * 80)
    print("GRAPH EXECUTION COMPLETE")
    print("=" * 80)

    print(f"\nStatus: {result.get('status')}")
    print(f"Estimation ID: {result.get('estimation_id')}")

    print(f"\nRequirements extracted: {len(result.get('requirements', []))}")
    for i, req in enumerate(result.get("requirements", [])[:5], 1):
        print(f"  {i}. {req[:80]}...")

    print(f"\nComponents classified: {len(result.get('components', []))}")
    for comp in result.get("components", []):
        print(f"  - {comp['name']} ({comp['category']}): {len(comp['requirements'])} requirements")

    print(f"\nBudget matches found: {len(result.get('budget_matches', []))}")
    for match in result.get("budget_matches", []):
        print(f"  - {match['component']}: {match['budget_id']} ({match['amount']} EUR)")

    print(f"\nEstimate: {result.get('estimate')}")

    if result.get("errors"):
        print(f"\nErrors: {result.get('errors')}")

    print("\n" + "=" * 80)
    print("Trace available in Logfire (if token configured)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
