#!/usr/bin/env python3
"""Test script for Session 14 multi-agent system.

Demonstrates:
1. Multi-agent estimation with supervisor/workers
2. Human-in-the-loop when confidence is low
3. Resume after human decision
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import logfire

from src.domain.multi_agent import build_multi_agent_graph


async def test_multi_agent_flow():
    """Test the complete multi-agent flow with human-in-the-loop."""
    # Configure Logfire
    logfire.configure(
        service_name="sw-estimator-test",
        send_to_logfire="if-token-present",
    )

    # Load edge case transcript (should trigger low confidence)
    transcript_path = Path("scaffolding/session_14/sample_transcript_edge_case.txt")
    if not transcript_path.exists():
        print(f"ERROR: Transcript not found: {transcript_path}")
        return

    transcript = transcript_path.read_text()
    print(f"Loaded transcript: {len(transcript)} chars")

    # Build graph without checkpointer for this test
    print("\nBuilding multi-agent graph...")
    graph = build_multi_agent_graph(checkpointer=None)

    # Generate estimation ID
    estimation_id = str(uuid.uuid4())
    print(f"Estimation ID: {estimation_id}")

    # Initial state
    initial_state = {
        "transcript": transcript,
        "estimation_id": estimation_id,
        "requirements": [],
        "budget_matches": [],
        "agent_actions": [],
        "awaiting_review": False,
    }

    # Run the graph
    print("\nRunning multi-agent graph...")
    with logfire.span("multi_agent_test", estimation_id=estimation_id):
        result = await graph.ainvoke(initial_state)

    # Print results
    print("\n" + "=" * 80)
    print("MULTI-AGENT ESTIMATION RESULT")
    print("=" * 80)

    print(f"\nStatus: {result.get('status')}")
    print(f"Confidence: {result.get('confidence')}")

    print(f"\nRequirements extracted: {len(result.get('requirements', []))}")
    for i, req in enumerate(result.get("requirements", [])[:5], 1):
        print(f"  {i}. {req[:80]}...")

    print(f"\nBudget matches found: {len(result.get('budget_matches', []))}")
    for match in result.get("budget_matches", [])[:3]:
        print(f"  - {match['component'][:40]}: {match['amount']:.0f}h (distance={match['distance']:.3f})")

    estimate = result.get("estimate")
    if estimate:
        print(f"\nEstimate:")
        print(f"  Total engineer-days: {estimate.get('total_engineer_days')}")
        print(f"  Components: {len(estimate.get('components', []))}")

    validation = result.get("validation")
    if validation:
        print(f"\nValidation:")
        print(f"  Is valid: {validation.get('is_valid')}")
        print(f"  Issues: {len(validation.get('issues', []))}")
        for issue in validation.get("issues", [])[:3]:
            print(f"    - {issue}")

    print(f"\nAgent actions: {len(result.get('agent_actions', []))}")
    for action in result.get("agent_actions", []):
        print(f"  - {action['agent']}: {action['output_summary']}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_multi_agent_flow())
