#!/usr/bin/env python3
"""Test script for the Session 12 estimation agent.

Run with: uv run python scripts/test_agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from src.core.config import get_settings
from src.generation.agentic.agent_loop import run_estimation_agent


async def main():
    """Run the agent on the complex transcript and print the trace."""
    settings = get_settings()
    
    if not settings.openai_api_key:
        print("❌ No OPENAI_API_KEY configured in .env")
        return
    
    # Load the complex transcript
    transcript_path = Path("exercises/session-12/sample_transcript_complex.txt")
    if not transcript_path.exists():
        print(f"❌ Transcript not found: {transcript_path}")
        return
    
    transcript = transcript_path.read_text()
    print(f"📄 Loaded transcript: {len(transcript)} chars")
    print(f"🤖 Model: {settings.openai_model}")
    print(f"⚙️  Reasoning effort: medium")
    print("=" * 80)
    
    # Create OpenAI client
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Run the agent
    print("\n🚀 Running estimation agent...\n")
    result = await run_estimation_agent(
        transcript=transcript,
        client=client,
        model=settings.openai_model,
        reasoning_effort="medium",
        max_iterations=10,
    )
    
    # Print the trace
    print("\n" + "=" * 80)
    print("📋 AGENT TRACE")
    print("=" * 80)
    print(result.trace.render())
    
    # Print the final estimate
    print("\n" + "=" * 80)
    print("💰 FINAL ESTIMATE")
    print("=" * 80)
    
    if result.estimate:
        print(f"\nTotal hours: {result.estimate.total_hours}")
        print(f"Confidence: {result.estimate.confidence}")
        print(f"\nComponents ({len(result.estimate.components)}):")
        for comp in result.estimate.components:
            print(f"  • {comp.name}: {comp.estimated_hours}h")
            print(f"    {comp.rationale}")
        
        if result.estimate.assumptions:
            print(f"\nAssumptions ({len(result.estimate.assumptions)}):")
            for assumption in result.estimate.assumptions:
                print(f"  • {assumption}")
    else:
        print("❌ No estimate produced")
        print(f"Stopped reason: {result.stopped_reason}")
    
    print(f"\n📊 Iterations: {result.iterations}")
    print(f"🛑 Stopped reason: {result.stopped_reason}")


if __name__ == "__main__":
    asyncio.run(main())
