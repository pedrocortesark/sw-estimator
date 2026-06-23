#!/usr/bin/env python3
"""Session 09 pre-work — manual trace of examples/transcripts/02_ambiguous.txt.

Step 1: embed the FULL transcript with the same module the search path uses
        (OpenAIEmbedder.embed_one → text-embedding-3-small). There is no HTTP
        endpoint to embed free text in S08, so we call the module directly.
Step 2: POST the transcript to /search (k=5) and dump the raw response.

Run:
    docker compose up -d postgres
    uv run alembic upgrade head
    # seed once: ESTIMATOR_BASE_URL=http://127.0.0.1:8010 uv run python scripts/query_examples.py
    ESTIMATOR_BASE_URL=http://127.0.0.1:8010 uv run python scripts/trace_02_ambiguous.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import httpx

from src.dependencies import get_embedder

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = ROOT / "examples" / "transcripts" / "02_ambiguous.txt"
BASE_URL = os.environ.get("ESTIMATOR_BASE_URL", "http://127.0.0.1:8010")
K = 5

text = TRANSCRIPT.read_text(encoding="utf-8")

# --- Step 1: embed the full transcript -------------------------------------
embedder = get_embedder()
assert embedder is not None, "No OPENAI_API_KEY configured"
vector = embedder.embed_one(text)
norm = math.sqrt(sum(c * c for c in vector))

print("=== STEP 1 — embed full transcript ===")
print(f"transcript_chars : {len(text)}")
print(f"model            : text-embedding-3-small")
print(f"dimensions       : {len(vector)}")
print(f"L2_norm          : {norm:.6f}")
print(f"first_component  : {vector[0]:.6f}")
print(f"last_component   : {vector[-1]:.6f}")

# --- Step 2: semantic search via the HTTP endpoint -------------------------
print("\n=== STEP 2 — POST /search (k=5) ===")
resp = httpx.post(f"{BASE_URL}/search", json={"query": text, "k": K}, timeout=30)
resp.raise_for_status()
body = resp.json()
# Trim long content for the raw dump but keep distances + metadata intact.
for hit in body["results"]:
    hit["content"] = " ".join(hit["content"].split())
print(json.dumps(body, ensure_ascii=False, indent=2))
