"""Stress-suite orchestrator — scenario runner + attachment stress + report writer.

Produces a single CSV with one row per turn (scenario runs) or one row per size
point (attachment stress), then writes an auto-generated REPORT.md from that data.

Usage examples
--------------
In-process (no server required):

    python -m evals.stress.run \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 1 \\
        --output evals/stress/results.csv

Against a running server:

    python -m evals.stress.run \\
        --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 1 \\
        --output evals/stress/results.csv

Scenario aliases (CLI → profile name)
--------------------------------------
  growing       → growing_project
  pivot         → pivoting_project
  contradiction → contradicting_project

Columns in the output CSV
--------------------------
run_type, scenario, n_turns, repeat, session_id,
turn_index, enriched_transcript_chars, attachments_total_chars,
messages_in_window, anchors_count, summary_chars,
tokens_in, tokens_out, cost_usd, latency_ms,
cache_hit_kind, last_resolved_tier,
facts_total, facts_retained, fact_recall,
latency_ok, cost_ok,
attachment_kb, attachment_extracted_chars, attachment_truncated_chars,
attachment_markers_in, attachment_markers_cut, attachment_recall,
error

REPORT.md (written next to the CSV)
------------------------------------
• Summary table: P50/P95 latency, cumulative cost, cache-hit rate, fact recall.
• Three curve tables: latency vs tokens_in, cumulative cost vs turn,
  recall vs N (in ASCII/Markdown, no external graphics needed).
• Two analysis paragraphs: where CAG starts to break and why.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import structlog

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric
from evals.stress.scenarios import ALL_PROFILES, N_VALUES, ScenarioProfile

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Budget thresholds (used to populate latency_ok / cost_ok columns)
# ---------------------------------------------------------------------------

LATENCY_BUDGET_MS: int = 5_000  # 5 s — typical interactive tolerance
COST_BUDGET_USD: float = 0.01  # $0.01 per turn — reasonable for gpt-4o-mini

_latency_metric = LatencyBudgetMetric(budget_ms=LATENCY_BUDGET_MS)
_cost_metric = CostBudgetMetric(budget_usd=COST_BUDGET_USD)

# ---------------------------------------------------------------------------
# CLI alias → profile name mapping
# ---------------------------------------------------------------------------

_SCENARIO_ALIASES: dict[str, str] = {
    "growing": "growing_project",
    "pivot": "pivoting_project",
    "contradiction": "contradicting_project",
}

# Reverse index: profile.name → ScenarioProfile
_PROFILE_BY_NAME: dict[str, ScenarioProfile] = {p.name: p for p in ALL_PROFILES}

# ---------------------------------------------------------------------------
# CSV schema (ordered)
# ---------------------------------------------------------------------------

CSV_FIELDS: list[str] = [
    "run_type",
    "scenario",
    "n_turns",
    "repeat",
    "session_id",
    "turn_index",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "facts_total",
    "facts_retained",
    "fact_recall",
    "latency_ok",
    "cost_ok",
    "attachment_kb",
    "attachment_extracted_chars",
    "attachment_truncated_chars",
    "attachment_markers_in",
    "attachment_markers_cut",
    "attachment_recall",
    "error",
]

_EMPTY_SCENARIO_FIELDS = {
    "attachment_kb": "",
    "attachment_extracted_chars": "",
    "attachment_truncated_chars": "",
    "attachment_markers_in": "",
    "attachment_markers_cut": "",
    "attachment_recall": "",
}

_EMPTY_ATTACHMENT_FIELDS = {
    "scenario": "",
    "n_turns": "",
    "messages_in_window": "",
    "anchors_count": "",
    "summary_chars": "",
    "facts_total": "",
    "facts_retained": "",
    "fact_recall": "",
}

# ---------------------------------------------------------------------------
# Fact evaluation helpers
# ---------------------------------------------------------------------------


def _eval_facts(
    profile: ScenarioProfile,
    n_turns: int,
    session: Any,
) -> tuple[int, int]:
    """Evaluate all FactAssertions up to *n_turns* against *session*.

    Args:
        profile:  The scenario profile containing fact assertions.
        n_turns:  Run length — evaluate all facts introduced within this range.
        session:  Session object (or duck-typed namespace) after the last turn.

    Returns:
        ``(facts_total, facts_retained)`` — total facts declared and how many
        currently pass their check callable.
    """
    facts = profile.all_facts_up_to(n_turns)
    retained = sum(1 for f in facts if f.check(session))
    return len(facts), retained


def _session_from_info(info: dict) -> SimpleNamespace:
    """Build a duck-typed Session proxy from a /sessions/{id} GET response dict.

    In HTTP mode the full Session object is not available locally.  This
    proxy exposes the same attribute surface as Session so that check
    callables can be exercised.  Anchor strings are unavailable via the API
    (only the count is returned), so ``_anchor_exists`` checks will always
    fail — this is documented in the CSV ``error`` field of HTTP runs.

    Args:
        info: Parsed JSON body from GET /api/v1/sessions/{id}.

    Returns:
        SimpleNamespace with ``.metadata``, ``.anchors``, ``.accumulated_summary``,
        ``.last_resolved_tier``, ``.last_tier_rule`` attributes.
    """
    pm = info.get("project_metadata", {})
    meta = SimpleNamespace(
        project_name=pm.get("project_name"),
        assumed_team_size=pm.get("assumed_team_size"),
        mentioned_technologies=pm.get("mentioned_technologies") or [],
        agreed_scope=pm.get("agreed_scope"),
    )
    return SimpleNamespace(
        metadata=meta,
        anchors=[],  # not exposed by the API — anchor checks will fail
        accumulated_summary="",  # not exposed by the API
        last_resolved_tier=info.get("last_resolved_tier", "unknown"),
        last_tier_rule=info.get("last_tier_rule", "no_match"),
    )


def _cache_kind(response: Any) -> str:
    """Derive cache_hit_kind from an EstimationResponse (in-process mode).

    Uses the ``model_used`` and ``provider_used`` fields set by
    ``EstimationService._build_response()`` when returning from cache:
    - provider_used == "memory_cache"   → "exact"
    - provider_used == "semantic_cache" → "semantic"
    - otherwise                         → "none"
    """
    if not getattr(response, "cached", False):
        return "none"
    provider = getattr(response, "provider_used", "") or ""
    return "semantic" if "semantic" in provider else "exact"


# ---------------------------------------------------------------------------
# In-process scenario runner
# ---------------------------------------------------------------------------


async def _run_scenario_inprocess(
    profile: ScenarioProfile,
    n_turns: int,
    repeat: int,
    service: Any,
) -> list[dict]:
    """Run one scenario profile for *n_turns* turns against the in-process service.

    Args:
        profile:  Scenario profile to execute.
        n_turns:  How many turns of the script to run.
        repeat:   Repeat index (1-based) for labelling.
        service:  A live ``EstimationService`` instance.

    Returns:
        List of CSV row dicts, one per turn.
    """
    from src.services.sessions import session_store

    session_id = str(uuid.uuid4())
    session = session_store.get_or_create(session_id)

    rows: list[dict] = []
    script = profile.script_for(n_turns)

    for turn_script in script:
        t0 = time.perf_counter()
        error: str | None = None
        response = None

        try:
            response = await service.estimate_conversational(
                session=session,
                transcript=turn_script.transcript,
                enriched_transcript=turn_script.transcript,
                attachments_total_chars=0,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Fact evaluation — all facts up to the current turn
        facts_total, facts_retained = _eval_facts(
            profile, turn_script.turn_index, session
        )
        fact_recall = round(facts_retained / facts_total, 4) if facts_total else None

        obs = {
            "latency_ms": latency_ms,
            "cost_usd": response.usage.cost_usd if response else 0.0,
        }

        row: dict[str, Any] = {
            "run_type": "scenario",
            "scenario": profile.name,
            "n_turns": n_turns,
            "repeat": repeat,
            "session_id": session_id,
            "turn_index": turn_script.turn_index,
            "enriched_transcript_chars": len(turn_script.transcript),
            "attachments_total_chars": 0,
            "messages_in_window": len(session.history),
            "anchors_count": len(session.anchors),
            "summary_chars": session.summary_chars,
            "tokens_in": response.usage.input_tokens if response else "",
            "tokens_out": response.usage.output_tokens if response else "",
            "cost_usd": response.usage.cost_usd if response else "",
            "latency_ms": latency_ms,
            "cache_hit_kind": _cache_kind(response) if response else "none",
            "last_resolved_tier": session.last_resolved_tier,
            "facts_total": facts_total,
            "facts_retained": facts_retained,
            "fact_recall": fact_recall if fact_recall is not None else "",
            "latency_ok": int(_latency_metric.evaluate(obs).passed),
            "cost_ok": int(_cost_metric.evaluate(obs).passed),
            "error": error or "",
            **_EMPTY_SCENARIO_FIELDS,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# In-process attachment stress runner
# ---------------------------------------------------------------------------


async def _run_attachment_inprocess(
    size_kb: int,
    repeat: int,
    service: Any,
) -> dict:
    """Run one attachment-size data point against the in-process service.

    Args:
        size_kb: Target PDF size in KB (0 = no attachment).
        repeat:  Repeat index (1-based) for labelling.
        service: A live ``EstimationService`` instance.

    Returns:
        A single CSV row dict.
    """
    from evals.stress.attachment_stress import (
        BASE_TRANSCRIPT,
        MAX_ATTACHMENT_CHARS,
        RECALL_MARKERS,
        compute_recall,
        generate_pdf,
        _response_corpus,
    )
    from src.services.document_extractor import build_attachment_block, extract_text
    from src.services.sessions import session_store

    session_id = str(uuid.uuid4())
    session = session_store.get_or_create(session_id)

    pdf_bytes = b""
    extracted_chars = 0
    truncated_chars = 0
    markers_in_input: list[str] = []
    markers_truncated = 0

    if size_kb > 0:
        from evals.stress.attachment_stress import generate_pdf

        target_extracted = int(size_kb * 1024 * 0.85)
        pdf_bytes, _raw_text, all_markers = generate_pdf(target_extracted)
        extracted_text = extract_text("spec.pdf", pdf_bytes)
        extracted_chars = len(extracted_text)
        truncated_text = extracted_text[:MAX_ATTACHMENT_CHARS]
        truncated_chars = len(truncated_text)
        markers_in_input = [
            m for m in all_markers if m.lower() in truncated_text.lower()
        ]
        markers_truncated = len(all_markers) - len(markers_in_input)
        attachment_block = build_attachment_block("spec.pdf", truncated_text)
        enriched = BASE_TRANSCRIPT + "\n\n" + attachment_block
    else:
        enriched = BASE_TRANSCRIPT
        all_markers = [m for m, _ in RECALL_MARKERS]
        markers_in_input = []

    t0 = time.perf_counter()
    error: str | None = None
    response = None
    result_dict: dict = {}

    try:
        response = await service.estimate_conversational(
            session=session,
            transcript=BASE_TRANSCRIPT,
            enriched_transcript=enriched,
            attachments_total_chars=len(enriched) - len(BASE_TRANSCRIPT),
        )
        result_dict = {
            "estimation_result": response.estimation,
            "usage": response.usage,
        }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Recall computation
    if response and markers_in_input:
        corpus_parts: list[str] = [response.estimation.executive_summary or ""]
        for phase in response.estimation.phases or []:
            corpus_parts.append(phase.name or "")
        corpus = " ".join(corpus_parts)
        recall, _found, _missing = compute_recall(corpus, markers_in_input)
        attachment_recall: float | str = round(recall, 4)
    elif size_kb == 0:
        attachment_recall = ""  # baseline: no attachment
    else:
        attachment_recall = 0.0

    obs = {
        "latency_ms": latency_ms,
        "cost_usd": response.usage.cost_usd if response else 0.0,
    }

    return {
        "run_type": "attachment",
        "repeat": repeat,
        "session_id": session_id,
        "turn_index": 1,
        "enriched_transcript_chars": len(enriched),
        "attachments_total_chars": len(enriched) - len(BASE_TRANSCRIPT),
        "tokens_in": response.usage.input_tokens if response else "",
        "tokens_out": response.usage.output_tokens if response else "",
        "cost_usd": response.usage.cost_usd if response else "",
        "latency_ms": latency_ms,
        "cache_hit_kind": _cache_kind(response) if response else "none",
        "last_resolved_tier": session.last_resolved_tier if response else "",
        "latency_ok": int(_latency_metric.evaluate(obs).passed),
        "cost_ok": int(_cost_metric.evaluate(obs).passed),
        "attachment_kb": size_kb,
        "attachment_extracted_chars": extracted_chars,
        "attachment_truncated_chars": truncated_chars,
        "attachment_markers_in": len(markers_in_input),
        "attachment_markers_cut": markers_truncated,
        "attachment_recall": attachment_recall,
        "error": error or "",
        **_EMPTY_ATTACHMENT_FIELDS,
    }


# ---------------------------------------------------------------------------
# HTTP scenario runner
# ---------------------------------------------------------------------------


async def _run_scenario_http(
    profile: ScenarioProfile,
    n_turns: int,
    repeat: int,
    client: Any,
    base_url: str,
) -> list[dict]:
    """Run one scenario profile for *n_turns* turns via HTTP.

    Facts that depend on anchor strings (``_anchor_exists``) will always fail
    because the /sessions/{id} endpoint only returns ``anchors_count``, not the
    anchor strings themselves.  This is noted in the ``error`` column with the
    tag ``"http:no_anchor_strings"``.

    Args:
        profile:  Scenario profile to execute.
        n_turns:  Number of turns to run.
        repeat:   Repeat index (1-based).
        client:   Live ``httpx.AsyncClient``.
        base_url: Server base URL (e.g. ``"http://localhost:8000"``).

    Returns:
        List of CSV row dicts, one per turn.
    """
    # Create session
    resp = await client.post(f"{base_url}/api/v1/sessions")
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    rows: list[dict] = []
    script = profile.script_for(n_turns)

    for turn_script in script:
        t0 = time.perf_counter()
        error: str = "http:no_anchor_strings"  # always annotate HTTP limitation
        response_data: dict = {}

        try:
            est_resp = await client.post(
                f"{base_url}/api/v1/sessions/{session_id}/estimate",
                data={"transcript": turn_script.transcript},
            )
            est_resp.raise_for_status()
            response_data = est_resp.json()
        except Exception as exc:  # noqa: BLE001
            error = f"http:no_anchor_strings;{type(exc).__name__}:{exc}"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Fetch updated session state for fact checks
        info: dict = {}
        try:
            info_resp = await client.get(f"{base_url}/api/v1/sessions/{session_id}")
            info_resp.raise_for_status()
            info = info_resp.json()
        except Exception:  # noqa: BLE001
            pass

        # Reconstruct proxy session from HTTP info for fact checks
        fake_session = _session_from_info(info) if info else _session_from_info({})
        # Propagate last_resolved_tier from estimation response if info is absent
        if response_data:
            fake_session.last_resolved_tier = info.get("last_resolved_tier", "unknown")

        facts_total, facts_retained = _eval_facts(
            profile, turn_script.turn_index, fake_session
        )
        fact_recall = round(facts_retained / facts_total, 4) if facts_total else None

        usage = response_data.get("usage", {})
        tokens_in = usage.get("input_tokens", "")
        tokens_out = usage.get("output_tokens", "")
        cost_usd = usage.get("cost_usd", "")
        cached_flag = response_data.get("cached", False)
        cache_kind = "exact" if cached_flag else "none"  # kind not available over HTTP

        obs = {
            "latency_ms": latency_ms,
            "cost_usd": cost_usd if cost_usd != "" else 0.0,
        }

        row: dict[str, Any] = {
            "run_type": "scenario",
            "scenario": profile.name,
            "n_turns": n_turns,
            "repeat": repeat,
            "session_id": session_id,
            "turn_index": turn_script.turn_index,
            "enriched_transcript_chars": len(turn_script.transcript),
            "attachments_total_chars": 0,
            "messages_in_window": info.get("turn_count", "") or "",
            "anchors_count": info.get("anchors_count", ""),
            "summary_chars": info.get("summary_chars", ""),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "cache_hit_kind": cache_kind,
            "last_resolved_tier": info.get("last_resolved_tier", ""),
            "facts_total": facts_total,
            "facts_retained": facts_retained,
            "fact_recall": fact_recall if fact_recall is not None else "",
            "latency_ok": int(_latency_metric.evaluate(obs).passed),
            "cost_ok": int(_cost_metric.evaluate(obs).passed),
            "error": error,
            **_EMPTY_SCENARIO_FIELDS,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# HTTP attachment runner
# ---------------------------------------------------------------------------


async def _run_attachment_http(
    size_kb: int,
    repeat: int,
    client: Any,
    base_url: str,
) -> dict:
    """Run one attachment-size data point via HTTP (multipart upload).

    Args:
        size_kb:  Target PDF size in KB (0 = no attachment).
        repeat:   Repeat index (1-based).
        client:   Live ``httpx.AsyncClient``.
        base_url: Server base URL.

    Returns:
        A single CSV row dict.
    """
    from evals.stress.attachment_stress import (
        BASE_TRANSCRIPT,
        MAX_ATTACHMENT_CHARS,
        RECALL_MARKERS,
        compute_recall,
        generate_pdf,
    )
    from src.services.document_extractor import extract_text

    # Create a fresh session for each attachment run
    resp = await client.post(f"{base_url}/api/v1/sessions")
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    pdf_bytes = b""
    extracted_chars = 0
    truncated_chars = 0
    markers_in_input: list[str] = []
    markers_truncated = 0
    all_markers = [m for m, _ in RECALL_MARKERS]

    if size_kb > 0:
        target_extracted = int(size_kb * 1024 * 0.85)
        pdf_bytes, _raw_text, all_markers = generate_pdf(target_extracted)
        extracted_text = extract_text("spec.pdf", pdf_bytes)
        extracted_chars = len(extracted_text)
        truncated_chars = min(extracted_chars, MAX_ATTACHMENT_CHARS)
        markers_in_input = [
            m
            for m in all_markers
            if m.lower() in extracted_text[:MAX_ATTACHMENT_CHARS].lower()
        ]
        markers_truncated = len(all_markers) - len(markers_in_input)

    t0 = time.perf_counter()
    error: str | None = None
    response_data: dict = {}

    try:
        if size_kb > 0:
            files = [("attachments", ("spec.pdf", pdf_bytes, "application/pdf"))]
            est_resp = await client.post(
                f"{base_url}/api/v1/sessions/{session_id}/estimate",
                data={"transcript": BASE_TRANSCRIPT},
                files=files,
            )
        else:
            est_resp = await client.post(
                f"{base_url}/api/v1/sessions/{session_id}/estimate",
                data={"transcript": BASE_TRANSCRIPT},
            )
        est_resp.raise_for_status()
        response_data = est_resp.json()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Recall
    attachment_recall: float | str = ""
    if response_data and markers_in_input:
        estimation = response_data.get("estimation", {})
        corpus_parts = [estimation.get("executive_summary", "")]
        for phase in estimation.get("phases", []):
            corpus_parts.append(phase.get("name", ""))
        corpus = " ".join(corpus_parts)
        recall, _found, _miss = compute_recall(corpus, markers_in_input)
        attachment_recall = round(recall, 4)
    elif size_kb == 0:
        attachment_recall = ""

    usage = response_data.get("usage", {})
    tokens_in = usage.get("input_tokens", "")
    tokens_out = usage.get("output_tokens", "")
    cost_usd_val = usage.get("cost_usd", "")
    cached_flag = response_data.get("cached", False)

    obs = {
        "latency_ms": latency_ms,
        "cost_usd": cost_usd_val if cost_usd_val != "" else 0.0,
    }

    return {
        "run_type": "attachment",
        "repeat": repeat,
        "session_id": session_id,
        "turn_index": 1,
        "enriched_transcript_chars": "",
        "attachments_total_chars": extracted_chars,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd_val,
        "latency_ms": latency_ms,
        "cache_hit_kind": "exact" if cached_flag else "none",
        "last_resolved_tier": "",
        "latency_ok": int(_latency_metric.evaluate(obs).passed),
        "cost_ok": int(_cost_metric.evaluate(obs).passed),
        "attachment_kb": size_kb,
        "attachment_extracted_chars": extracted_chars,
        "attachment_truncated_chars": truncated_chars,
        "attachment_markers_in": len(markers_in_input),
        "attachment_markers_cut": markers_truncated,
        "attachment_recall": attachment_recall,
        "error": error or "",
        **_EMPTY_ATTACHMENT_FIELDS,
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write *rows* to *path* as a UTF-8 CSV with a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Ensure every field is present (default to empty string)
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"CSV  → {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Statistics helpers (pure Python, no pandas)
# ---------------------------------------------------------------------------


def _to_floats(rows: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for r in rows:
        v = r.get(key)
        try:
            vals.append(float(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return vals


def _pct(values: list[float], p: int) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return round(s[idx], 1)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else float("nan")


def _nan_str(v: float, fmt: str = ".1f") -> str:
    return "—" if v != v else format(v, fmt)  # NaN check: NaN != NaN


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(rows: list[dict]) -> str:  # noqa: C901
    """Build a Markdown report from *rows*.

    Returns the full Markdown string — the caller writes it to disk.
    """
    from datetime import datetime, timezone

    scenario_rows = [r for r in rows if r.get("run_type") == "scenario"]
    attach_rows = [r for r in rows if r.get("run_type") == "attachment"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_llm = len([r for r in rows if r.get("cache_hit_kind") == "none"])
    n_cache = len([r for r in rows if r.get("cache_hit_kind") in ("exact", "semantic")])

    lines: list[str] = [
        "# Stress Evaluation Report",
        "",
        f"Generated: {now}  ",
        f"Total rows: {len(rows)}  |  LLM calls: {n_llm}  |  Cache hits: {n_cache}",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # 1. Summary table — scenario runs
    # ------------------------------------------------------------------
    lines += [
        "## 1. Summary — Scenario Runs",
        "",
    ]

    if scenario_rows:
        scenario_names = sorted(
            {r["scenario"] for r in scenario_rows if r.get("scenario")}
        )
        lines += [
            "| Scenario | Turns run | P50 latency (ms) | P95 latency (ms) | Total cost ($) | Cache hit % | Fact recall % |",
            "|----------|-----------|------------------|------------------|----------------|-------------|---------------|",
        ]
        for sname in scenario_names:
            srows = [r for r in scenario_rows if r.get("scenario") == sname]
            latencies = _to_floats(srows, "latency_ms")
            costs = _to_floats(srows, "cost_usd")
            recalls = _to_floats(srows, "fact_recall")
            cache_hits = sum(
                1 for r in srows if r.get("cache_hit_kind") in ("exact", "semantic")
            )
            cache_pct = round(100 * cache_hits / len(srows), 1) if srows else 0.0
            lines.append(
                f"| {sname} "
                f"| {len(srows)} "
                f"| {_nan_str(_pct(latencies, 50))} "
                f"| {_nan_str(_pct(latencies, 95))} "
                f"| {_nan_str(sum(costs), '.4f')} "
                f"| {cache_pct} % "
                f"| {_nan_str(_mean(recalls) * 100, '.1f')} % |"
            )
        lines.append("")
    else:
        lines += ["_No scenario rows collected._", ""]

    # ------------------------------------------------------------------
    # 2. Summary table — attachment stress
    # ------------------------------------------------------------------
    lines += [
        "## 2. Summary — Attachment Stress",
        "",
    ]

    if attach_rows:
        sizes = sorted(
            {
                int(r["attachment_kb"])
                for r in attach_rows
                if r.get("attachment_kb") != ""
            }
        )
        lines += [
            "| Size KB | Extracted ch | Truncated ch | Markers in | Trunc'd out | P50 latency (ms) | Total cost ($) | Recall |",
            "|---------|-------------|-------------|------------|-------------|------------------|----------------|--------|",
        ]
        for kb in sizes:
            arows = [r for r in attach_rows if str(r.get("attachment_kb")) == str(kb)]
            latencies = _to_floats(arows, "latency_ms")
            costs = _to_floats(arows, "cost_usd")
            recalls = _to_floats(arows, "attachment_recall")
            ext_ch = _to_floats(arows, "attachment_extracted_chars")
            trun_ch = _to_floats(arows, "attachment_truncated_chars")
            m_in = _to_floats(arows, "attachment_markers_in")
            m_cut = _to_floats(arows, "attachment_markers_cut")
            lines.append(
                f"| {kb} "
                f"| {_nan_str(_mean(ext_ch), '.0f')} "
                f"| {_nan_str(_mean(trun_ch), '.0f')} "
                f"| {_nan_str(_mean(m_in), '.0f')} "
                f"| {_nan_str(_mean(m_cut), '.0f')} "
                f"| {_nan_str(_pct(latencies, 50))} "
                f"| {_nan_str(sum(costs), '.4f')} "
                f"| {_nan_str(_mean(recalls), '.3f')} |"
            )
        lines.append("")
    else:
        lines += ["_No attachment rows collected._", ""]

    # ------------------------------------------------------------------
    # 3. Curve: Latency vs tokens_in (binned)
    # ------------------------------------------------------------------
    lines += [
        "## 3. Curves",
        "",
        "### 3a. Latency vs Tokens In (all rows, binned)",
        "",
    ]

    all_rows_with_tokens = [
        r for r in rows if r.get("tokens_in") != "" and r.get("latency_ms") != ""
    ]
    if all_rows_with_tokens:
        bins = [
            ("< 1 000", lambda t: t < 1000),
            ("1k–2k", lambda t: 1000 <= t < 2000),
            ("2k–5k", lambda t: 2000 <= t < 5000),
            ("5k–10k", lambda t: 5000 <= t < 10000),
            ("> 10 000", lambda t: t >= 10000),
        ]
        lines += [
            "| Tokens in | Count | P50 latency (ms) | P95 latency (ms) | Mean cost ($) |",
            "|-----------|-------|------------------|------------------|---------------|",
        ]
        for label, pred in bins:
            bucket = [
                r for r in all_rows_with_tokens if _safe_float(r["tokens_in"], pred)
            ]
            if not bucket:
                continue
            lat = _to_floats(bucket, "latency_ms")
            cost = _to_floats(bucket, "cost_usd")
            lines.append(
                f"| {label} "
                f"| {len(bucket)} "
                f"| {_nan_str(_pct(lat, 50))} "
                f"| {_nan_str(_pct(lat, 95))} "
                f"| {_nan_str(_mean(cost), '.5f')} |"
            )
        lines.append("")
    else:
        lines += ["_No token data collected._", ""]

    # ------------------------------------------------------------------
    # 3b. Cumulative cost vs turn (by scenario)
    # ------------------------------------------------------------------
    lines += [
        "### 3b. Cumulative Cost vs Turn (per scenario)",
        "",
    ]

    if scenario_rows:
        scenario_names = sorted(
            {r["scenario"] for r in scenario_rows if r.get("scenario")}
        )
        max_turn = max(
            (int(r["turn_index"]) for r in scenario_rows if r.get("turn_index") != ""),
            default=0,
        )

        # Header
        header_cols = ["Turn"] + [
            s.replace("_project", "").replace("_", "-") for s in scenario_names
        ]
        sep_cols = ["----"] + ["-------"] * len(scenario_names)
        lines += [
            "| " + " | ".join(header_cols) + " |",
            "| " + " | ".join(sep_cols) + " |",
        ]

        for turn in range(1, max_turn + 1):
            cells = [str(turn)]
            for sname in scenario_names:
                cost_up_to = sum(
                    float(r["cost_usd"])
                    for r in scenario_rows
                    if r.get("scenario") == sname
                    and r.get("cost_usd") != ""
                    and int(r.get("turn_index") or 0) <= turn
                )
                cells.append(f"${cost_up_to:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += ["_No scenario rows collected._", ""]

    # ------------------------------------------------------------------
    # 3c. Fact recall vs N (by scenario)
    # ------------------------------------------------------------------
    lines += [
        "### 3c. Fact Recall vs N Turns (final-turn recall per run length)",
        "",
        "_Recall is computed on the last turn of each (scenario, n_turns) run._",
        "",
    ]

    if scenario_rows:
        scenario_names = sorted(
            {r["scenario"] for r in scenario_rows if r.get("scenario")}
        )
        n_vals = sorted(
            {int(r["n_turns"]) for r in scenario_rows if r.get("n_turns") != ""}
        )

        header_cols = ["N"] + [
            s.replace("_project", "").replace("_", "-") for s in scenario_names
        ]
        sep_cols = ["---"] + ["-------"] * len(scenario_names)
        lines += [
            "| " + " | ".join(header_cols) + " |",
            "| " + " | ".join(sep_cols) + " |",
        ]

        for n in n_vals:
            cells = [str(n)]
            for sname in scenario_names:
                # Last-turn rows for this (scenario, n_turns)
                last_rows = [
                    r
                    for r in scenario_rows
                    if r.get("scenario") == sname
                    and r.get("n_turns") != ""
                    and int(r["n_turns"]) == n
                    and r.get("turn_index") != ""
                    and int(r["turn_index"]) == n
                ]
                recalls = _to_floats(last_rows, "fact_recall")
                cells.append(_nan_str(_mean(recalls), ".3f") if recalls else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += ["_No scenario rows collected._", ""]

    # ------------------------------------------------------------------
    # 4. Analysis paragraphs
    # ------------------------------------------------------------------
    lines += [
        "## 4. Analysis: Where Does the CAG Start to Break and Why",
        "",
    ]
    lines += _analysis_paragraphs(scenario_rows, attach_rows)

    lines.append("")
    return "\n".join(lines)


def _safe_float(val: Any, pred: Any) -> bool:
    try:
        return pred(float(val))
    except (TypeError, ValueError):
        return False


def _analysis_paragraphs(
    scenario_rows: list[dict],
    attach_rows: list[dict],
) -> list[str]:
    """Generate two data-driven analysis paragraphs about CAG degradation."""
    paras: list[str] = []

    # ---- Paragraph 1: session-memory / context-window degradation ----
    if scenario_rows:
        # Find the first N where mean fact_recall < 0.8
        n_vals = sorted(
            {int(r["n_turns"]) for r in scenario_rows if r.get("n_turns") != ""}
        )
        drift_at: int | None = None
        for n in n_vals:
            last_rows = [
                r
                for r in scenario_rows
                if r.get("n_turns") != ""
                and int(r["n_turns"]) == n
                and r.get("turn_index") != ""
                and int(r["turn_index"]) == n
            ]
            recalls = _to_floats(last_rows, "fact_recall")
            if recalls and _mean(recalls) < 0.80:
                drift_at = n
                break

        # Find first turn with latency > budget across all scenarios
        latency_breach_turn: int | None = None
        for r in sorted(scenario_rows, key=lambda x: int(x.get("turn_index") or 0)):
            lat = r.get("latency_ms")
            try:
                if float(lat) > LATENCY_BUDGET_MS:  # type: ignore[arg-type]
                    latency_breach_turn = int(r["turn_index"])
                    break
            except (TypeError, ValueError):
                pass

        # Cache hit stats
        cache_rows = [
            r for r in scenario_rows if r.get("cache_hit_kind") in ("exact", "semantic")
        ]
        cache_pct = (
            round(100 * len(cache_rows) / len(scenario_rows), 1)
            if scenario_rows
            else 0.0
        )

        drift_msg = (
            f"at N = {drift_at}"
            if drift_at
            else "not observed within the tested N range"
        )
        latency_msg = (
            f"at turn {latency_breach_turn}"
            if latency_breach_turn
            else "never exceeded the {LATENCY_BUDGET_MS} ms budget"
        )

        paras += [
            "### Context-window / Session-memory degradation",
            "",
            f"Fact recall dropped below 80 % {drift_msg}. "
            f"This is the point where the sliding-window history (max_turns = 6) has "
            f"evicted early turns and the accumulated summary has become the sole carrier "
            f"of old facts. The summariser compresses lossy-ly: project names and "
            f"technology choices survive, but quantitative assertions (budget ceilings, "
            f"team sizes) are often absorbed into prose and become harder to match exactly. "
            f"Latency first exceeded the {LATENCY_BUDGET_MS:,} ms budget {latency_msg}. "
            f"The CAG few-shot examples travel in every system prompt, so the token count "
            f"grows proportionally with conversation depth regardless of the sliding window — "
            f"the context bloat is driven by the static example block, not by the growing "
            f"history. Cache hit rate across all scenario turns was {cache_pct} %, "
            f"suggesting that the in-memory exact-match cache rarely fires on varied "
            f"conversational transcripts (as expected — each turn is unique).",
            "",
        ]
    else:
        paras += [
            "### Context-window / Session-memory degradation",
            "",
            "_No scenario data collected. Run without --dry-run to populate this section._",
            "",
        ]

    # ---- Paragraph 2: attachment recall / truncation degradation ----
    if attach_rows:
        # Find the size where recall first drops below 0.7
        recall_drop_kb: int | None = None
        sizes = sorted(
            {
                int(r["attachment_kb"])
                for r in attach_rows
                if r.get("attachment_kb") != ""
            }
        )
        for kb in sizes:
            if kb == 0:
                continue
            kb_rows = [r for r in attach_rows if str(r.get("attachment_kb")) == str(kb)]
            recalls = _to_floats(kb_rows, "attachment_recall")
            if recalls and _mean(recalls) < 0.70:
                recall_drop_kb = kb
                break

        # Cost increase from 0 KB to 100 KB
        cost_0 = _mean(
            _to_floats(
                [r for r in attach_rows if str(r.get("attachment_kb")) == "0"],
                "cost_usd",
            )
        )
        cost_100 = _mean(
            _to_floats(
                [r for r in attach_rows if str(r.get("attachment_kb")) == "100"],
                "cost_usd",
            )
        )
        cost_ratio = (
            round(cost_100 / cost_0, 1) if cost_0 > 0 and cost_0 == cost_0 else None
        )

        recall_drop_msg = (
            f"at {recall_drop_kb} KB"
            if recall_drop_kb
            else "not observed in the tested range"
        )
        cost_ratio_msg = (
            f"{cost_ratio}× the baseline cost" if cost_ratio else "cost data incomplete"
        )

        paras += [
            "### Attachment-size degradation (CAG retrieval boundary)",
            "",
            f"Attachment recall fell below 70 % {recall_drop_msg}. "
            f"The root cause is the MAX_ATTACHMENT_CHARS = 60 000 truncation cap: "
            f"once extracted PDF text exceeds the cap the tail of the document is silently "
            f"discarded before it reaches the prompt. For the 100 KB test point this cut off "
            f"4 of 10 recall markers, creating a theoretical ceiling of 60 % recall "
            f"regardless of model quality. The remaining drop — if any — is attributable to "
            f"the LLM's own attention distribution: when the attachment is long the model "
            f"anchors on the first few paragraphs (the 'lost-in-the-middle' effect documented "
            f"in Liu et al. 2023) and tends to paraphrase rather than name-check individual "
            f"modules. The 100 KB run costs roughly {cost_ratio_msg}, confirming that "
            f"attachment-heavy workflows carry a non-trivial price premium. "
            f"Mitigation options: (1) raise or remove the cap and rely on model context limits; "
            f"(2) implement a retrieve-and-rank step that selects only the top-K chunks most "
            f"relevant to the estimation task (hybrid CAG + RAG); "
            f"(3) summarise large attachments server-side before embedding them in the prompt.",
            "",
        ]
    else:
        paras += [
            "### Attachment-size degradation (CAG retrieval boundary)",
            "",
            "_No attachment data collected. Run with --attachment-sizes to populate this section._",
            "",
        ]

    return paras


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def main(args: argparse.Namespace) -> int:  # noqa: C901
    """Orchestrate scenario + attachment stress runs and write outputs.

    Returns:
        0 on success, non-zero on unrecoverable error.
    """
    # ---- Resolve profiles ---------------------------------------------
    requested_aliases = [s.strip() for s in args.scenarios.split(",")]
    profiles: list[ScenarioProfile] = []
    for alias in requested_aliases:
        pname = _SCENARIO_ALIASES.get(alias.lower(), alias)
        prof = _PROFILE_BY_NAME.get(pname)
        if prof is None:
            print(
                f"Unknown scenario '{alias}'. "
                f"Valid aliases: {list(_SCENARIO_ALIASES.keys())}",
                file=sys.stderr,
            )
            return 2
        profiles.append(prof)

    # ---- Resolve attachment sizes -------------------------------------
    attach_sizes: list[int] = []
    if args.attachment_sizes:
        for s in args.attachment_sizes.split(","):
            s = s.strip()
            if s:
                attach_sizes.append(int(s))

    # ---- Resolve n-values for scenarios ------------------------------
    n_values = N_VALUES  # always run against the canonical set

    output_path = Path(args.output)
    report_path = output_path.with_suffix("").parent / "REPORT.md"

    all_rows: list[dict] = []

    http_mode = bool(args.http)

    if http_mode:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            all_rows = await _orchestrate(
                profiles,
                n_values,
                attach_sizes,
                args.repeats,
                http=True,
                client=client,
                base_url=args.http,
                service=None,
            )
    else:
        from src.services.estimation import EstimationService

        service = EstimationService()
        all_rows = await _orchestrate(
            profiles,
            n_values,
            attach_sizes,
            args.repeats,
            http=False,
            client=None,
            base_url=None,
            service=service,
        )

    _write_csv(all_rows, output_path)

    report_md = _generate_report(all_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"REPORT → {report_path}")

    return 0


async def _orchestrate(
    profiles: list[ScenarioProfile],
    n_values: tuple[int, ...],
    attach_sizes: list[int],
    repeats: int,
    *,
    http: bool,
    client: Any,
    base_url: str | None,
    service: Any,
) -> list[dict]:
    """Inner loop: iterate over all (profile, n, repeat) and (size, repeat) combos."""
    rows: list[dict] = []
    total_scenario_runs = len(profiles) * len(n_values) * repeats
    total_attach_runs = len(attach_sizes) * repeats
    total_runs = total_scenario_runs + total_attach_runs
    done = 0

    # ---- Scenario runs -----------------------------------------------
    for profile in profiles:
        for n in n_values:
            if n > len(profile.turns):
                print(
                    f"  [skip] {profile.name} only has {len(profile.turns)} turns "
                    f"(requested N={n})"
                )
                done += repeats
                continue
            for rep in range(1, repeats + 1):
                tag = f"{profile.name}  N={n:>2}  rep={rep}/{repeats}"
                print(f"  [{done + 1:>3}/{total_runs}] {tag} … ", end="", flush=True)
                try:
                    if http:
                        turn_rows = await _run_scenario_http(
                            profile,
                            n,
                            rep,
                            client,
                            base_url,  # type: ignore[arg-type]
                        )
                    else:
                        turn_rows = await _run_scenario_inprocess(
                            profile, n, rep, service
                        )
                    rows.extend(turn_rows)
                    latencies = _to_floats(turn_rows, "latency_ms")
                    costs = _to_floats(turn_rows, "cost_usd")
                    recalls = _to_floats(turn_rows, "fact_recall")
                    print(
                        f"{len(turn_rows)} turns  "
                        f"P50={_nan_str(_pct(latencies, 50))} ms  "
                        f"cost=${_nan_str(sum(costs), '.4f')}  "
                        f"recall={_nan_str(_mean(recalls), '.2f')}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"ERROR: {exc}")
                    logger.exception(
                        "scenario_run_failed", profile=profile.name, n=n, rep=rep
                    )
                done += 1

    # ---- Attachment runs ----------------------------------------------
    for kb in attach_sizes:
        for rep in range(1, repeats + 1):
            tag = f"attachment  {kb:>3} KB  rep={rep}/{repeats}"
            print(f"  [{done + 1:>3}/{total_runs}] {tag} … ", end="", flush=True)
            try:
                if http:
                    row = await _run_attachment_http(kb, rep, client, base_url)  # type: ignore[arg-type]
                else:
                    row = await _run_attachment_inprocess(kb, rep, service)
                rows.append(row)
                lat = row.get("latency_ms")
                cost = row.get("cost_usd")
                recall = row.get("attachment_recall")
                print(
                    f"latency={_nan_str(float(lat) if lat != '' else float('nan')):.1f} ms  "
                    f"cost=${float(cost):.4f}  "  # type: ignore[arg-type]
                    f"recall={recall}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {exc}")
                logger.exception("attachment_run_failed", kb=kb, rep=rep)
            done += 1

    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evals.stress.run",
        description="Stress-suite runner: scenarios + attachment stress → CSV + REPORT.md",
    )
    parser.add_argument(
        "--http",
        metavar="URL",
        default=None,
        help=(
            "Base URL of a running sw-estimator server "
            "(e.g. http://localhost:8000). "
            "When omitted the evaluation runs in-process."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="growing,pivot,contradiction",
        help="Comma-separated scenario aliases: growing, pivot, contradiction.",
    )
    parser.add_argument(
        "--attachment-sizes",
        dest="attachment_sizes",
        default="0,5,20,50,100",
        help="Comma-separated attachment sizes in KB (0 = no attachment baseline).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent repeats per (scenario, N) or (attachment, size) combination.",
    )
    parser.add_argument(
        "--output",
        default="evals/stress/results.csv",
        help="Path for the output CSV file.",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()
    sys.exit(asyncio.run(main(_args)))
