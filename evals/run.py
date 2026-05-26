"""CLI runner for the sw-estimator evaluation suite.

Usage examples
--------------
Run all golden cases with the default actor (EstimationService):

    python -m evals.run --mode actor

Run a specific subset of cases:

    python -m evals.run --mode actor --cases gc-01,gc-05,gc-13

Save JSON results to a file:

    python -m evals.run --mode actor --output results/run_01.json

Modes
-----
actor
    Calls EstimationService.estimate() directly (no HTTP, no guardrails cache).
    Requires a valid OPENAI_API_KEY (or ANTHROPIC_API_KEY) in the environment.

acb  (Acceptance / CI batch)
    Same as actor but exits with code 1 if any case fails all three metrics.
    Designed for use in CI pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import structlog

# Ensure the project root is on sys.path when invoked as `python -m evals.run`
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schemas.estimation import EstimationRequest
from src.services.estimation import EstimationService
from evals.metrics import CostBoundsMetric, ContentRecallMetric, SchemaAdherenceMetric

logger = structlog.get_logger()

_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

_METRICS = [
    SchemaAdherenceMetric(),
    CostBoundsMetric(),
    ContentRecallMetric(),
]


def _load_cases(ids: list[str] | None) -> list[dict]:
    """Load golden cases from the JSON dataset, optionally filtered by ID."""
    with open(_DATASET_PATH, encoding="utf-8") as f:
        cases: list[dict] = json.load(f)
    if ids:
        id_set = set(ids)
        cases = [c for c in cases if c["id"] in id_set]
        missing = id_set - {c["id"] for c in cases}
        if missing:
            logger.warning("unknown_case_ids", ids=sorted(missing))
    return cases


def _print_header() -> None:
    cols = ["ID", "Description", "schema", "cost_bounds", "content_recall", "ms"]
    widths = [8, 42, 8, 12, 16, 7]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    separator = "  ".join("-" * w for w in widths)
    print(header)
    print(separator)


def _print_row(result: dict) -> None:
    widths = [8, 42, 8, 12, 16, 7]
    desc = result["description"][:42]
    values = [
        result["id"],
        desc,
        "PASS" if result["metrics"]["schema_adherence"] else "FAIL",
        "PASS" if result["metrics"]["cost_bounds"] else "FAIL",
        "PASS" if result["metrics"]["content_recall"] else "FAIL",
        f"{result['latency_ms']:.0f}",
    ]
    print("  ".join(f"{v:<{w}}" for v, w in zip(values, widths)))


async def _run_case(service: EstimationService, case: dict) -> dict:
    """Execute a single golden case and return a results dict."""
    inp = case["input"]
    request = EstimationRequest(
        transcript=inp["transcript"],
        project_type=inp.get("project_type"),
        detail_level=inp.get("detail_level"),
        output_format=inp.get("output_format"),
    )
    expected = case["expected"]
    error: str | None = None
    response_dict: dict[str, Any] = {}

    t0 = time.monotonic()
    try:
        response = await service.estimate(request)
        response_dict = response.model_dump()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        logger.error("eval_case_error", case_id=case["id"], error=error)
    latency_ms = (time.monotonic() - t0) * 1000

    schema_metric = _METRICS[0]
    cost_metric = _METRICS[1]
    content_metric = _METRICS[2]

    metrics = {
        "schema_adherence": schema_metric.score(response_dict) if not error else False,
        "cost_bounds": cost_metric.score(response_dict, expected) if not error else False,
        "content_recall": content_metric.score(response_dict, expected) if not error else False,
    }

    return {
        "id": case["id"],
        "description": case["description"],
        "metrics": metrics,
        "latency_ms": latency_ms,
        "error": error,
        "response": response_dict,
    }


async def _run_all(cases: list[dict]) -> list[dict]:
    """Run all cases sequentially and return results."""
    service = EstimationService()
    results = []
    _print_header()
    for case in cases:
        result = await _run_case(service, case)
        _print_row(result)
        results.append(result)
    return results


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = {
        "schema_adherence": sum(1 for r in results if r["metrics"]["schema_adherence"]),
        "cost_bounds": sum(1 for r in results if r["metrics"]["cost_bounds"]),
        "content_recall": sum(1 for r in results if r["metrics"]["content_recall"]),
    }
    errors = sum(1 for r in results if r["error"])
    avg_ms = sum(r["latency_ms"] for r in results) / max(total, 1)

    print()
    print(f"Total cases : {total}")
    print(f"Errors      : {errors}")
    print(f"schema_adherence : {passed['schema_adherence']}/{total}")
    print(f"cost_bounds      : {passed['cost_bounds']}/{total}")
    print(f"content_recall   : {passed['content_recall']}/{total}")
    print(f"Avg latency : {avg_ms:.0f} ms")


def _all_pass(results: list[dict]) -> bool:
    """Return True only if every case passes all three metrics."""
    return all(
        r["metrics"]["schema_adherence"]
        and r["metrics"]["cost_bounds"]
        and r["metrics"]["content_recall"]
        for r in results
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run sw-estimator evaluation suite against the golden dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["actor", "acb"],
        default="actor",
        help="actor: run and report; acb: run and exit 1 on any failure.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated list of case IDs to run (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results file (optional).",
    )
    args = parser.parse_args(argv)

    ids = [s.strip() for s in args.cases.split(",")] if args.cases else None
    cases = _load_cases(ids)
    if not cases:
        print("No cases matched. Exiting.", file=sys.stderr)
        return 2

    results = asyncio.run(_run_all(cases))
    _print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # Strip the raw LLM response from the saved file to keep it small
            slim = [
                {k: v for k, v in r.items() if k != "response"}
                for r in results
            ]
            json.dump(slim, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    if args.mode == "acb" and not _all_pass(results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
