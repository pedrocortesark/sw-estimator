"""Attachment-size stress test for the sw-estimator pipeline.

Generates synthetic PDFs at five calibrated sizes, submits each through the
same estimation call, and records three curves:

  1. latency_ms     — wall-clock time from call to structured response.
  2. cost_usd       — LLM API cost as reported by the pricing table.
  3. recall         — fraction of attachment markers mentioned in the response.

Attachment sizes tested
-----------------------
  0 KB   no attachment (baseline — transcript only)
  5 KB   ≈ 2 pages of spec text  (≈ 4 000 extracted chars)
 20 KB   ≈ 8 pages               (≈ 16 000 extracted chars)
 50 KB   ≈ 20 pages              (≈ 40 000 extracted chars)
100 KB   near the MAX_ATTACHMENT_CHARS = 60 000 truncation cap;
         markers placed beyond the cap will be missing from the LLM input,
         producing a measurable drop in recall.

Recall mechanics
----------------
Each generated spec document contains 10 unique module names ("RECALL_MARKERS")
spread evenly throughout the text body.  After the LLM returns a response,
recall is computed as:

    recall = (# markers found in response text) / (# markers present in truncated input)

Using the denominator as "present in truncated input" (not total markers)
isolates the LLM's attention quality from the truncation effect; a separate
column ``markers_truncated`` counts how many were cut off before reaching
the model.

Truncation
----------
``MAX_ATTACHMENT_CHARS = 60_000`` is the cap applied to the extracted text
before it is concatenated to the transcript and forwarded to the LLM.  This
mirrors what a production truncation guard would enforce.  The script
applies the cap itself because the ``EstimationRequest`` schema has an even
stricter ``max_length=2000`` guard that the HTTP layer enforces on raw
user-typed transcripts — not on programmatically constructed enriched text.
To measure at realistic scale we call ``generate_estimation()`` directly,
bypassing Pydantic validation on the combined transcript length.

Usage
-----
Run all five sizes with LLM calls::

    python -m evals.stress.attachment_stress

Dry-run (generate PDFs, print sizes, no LLM calls)::

    python -m evals.stress.attachment_stress --dry-run

Save results to CSV::

    python -m evals.stress.attachment_stress --output results/attach_stress.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sys
import time
from pathlib import Path

import structlog

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.document_extractor import build_attachment_block, extract_text
from src.services.llm_service import generate_estimation

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Attachment sizes to test, in target PDF bytes.
# The "0" entry is the baseline (no attachment).
ATTACHMENT_SIZES_KB: tuple[int, ...] = (0, 5, 20, 50, 100)

# Maximum characters kept from extracted attachment text before concatenating
# to the transcript.  Text beyond this threshold is silently discarded.
MAX_ATTACHMENT_CHARS: int = 60_000

# Fixed short transcript that stays identical across all size points so that
# only the attachment content varies.
BASE_TRANSCRIPT: str = (
    "We need to build a SaaS e-commerce platform with product catalog, "
    "user accounts with roles, shopping cart, and checkout. "
    "The tech stack is FastAPI (Python) backend, React frontend, "
    "PostgreSQL database. Team of 3 engineers, 6-month timeline."
)

# Unique module names embedded in the synthetic spec doc.
# Their presence (or absence) in the LLM response drives the recall metric.
# Placed at decile boundaries (5 %, 15 %, …, 95 %) of the text body so that
# the 100 KB document has its last four markers beyond the 60 000-char cap.
RECALL_MARKERS: list[tuple[str, str]] = [
    ("ZephyrAuth", "authentication and authorisation subsystem"),
    ("NebulaPay", "payment processing engine with Stripe integration"),
    ("AuroraCache", "distributed caching layer backed by Redis"),
    ("StellarAPI", "public RESTful API gateway with rate limiting"),
    ("CosmosSearch", "full-text search integration via Elasticsearch"),
    ("OrionMetrics", "analytics and KPI reporting dashboard"),
    ("PulsarNotify", "real-time notification engine (email + push)"),
    ("VegaExport", "data export module - CSV, Excel, and PDF reports"),
    ("SiriusAdmin", "administrative back-office and user management panel"),
    ("LyraCI", "CI/CD pipeline and deployment automation"),
]

# Plausible filler text used as padding between marker sections.
# Long enough to avoid repeating visible patterns.
_LOREM = (
    "The system must handle concurrent requests from multiple tenants without "
    "performance degradation. Each service component is independently scalable "
    "via Kubernetes horizontal pod autoscaling. Database migrations are managed "
    "through Alembic with zero-downtime strategies. All API endpoints are versioned "
    "under /api/v1 and protected by JWT bearer tokens. Observability is implemented "
    "with structured JSON logs (structlog), distributed tracing (OpenTelemetry), "
    "and metrics exported to Prometheus. The infrastructure is defined as code "
    "using Terraform modules targeting AWS ECS Fargate. Staging and production "
    "environments are fully isolated with separate VPCs and IAM roles. "
    "Automated security scans (Trivy, Bandit) run on every pull request. "
    "The frontend communicates exclusively through the API gateway, never "
    "directly with backend microservices. Load testing is performed with "
    "Locust targeting 500 concurrent virtual users at peak. "
)

# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------


def _build_spec_text(target_chars: int) -> tuple[str, list[str]]:
    """Build a spec-like text body of approximately *target_chars* characters.

    Embeds each of the 10 :data:`RECALL_MARKERS` at decile boundaries so that
    a later truncation cut at 60 000 chars predictably severs markers whose
    decile position × ``target_chars`` exceeds the cap.

    Returns:
        A ``(text, present_markers)`` tuple where *present_markers* lists the
        marker names actually embedded in the returned text.
    """
    n_markers = len(RECALL_MARKERS)
    marker_names: list[str] = []

    # Compute the char index where each marker should appear.
    # Deciles: 5 %, 15 %, 25 %, …, 95 % of target_chars.
    marker_positions: list[int] = [
        int(target_chars * (2 * i + 1) / (2 * n_markers)) for i in range(n_markers)
    ]

    # Build the document section by section.
    # We fill with lorem-style filler between marker insertion points.
    parts: list[str] = [
        "TECHNICAL SPECIFICATION\n"
        "E-Commerce SaaS Platform\n"
        "========================\n\n"
        "This document describes the functional and non-functional requirements "
        "of the e-commerce platform as agreed during the discovery phase. "
        "Each section corresponds to a distinct deliverable module.\n\n"
    ]
    current_chars = sum(len(p) for p in parts)

    def _filler_up_to(target: int) -> str:
        """Return enough filler text to advance current_chars to ~target."""
        needed = max(0, target - current_chars - sum(len(p) for p in parts))
        if needed == 0:
            return ""
        repetitions = (needed // len(_LOREM)) + 2
        return (_LOREM * repetitions)[:needed]

    for idx, (marker_name, marker_desc) in enumerate(RECALL_MARKERS):
        target_pos = marker_positions[idx]
        # Pad to the target position.
        filler = _filler_up_to(target_pos)
        if filler:
            parts.append(filler)
        # Insert the marker section.
        section = (
            f"\n--- Module: {marker_name} ---\n"
            f"Description: {marker_desc.capitalize()}.\n"
            f"The {marker_name} module is responsible for handling all aspects of "
            f"{marker_desc}. It exposes a dedicated REST API surface and integrates "
            f"with the core platform via internal service-to-service calls. "
            f"All {marker_name} operations are logged at INFO level for audit trails. "
            f"Performance SLA: p99 latency < 200 ms at 100 rps.\n\n"
        )
        parts.append(section)
        marker_names.append(marker_name)

    # Pad tail to reach target_chars.
    total_so_far = sum(len(p) for p in parts)
    if total_so_far < target_chars:
        tail_needed = target_chars - total_so_far
        repetitions = (tail_needed // len(_LOREM)) + 2
        parts.append((_LOREM * repetitions)[:tail_needed])

    text = "".join(parts)
    return text, marker_names


def generate_pdf(target_extracted_chars: int) -> tuple[bytes, str, list[str]]:
    """Generate a synthetic PDF whose extracted text has ≈ *target_extracted_chars* chars.

    fpdf2 embeds text directly; pypdf extracts it back with high fidelity.
    We overshoot by 10 % to account for newline normalisation and whitespace
    collapsing during extraction.

    Args:
        target_extracted_chars: Desired character count after pypdf extraction.

    Returns:
        ``(pdf_bytes, raw_text, marker_names)`` — the PDF bytes, the raw text
        body used to build it, and the list of marker names embedded in that
        body (all 10 for any size > 0).
    """
    from fpdf import FPDF

    target_with_margin = int(target_extracted_chars * 1.10)
    raw_text, markers = _build_spec_text(target_with_margin)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    # multi_cell handles word-wrapping automatically.
    # w=0 means "use the full available width".
    pdf.multi_cell(w=0, h=6, text=raw_text)

    return pdf.output(), raw_text, markers


# ---------------------------------------------------------------------------
# Recall metric
# ---------------------------------------------------------------------------


def compute_recall(
    response_text: str,
    markers_in_input: list[str],
) -> tuple[float, list[str], list[str]]:
    """Compute what fraction of *markers_in_input* appear in *response_text*.

    Case-insensitive substring match — if the LLM capitalises differently
    (e.g. "ZephyrAuth" vs "zephyrauth") we still count it.

    Args:
        response_text:   Full text to search (executive_summary + phase names).
        markers_in_input: Marker names that were present in the truncated input.

    Returns:
        ``(score, found, missing)`` — float in [0, 1], list of found marker
        names, list of missing marker names.
    """
    if not markers_in_input:
        return 0.0, [], []

    lower_response = response_text.lower()
    found = [m for m in markers_in_input if m.lower() in lower_response]
    missing = [m for m in markers_in_input if m.lower() not in lower_response]
    score = len(found) / len(markers_in_input)
    return score, found, missing


def _response_corpus(result_dict: dict) -> str:
    """Extract all searchable text from a ``generate_estimation`` result dict."""
    estimation = result_dict.get("estimation_result")
    if estimation is None:
        return ""
    parts: list[str] = [estimation.executive_summary or ""]
    for phase in estimation.phases or []:
        parts.append(phase.name or "")
        for task in phase.tasks or []:
            parts.append(task.name or "")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Single-point runner
# ---------------------------------------------------------------------------


async def _run_point(
    size_kb: int,
    dry_run: bool = False,
) -> dict:
    """Execute one attachment-size data point and return a metrics dict.

    For the 0 KB baseline no PDF is generated; the base transcript is sent
    directly.  For other sizes the PDF is generated, text extracted, the
    cap applied, and the enriched transcript forwarded to the LLM.

    Args:
        size_kb:  Target PDF size in kilobytes (0 for no-attachment baseline).
        dry_run:  When True, skip the LLM call and return only sizing info.

    Returns:
        dict with keys: size_kb, pdf_bytes, extracted_chars,
        truncated_chars, markers_in_input, markers_truncated,
        latency_ms, tokens_in, tokens_out, cost_usd, recall,
        found_markers, missing_markers, error.
    """
    label = f"{size_kb} KB"

    # --- Build attachment -----------------------------------------------
    pdf_bytes: bytes = b""
    extracted_chars = 0
    truncated_chars = 0
    markers_in_input: list[str] = []
    markers_truncated = 0
    raw_text = ""

    if size_kb > 0:
        target_bytes = size_kb * 1024
        # Target extracted chars ≈ 85 % of PDF bytes (empirical ratio for
        # single-column Helvetica 11 pt with default fpdf2 margins).
        target_extracted = int(target_bytes * 0.85)
        pdf_bytes, raw_text, all_markers = generate_pdf(target_extracted)

        # Extract text exactly as the production pipeline would.
        extracted_text = extract_text("spec.pdf", pdf_bytes)
        extracted_chars = len(extracted_text)

        # Apply the production truncation cap.
        truncated_text = extracted_text[:MAX_ATTACHMENT_CHARS]
        truncated_chars = len(truncated_text)

        # Determine which markers survived truncation.
        markers_in_input = [
            m for m in all_markers if m.lower() in truncated_text.lower()
        ]
        markers_truncated = len(all_markers) - len(markers_in_input)

        attachment_block = build_attachment_block("spec.pdf", truncated_text)
        enriched_transcript = BASE_TRANSCRIPT + "\n\n" + attachment_block
    else:
        # 0 KB baseline — transcript only, no attachment.
        enriched_transcript = BASE_TRANSCRIPT
        markers_in_input = []
        markers_truncated = 0

    logger.info(
        "attachment_point_prepared",
        size_kb=size_kb,
        pdf_bytes=len(pdf_bytes),
        extracted_chars=extracted_chars,
        truncated_chars=truncated_chars,
        markers_in_input=len(markers_in_input),
        markers_truncated=markers_truncated,
        enriched_chars=len(enriched_transcript),
    )

    if dry_run:
        return {
            "size_kb": size_kb,
            "pdf_bytes": len(pdf_bytes),
            "extracted_chars": extracted_chars,
            "truncated_chars": truncated_chars,
            "markers_in_input": len(markers_in_input),
            "markers_truncated": markers_truncated,
            "enriched_chars": len(enriched_transcript),
            "latency_ms": None,
            "tokens_in": None,
            "tokens_out": None,
            "cost_usd": None,
            "recall": None,
            "found_markers": [],
            "missing_markers": [],
            "error": None,
        }

    # --- LLM call -------------------------------------------------------
    t0 = time.perf_counter()
    error: str | None = None
    result_dict: dict = {}

    try:
        result_dict = await generate_estimation(transcript=enriched_transcript)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        logger.error("attachment_point_failed", size_kb=size_kb, error=error)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # --- Metrics --------------------------------------------------------
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    recall = 0.0
    found_markers: list[str] = []
    missing_markers: list[str] = []

    if result_dict:
        usage = result_dict.get("usage")
        if usage is not None:
            tokens_in = usage.input_tokens
            tokens_out = usage.output_tokens
            cost_usd = round(usage.cost_usd, 6)

        corpus = _response_corpus(result_dict)
        recall, found_markers, missing_markers = compute_recall(
            corpus, markers_in_input
        )
        recall = round(recall, 4)

    return {
        "size_kb": size_kb,
        "pdf_bytes": len(pdf_bytes),
        "extracted_chars": extracted_chars,
        "truncated_chars": truncated_chars,
        "markers_in_input": len(markers_in_input),
        "markers_truncated": markers_truncated,
        "enriched_chars": len(enriched_transcript),
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "recall": recall,
        "found_markers": found_markers,
        "missing_markers": missing_markers,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("size_kb", 7, "Size KB"),
    ("extracted_chars", 14, "Extracted ch"),
    ("truncated_chars", 14, "Truncated ch"),
    ("markers_in_input", 10, "Markers in"),
    ("markers_truncated", 11, "Trunc'd out"),
    ("latency_ms", 10, "Latency ms"),
    ("tokens_in", 9, "Tok in"),
    ("tokens_out", 9, "Tok out"),
    ("cost_usd", 10, "Cost USD"),
    ("recall", 8, "Recall"),
]


def _fmt(value, width: int) -> str:
    if value is None:
        return f"{'—':>{width}}"
    if isinstance(value, float):
        return f"{value:>{width}.4f}"
    return f"{str(value):>{width}}"


def _print_table(results: list[dict]) -> None:
    header = "  ".join(f"{label:>{width}}" for _, width, label in _TABLE_COLS)
    sep = "  ".join("-" * width for _, width, _ in _TABLE_COLS)
    print()
    print(header)
    print(sep)
    for r in results:
        row = "  ".join(_fmt(r[key], width) for key, width, _ in _TABLE_COLS)
        err = f"  ← ERROR: {r['error']}" if r.get("error") else ""
        print(row + err)
    print()


def _save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "size_kb",
        "pdf_bytes",
        "extracted_chars",
        "truncated_chars",
        "markers_in_input",
        "markers_truncated",
        "enriched_chars",
        "latency_ms",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "recall",
        "found_markers",
        "missing_markers",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["found_markers"] = "|".join(row["found_markers"])
            row["missing_markers"] = "|".join(row["missing_markers"])
            writer.writerow(row)
    print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(dry_run: bool = False, output: str | None = None) -> list[dict]:
    """Run the attachment-stress suite and return all results dicts.

    Args:
        dry_run: Skip LLM calls; only generate PDFs and report sizes.
        output:  Optional path to write a CSV results file.
    """
    results: list[dict] = []
    print(f"\nAttachment stress test — MAX_ATTACHMENT_CHARS = {MAX_ATTACHMENT_CHARS:,}")
    print(f"Sizes: {ATTACHMENT_SIZES_KB} KB  |  dry_run={dry_run}\n")

    for size_kb in ATTACHMENT_SIZES_KB:
        print(f"  Running {size_kb:>3} KB … ", end="", flush=True)
        result = await _run_point(size_kb, dry_run=dry_run)
        results.append(result)
        if dry_run:
            print(
                f"extracted={result['extracted_chars']:>7} ch  "
                f"markers_in={result['markers_in_input']}  "
                f"markers_cut={result['markers_truncated']}"
            )
        else:
            status = (
                f"ERROR: {result['error']}"
                if result["error"]
                else (
                    f"latency={result['latency_ms']:>7.0f} ms  "
                    f"cost=${result['cost_usd']:.6f}  "
                    f"recall={result['recall']:.0%}"
                )
            )
            print(status)

    _print_table(results)

    if output:
        _save_csv(results, Path(output))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attachment-size stress test for sw-estimator."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate PDFs and print sizes without calling the LLM.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write results to a CSV file at PATH.",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, output=args.output))
