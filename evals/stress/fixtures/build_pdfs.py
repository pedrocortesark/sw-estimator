"""Standalone script that produces four calibrated synthetic PDFs.

The generated files are **deterministic** — running this script twice with the
same version of fpdf2 produces byte-for-byte identical PDFs.  Because of this
they are NOT committed to the repository; the runner regenerates them as needed.

Output files (written to the same directory as this script)
------------------------------------------------------------
  attach_5kb.pdf    ≈  5 KB  (≈  4 000 extracted chars)
  attach_20kb.pdf   ≈ 20 KB  (≈ 16 000 extracted chars)
  attach_50kb.pdf   ≈ 50 KB  (≈ 40 000 extracted chars)
  attach_100kb.pdf  ≈100 KB  (≈ 80 000 extracted chars — exceeds the
                               MAX_ATTACHMENT_CHARS = 60 000 truncation cap,
                               so the last 4 recall markers are cut before
                               reaching the LLM)

Note on the 0 KB baseline
--------------------------
There is intentionally no ``attach_0kb.pdf``: the zero-attachment baseline is
represented by the *absence* of a file argument in the runner, not by an empty
PDF (which would still add a PDF header and inflate the token count slightly).

Recall markers
--------------
Each PDF embeds the 10 unique module names from
:data:`evals.stress.attachment_stress.RECALL_MARKERS` spread at decile
positions (5 %, 15 %, …, 95 %) of the target character count.  For the 100 KB
PDF the last four markers (VegaExport, SiriusAdmin, LyraCI and the virtual
10th-decile position) appear after the 60 000-char truncation boundary; this is
intentional and produces a measurable recall drop that the stress runner
measures.

Usage
-----
Generate all four PDFs into the fixtures directory::

    python -m evals.stress.fixtures.build_pdfs

Generate a specific size::

    python -m evals.stress.fixtures.build_pdfs --sizes 5,50

Verify without writing::

    python -m evals.stress.fixtures.build_pdfs --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.stress.attachment_stress import generate_pdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target sizes and their output filenames.
# Key   = size in KB  (same unit as ATTACHMENT_SIZES_KB in attachment_stress.py)
# Value = output filename (relative to this script's directory)
PDF_TARGETS: dict[int, str] = {
    5:   "attach_5kb.pdf",
    20:  "attach_20kb.pdf",
    50:  "attach_50kb.pdf",
    100: "attach_100kb.pdf",
}

# Conversion from target KB to target *extracted* characters.
# The PDF container adds ~15 % overhead over raw text, so we aim for 85 % of
# the byte count as character count — the same heuristic used in run.py and
# attachment_stress.py.
_CHARS_PER_KB = 1024 * 0.85


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def build_one(size_kb: int, dest: Path, *, dry_run: bool) -> None:
    """Generate a single calibrated PDF and write it to *dest*.

    Args:
        size_kb:  Target size in KB.
        dest:     Absolute path to write the PDF (parent directory must exist).
        dry_run:  When True, generate the PDF in memory but do not write it.
                  Useful for verifying that the generation succeeds without
                  modifying the filesystem.
    """
    target_extracted = int(size_kb * _CHARS_PER_KB)
    pdf_bytes, raw_text, markers = generate_pdf(target_extracted)

    actual_kb = len(pdf_bytes) / 1024
    actual_chars = len(raw_text)
    print(
        f"  {dest.name:<22}  "
        f"target={size_kb:>3} KB  "
        f"actual={actual_kb:>6.1f} KB  "
        f"extracted_chars={actual_chars:>7,}  "
        f"markers={len(markers):>2}"
    )

    if not dry_run:
        dest.write_bytes(pdf_bytes)


def build_all(sizes: list[int], *, dry_run: bool) -> None:
    """Generate PDFs for each entry in *sizes*.

    Args:
        sizes:   List of KB sizes to generate (must be keys in :data:`PDF_TARGETS`).
        dry_run: Passed through to :func:`build_one`.
    """
    fixtures_dir = Path(__file__).parent
    if not dry_run:
        fixtures_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'DRY-RUN — ' if dry_run else ''}Generating {len(sizes)} PDF(s):")
    for kb in sorted(sizes):
        filename = PDF_TARGETS.get(kb)
        if filename is None:
            print(f"  [skip] {kb} KB — not in PDF_TARGETS; add an entry to build_pdfs.py")
            continue
        dest = fixtures_dir / filename
        build_one(kb, dest, dry_run=dry_run)

    if dry_run:
        print("\nDry-run complete. No files written.")
    else:
        print(f"\nPDFs written to: {fixtures_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evals.stress.fixtures.build_pdfs",
        description=(
            "Generate calibrated synthetic PDFs used by the stress-suite runner."
        ),
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(k) for k in sorted(PDF_TARGETS)),
        help=(
            "Comma-separated list of KB sizes to generate.  "
            f"Valid values: {sorted(PDF_TARGETS.keys())}.  "
            "Default: all four sizes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Generate PDFs in memory and print stats without writing any files.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    requested: list[int] = []
    for s in args.sizes.split(","):
        s = s.strip()
        if s:
            requested.append(int(s))

    build_all(requested, dry_run=args.dry_run)
