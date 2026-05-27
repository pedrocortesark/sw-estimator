"""Heuristic metadata extractor — derives ProjectMetadata from a structured EstimationResult.

Why heuristic over a second LLM call
--------------------------------------
The LLM already returns a fully structured ``EstimationResult`` via Instructor.
That object contains exactly the facts we need:

* ``team_composition`` → ``assumed_team_size`` (sum of headcounts, no parsing needed)
* ``executive_summary`` → ``agreed_scope`` (the LLM wrote it; it is already a scope summary)
* task/phase names + executive_summary → ``mentioned_technologies`` (regex against a
  curated vocabulary applied to text we already have in memory)
* the original transcript → ``project_name`` (pattern matching on common naming phrases)

A second LLM call would add ~300 ms latency and ~$0.001 per turn for information that
is already available in structured form.  The one genuine weakness of the heuristic
approach is technology detection: it can only recognise technologies that appear in
the vocabulary list below.  That list is easy to extend as new stacks appear in
client briefs.
"""

from __future__ import annotations

import re

from src.schemas.estimation import EstimationResult
from src.services.sessions import ProjectMetadata

# ---------------------------------------------------------------------------
# Technology vocabulary — extend freely; matching is case-insensitive
# ---------------------------------------------------------------------------

_TECH_VOCAB: list[str] = [
    # Languages
    "Python",
    "TypeScript",
    "JavaScript",
    "Java",
    "Go",
    "Rust",
    "Ruby",
    "PHP",
    "Swift",
    "Kotlin",
    "C#",
    ".NET",
    "Scala",
    # Frontend
    "React",
    "Vue",
    "Angular",
    "Next.js",
    "Nuxt",
    "Svelte",
    "Tailwind",
    # Backend / API
    "FastAPI",
    "Django",
    "Flask",
    "Express",
    "NestJS",
    "Spring Boot",
    "Rails",
    "GraphQL",
    "REST",
    "gRPC",
    "WebSocket",
    # Mobile
    "React Native",
    "Flutter",
    "iOS",
    "Android",
    # Data & AI
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "Kafka",
    "Spark",
    "dbt",
    "Airflow",
    "Pandas",
    "NumPy",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "LangChain",
    "OpenAI",
    "Anthropic",
    # Cloud & Infra
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Ansible",
    "S3",
    "Lambda",
    "EC2",
    "RDS",
    "BigQuery",
    "Snowflake",
    # Auth & security
    "OAuth",
    "JWT",
    "Auth0",
    "Keycloak",
    # Other
    "Stripe",
    "Twilio",
    "SendGrid",
    "Firebase",
    "Supabase",
    "Hasura",
]

# Pre-compiled pattern: matches any vocabulary word (word-boundary aware)
_TECH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _TECH_VOCAB) + r")\b",
    re.IGNORECASE,
)

# Patterns that suggest a project name in the transcript
_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:project|app|platform|system|tool|product)\s+(?:called|named|is)\s+[\"']?([A-Z][A-Za-z0-9 _-]{1,40})[\"']?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:build|develop|create)\s+[\"']([A-Z][A-Za-z0-9 _-]{1,40})[\"']",
        re.IGNORECASE,
    ),
    re.compile(
        r"the\s+([A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+)?)\s+(?:project|platform|app|system)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_from_result(
    transcript: str,
    result: EstimationResult,
    existing: ProjectMetadata,
) -> ProjectMetadata:
    """Return an updated ``ProjectMetadata`` by merging new facts into *existing*.

    Fields already populated in *existing* are **not overwritten** (except
    ``mentioned_technologies``, which is always extended).  This preserves
    facts agreed upon in earlier turns even if a later transcript does not
    repeat them.

    Args:
        transcript: The raw user transcript from this turn (used for name extraction).
        result:     The structured LLM response for this turn.
        existing:   Current session metadata (may have empty fields on first call).

    Returns:
        A new ``ProjectMetadata`` instance with merged data.
    """
    # --- project_name: only fill if not yet known ---
    project_name = existing.project_name or _extract_project_name(transcript)

    # --- assumed_team_size: derive from team_composition on every turn ---
    assumed_team_size = (
        sum(m.count for m in result.team_composition) or existing.assumed_team_size
    )

    # --- mentioned_technologies: cumulative across all turns ---
    new_techs = _extract_technologies(
        result.executive_summary
        + " "
        + " ".join(t.name for phase in result.phases for t in phase.tasks)
        + " "
        + transcript
    )
    merged_techs = _merge_unique(existing.mentioned_technologies, new_techs)

    # --- agreed_scope: always refresh with the latest executive summary ---
    agreed_scope = result.executive_summary

    return ProjectMetadata(
        project_name=project_name,
        assumed_team_size=assumed_team_size,
        mentioned_technologies=merged_techs,
        agreed_scope=agreed_scope,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_project_name(text: str) -> str | None:
    for pattern in _NAME_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_technologies(text: str) -> list[str]:
    """Return unique technology names found in *text*, preserving canonical casing."""
    found: dict[str, str] = {}  # lowercase → canonical
    for m in _TECH_PATTERN.finditer(text):
        found[m.group(0).lower()] = m.group(0)
    # Return canonical forms sorted for deterministic output
    return sorted(found.values(), key=str.lower)


def _merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """Merge two lists keeping unique items (case-insensitive dedup)."""
    seen: dict[str, str] = {t.lower(): t for t in existing}
    for t in new:
        seen.setdefault(t.lower(), t)
    return sorted(seen.values(), key=str.lower)
