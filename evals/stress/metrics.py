"""Evaluation metrics for stress and session-level scenarios.

Module choice: ``evals/stress/metrics.py`` rather than ``evals/metrics.py``
-----------------------------------------------------------------------------
``evals/metrics.py`` is scoped to golden-dataset quality checks — it operates
on ``EstimationResponse`` dicts and knows about Pydantic schema validation,
cost bounds tables, and required-keyword lists.

The three metrics here operate on a **different observation shape**:

* :class:`LatencyBudgetMetric` and :class:`CostBudgetMetric` receive a
  per-turn *observation* — a dict (or attribute-accessible object) produced
  by the ``turn_observed`` log event or by the ``attachment_stress`` runner.

* :class:`MemoryDriftMetric` receives a *session snapshot* — a live
  :class:`~src.services.sessions.Session` object whose ``accumulated_summary``,
  ``anchors``, and ``metadata`` fields are inspected for the presence of a
  declared fact.

Co-locating these metrics with ``evals/stress/scenarios.py`` and
``evals/stress/attachment_stress.py`` — their primary consumers — keeps the
golden-dataset module free of session-layer imports and avoids coupling two
separate evaluation concerns in one file.

Shared contract
---------------
All three metrics return :class:`~evals.metrics.MetricResult` so that every
metric in the framework — regardless of module — has the same
``(name, score, passed, details)`` shape.

Determinism
-----------
No embeddings, no LLM-as-judge, no probabilistic thresholds.
Every evaluation is a **case-insensitive exact substring match** or a
numeric comparison against a declared budget.  Given the same inputs the
result is always identical.
"""

from __future__ import annotations

from typing import Any

from evals.metrics import MetricResult


# ---------------------------------------------------------------------------
# Internal helper — duck-typed field access
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    """Return ``obj[key]`` (dict) or ``getattr(obj, key)`` (object), or None."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# ---------------------------------------------------------------------------
# LatencyBudgetMetric
# ---------------------------------------------------------------------------


class LatencyBudgetMetric:
    """Binary metric: latency_ms ≤ budget_ms.

    Scores 1.0 when the observed wall-clock latency is within the declared
    budget, 0.0 otherwise.  Designed for per-turn observations emitted by
    ``turn_observed`` log events or by the attachment-stress runner.

    Args:
        budget_ms: Maximum acceptable latency in milliseconds (inclusive).

    Example::

        metric = LatencyBudgetMetric(budget_ms=3000)
        result = metric.evaluate({"latency_ms": 2400.0})
        assert result.passed
    """

    name = "latency_budget"

    def __init__(self, budget_ms: int) -> None:
        if budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive, got {budget_ms}")
        self.budget_ms = budget_ms

    def evaluate(self, observation: Any) -> MetricResult:
        """Evaluate a single turn observation against the latency budget.

        Args:
            observation: Dict or object with a ``latency_ms`` field.
                         ``None`` values produce a failed result with an
                         explanatory message rather than a hard error.

        Returns:
            :class:`~evals.metrics.MetricResult` with score 1.0 or 0.0.
        """
        raw = _get(observation, "latency_ms")

        if raw is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="latency_ms not found in observation",
            )

        latency = float(raw)
        passed = latency <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"latency {latency:.1f} ms "
                f"{'≤' if passed else '>'} budget {self.budget_ms} ms"
            ),
        )


# ---------------------------------------------------------------------------
# CostBudgetMetric
# ---------------------------------------------------------------------------


class CostBudgetMetric:
    """Binary metric: cost_usd ≤ budget_usd.

    Scores 1.0 when the observed API cost is within the declared budget,
    0.0 otherwise.

    Args:
        budget_usd: Maximum acceptable cost in USD (inclusive).

    Example::

        metric = CostBudgetMetric(budget_usd=0.005)
        result = metric.evaluate({"cost_usd": 0.0031})
        assert result.passed
    """

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        if budget_usd < 0:
            raise ValueError(f"budget_usd must be non-negative, got {budget_usd}")
        self.budget_usd = budget_usd

    def evaluate(self, observation: Any) -> MetricResult:
        """Evaluate a single turn observation against the cost budget.

        Args:
            observation: Dict or object with a ``cost_usd`` field.

        Returns:
            :class:`~evals.metrics.MetricResult` with score 1.0 or 0.0.
        """
        raw = _get(observation, "cost_usd")

        if raw is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="cost_usd not found in observation",
            )

        cost = float(raw)
        passed = cost <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"cost ${cost:.6f} "
                f"{'≤' if passed else '>'} budget ${self.budget_usd:.6f}"
            ),
        )


# ---------------------------------------------------------------------------
# MemoryDriftMetric
# ---------------------------------------------------------------------------

_VALID_WHERE: frozenset[str] = frozenset({"summary", "anchors", "metadata"})


def _build_session_corpus(session: Any, where: list[str]) -> str:
    """Concatenate the requested session fields into a single searchable string.

    Supported *where* values:

    * ``"summary"``  — ``session.accumulated_summary``
    * ``"anchors"``  — every string in ``session.anchors`` joined with spaces
    * ``"metadata"`` — ``project_name``, ``assumed_team_size`` (as str),
      every item in ``mentioned_technologies``, and ``agreed_scope``
      taken from ``session.metadata``

    Unknown keys are silently skipped.

    Args:
        session: A :class:`~src.services.sessions.Session` instance or any
                 object (or dict) exposing the same attributes.
        where:   Subset of ``{"summary", "anchors", "metadata"}``.

    Returns:
        Single lowercase string corpus ready for substring search.
    """
    parts: list[str] = []

    if "summary" in where:
        summary = _get(session, "accumulated_summary") or ""
        parts.append(str(summary))

    if "anchors" in where:
        anchors = _get(session, "anchors") or []
        parts.extend(str(a) for a in anchors)

    if "metadata" in where:
        meta = _get(session, "metadata")
        if meta is not None:
            for field in ("project_name", "assumed_team_size", "agreed_scope"):
                val = _get(meta, field)
                if val is not None:
                    parts.append(str(val))
            techs = _get(meta, "mentioned_technologies") or []
            parts.extend(str(t) for t in techs)

    return " ".join(parts).lower()


class MemoryDriftMetric:
    """Binary metric: a fact introduced at turn *k* is still present at turn *N*.

    Scores 1.0 when *fact* (declared at an earlier turn) is found as a
    case-insensitive substring somewhere in the session's accumulated
    memory — accumulated summary, anchors list, or ProjectMetadata fields —
    at the time of evaluation (turn N > k).  Scores 0.0 if the fact has
    drifted out of the session state ("memory drift").

    The metric is purely deterministic: no embeddings, no LLM-as-judge.
    Only exact (case-insensitive) substring matching is used.

    Args:
        fact:  The string to look for (e.g. ``"FastAPI"``, ``"team_size:4"``).
        where: Which session fields to search.  Defaults to all three:
               ``["summary", "anchors", "metadata"]``.

    Example::

        metric = MemoryDriftMetric(fact="FastAPI", where=["anchors", "metadata"])
        result = metric.evaluate(session)   # session is a Session object
        assert result.passed

    Raises:
        ValueError: If *where* contains unknown field names.
    """

    name = "memory_drift"

    def __init__(
        self,
        fact: str,
        where: list[str] | None = None,
    ) -> None:
        self.fact = fact
        self.where: list[str] = (
            where if where is not None else ["summary", "anchors", "metadata"]
        )
        unknown = set(self.where) - _VALID_WHERE
        if unknown:
            raise ValueError(
                f"Unknown 'where' values: {sorted(unknown)}. "
                f"Valid options: {sorted(_VALID_WHERE)}"
            )

    def evaluate(self, session_snapshot: Any) -> MetricResult:
        """Check whether *fact* is still present in the session at turn N.

        Args:
            session_snapshot: A :class:`~src.services.sessions.Session`
                              instance (or any duck-typed equivalent) captured
                              after the turn-N estimation call completes.

        Returns:
            :class:`~evals.metrics.MetricResult` with score 1.0 if the fact
            is found, 0.0 if it has drifted out of the observed fields.
        """
        corpus = _build_session_corpus(session_snapshot, self.where)
        fact_lower = self.fact.lower()
        passed = fact_lower in corpus

        if passed:
            details = f"Fact '{self.fact}' found in {self.where}"
        else:
            details = f"Fact '{self.fact}' NOT found in {self.where} (memory drift)"

        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=details,
        )
