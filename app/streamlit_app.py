"""Streamlit front-end for the SW Estimator API.

Communicates exclusively via HTTP with the FastAPI backend.
No src/ imports — this is a standalone client app.

Environment variables
---------------------
ESTIMATOR_API_URL   Base URL of the backend (default: http://localhost:8000)
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("ESTIMATOR_API_URL", "http://localhost:8000").rstrip("/")
ESTIMATE_URL = f"{API_BASE}/api/v1/estimate"
TIMEOUT_SECONDS = 120
HOURS_PER_WEEK = 32  # matches system prompt rate


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GuardrailError(Exception):
    """Raised when the backend returns HTTP 400 (input guardrail violation)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class UpstreamError(Exception):
    """Raised when the backend returns HTTP 502 (LLM unavailable)."""

    def __init__(self, message: str = "LLM upstream error") -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def call_estimator(payload: dict) -> dict:
    """POST /api/v1/estimate and return the parsed response body.

    Raises
    ------
    GuardrailError   on HTTP 400 (input rejected by guardrails)
    UpstreamError    on HTTP 502 (LLM failed to produce a valid response)
    httpx.TimeoutException   on network timeout (caller catches and shows UI msg)
    Exception        on any other non-200 status
    """
    response = httpx.post(ESTIMATE_URL, json=payload, timeout=TIMEOUT_SECONDS)

    if response.status_code == 200:
        return response.json()

    if response.status_code == 400:
        body = response.json()
        raise GuardrailError(
            reason=body.get("reason", "unknown"),
            message=body.get("detail", "Input rejected."),
        )

    if response.status_code == 502:
        raise UpstreamError()

    raise Exception(
        f"Unexpected response {response.status_code}: {response.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


def _render_metadata_badges(result: dict) -> None:
    """Render prompt_version and CACHED badge inline."""
    badges: list[str] = [f"`{result.get('prompt_version', 'v1')}`"]
    if result.get("cached"):
        badges.append("🟢 **CACHED**")
    st.markdown("  ".join(badges))


def _render_result(result: dict) -> None:
    """Render the estimation result returned by the API."""
    estimation = result.get("estimation", {})
    summary = estimation.get("executive_summary", "")

    _render_metadata_badges(result)

    # Out-of-scope shortcut: show warning and stop rendering
    if summary.startswith("Out of scope:"):
        st.warning(summary)
        return

    # ---- Three top-level metrics ----
    col_dur, col_cost, col_conf = st.columns(3)
    with col_dur:
        st.metric(
            "Duration",
            f"{estimation.get('duration_weeks', 0):.1f} weeks",
        )
    with col_cost:
        st.metric(
            "Total cost",
            f"€ {estimation.get('total_cost_usd', 0):,.0f}",
        )
    with col_conf:
        confidence = estimation.get("confidence_pct", 0)
        st.metric("Confidence", f"{confidence:.0f}%")

    st.markdown("**Executive summary**")
    st.write(summary)

    # ---- Phases table ----
    phases = estimation.get("phases", [])
    if phases:
        rows = []
        for phase in phases:
            phase_hours = phase.get("total_hours", 0)
            rows.append(
                {
                    "Phase": phase.get("name", ""),
                    "Weeks": round(phase_hours / HOURS_PER_WEEK, 1),
                    "Cost €": f"€ {phase.get('total_cost_usd', 0):,.0f}",
                    "Tasks": ", ".join(
                        t.get("name", "") for t in phase.get("tasks", [])
                    ),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ---- Team composition ----
    team = estimation.get("team_composition", [])
    if team:
        with st.expander("Team composition"):
            team_rows = [
                {
                    "Role": m.get("role", ""),
                    "Count": m.get("count", 1),
                    "Dedication": m.get("dedication", ""),
                }
                for m in team
            ]
            st.dataframe(team_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="SW Estimator",
        page_icon="📐",
        layout="wide",
    )

    st.title("📐 SW Estimator")
    st.caption(
        "Describe your project or paste a meeting transcript to get a structured estimate."
    )

    # ---- Estimation form ----
    with st.form("estimation_form"):
        transcript = st.text_area(
            "Description",
            height=180,
            placeholder=(
                "Paste a meeting transcript or write a project description here.\n"
                "E.g.: 'We need a B2B SaaS with role-based access, Slack notifications "
                "and an admin dashboard. The team has 2 engineers available part-time.'"
            ),
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            project_type = st.selectbox(
                "Project type",
                ["mobile_app", "web_saas", "internal_tool", "data_pipeline"],
            )
        with col_b:
            detail_level = st.selectbox(
                "Detail level",
                ["summary", "medium", "detailed"],
                index=1,
            )
        with col_c:
            output_format = st.selectbox(
                "Output format",
                ["phases_table", "line_items", "narrative"],
            )

        submitted = st.form_submit_button("Generate estimation", type="primary")

    # ---- Process submission ----
    if submitted:
        if len(transcript.strip()) < 20:
            st.error("Description must be at least 20 characters.")
            return

        payload = {
            "transcript": transcript.strip(),
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
        }

        with st.spinner("Generating estimation… this may take up to 2 minutes."):
            try:
                result = call_estimator(payload)
            except GuardrailError as err:
                st.error(f"Input rejected ({err.reason}): {err.message}")
                return
            except UpstreamError:
                st.error(
                    "The estimation service is unavailable. Try again later."
                )
                return
            except httpx.TimeoutException:
                st.error("Request timed out after 120s.")
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")
                return

        st.divider()
        _render_result(result)


if __name__ == "__main__":
    main()
