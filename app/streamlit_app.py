"""Streamlit front-end for the SW Estimator API — multi-turn session client.

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
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
TIMEOUT_SECONDS = 120
HOURS_PER_WEEK = 32  # matches system prompt rate


def _headers() -> dict[str, str]:
    """Return HTTP headers including the service token if configured."""
    headers: dict[str, str] = {}
    if SERVICE_TOKEN:
        headers["X-Service-Token"] = SERVICE_TOKEN
    return headers


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GuardrailError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class UpstreamError(Exception):
    def __init__(self, message: str = "LLM upstream error") -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _create_session() -> str:
    """POST /api/v1/sessions → returns the new session_id."""
    resp = httpx.post(f"{API_BASE}/api/v1/sessions", timeout=10, headers=_headers())
    resp.raise_for_status()
    return resp.json()["session_id"]


def _get_session_info(session_id: str) -> dict:
    """GET /api/v1/sessions/{id} → {session_id, turn_count, project_metadata}."""
    resp = httpx.get(f"{API_BASE}/api/v1/sessions/{session_id}", timeout=10, headers=_headers())
    resp.raise_for_status()
    return resp.json()


def _call_session_estimate(
    session_id: str,
    transcript: str,
    uploaded_files: list,
) -> dict:
    """POST /api/v1/sessions/{id}/estimate (multipart/form-data).

    Raises
    ------
    GuardrailError   on HTTP 400
    UpstreamError    on HTTP 502
    httpx.HTTPStatusError  on 404 / 413 / 415 / other
    """
    files = [
        ("attachments", (f.name, f.getvalue(), f.type))
        for f in (uploaded_files or [])
    ]
    resp = httpx.post(
        f"{API_BASE}/api/v1/sessions/{session_id}/estimate",
        data={"transcript": transcript},
        files=files or None,
        timeout=TIMEOUT_SECONDS,
        headers=_headers(),
    )

    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 400:
        body = resp.json()
        raise GuardrailError(
            reason=body.get("reason", "unknown"),
            message=body.get("detail", "Input rejected."),
        )
    if resp.status_code == 502:
        raise UpstreamError()

    resp.raise_for_status()
    raise Exception(f"Unexpected {resp.status_code}: {resp.text[:200]}")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Create a fresh backend session and reset all local state."""
    session_id = _create_session()
    st.session_state.session_id = session_id
    st.session_state.conversation = []  # list of {transcript, result} dicts
    st.session_state.project_metadata = {}
    st.session_state.turn_count = 0


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_project_metadata(metadata: dict) -> None:
    """Render the accumulated project metadata in the sidebar."""
    project_name = metadata.get("project_name")
    team_size = metadata.get("assumed_team_size")
    techs = metadata.get("mentioned_technologies") or []
    scope = metadata.get("agreed_scope")

    if not any([project_name, team_size, techs, scope]):
        st.caption("Sin contexto acumulado todavía.")
        return

    if project_name:
        st.markdown(f"**Proyecto:** {project_name}")
    if team_size:
        st.markdown(f"**Equipo estimado:** {team_size} personas")
    if techs:
        st.markdown(f"**Tecnologías:** {', '.join(techs)}")
    if scope:
        with st.expander("Alcance acordado", expanded=False):
            st.write(scope)


def _render_estimation_result(result: dict) -> None:
    """Render a single EstimationResponse dict."""
    estimation = result.get("estimation", {})
    summary = estimation.get("executive_summary", "")

    badges: list[str] = [f"`{result.get('prompt_version', 'v1')}`"]
    if result.get("cached"):
        badges.append("🟢 **CACHED**")
    st.caption("  ".join(badges))

    if summary.startswith("Out of scope:"):
        st.warning(summary)
        return

    col_dur, col_cost, col_conf = st.columns(3)
    with col_dur:
        st.metric("Duración", f"{estimation.get('duration_weeks', 0):.1f} sem")
    with col_cost:
        st.metric("Coste total", f"€ {estimation.get('total_cost_usd', 0):,.0f}")
    with col_conf:
        st.metric("Confianza", f"{estimation.get('confidence_pct', 0):.0f}%")

    st.markdown("**Resumen ejecutivo**")
    st.write(summary)

    phases = estimation.get("phases", [])
    if phases:
        rows = [
            {
                "Fase": p.get("name", ""),
                "Semanas": round(p.get("total_hours", 0) / HOURS_PER_WEEK, 1),
                "Coste €": f"€ {p.get('total_cost_usd', 0):,.0f}",
                "Tareas": ", ".join(t.get("name", "") for t in p.get("tasks", [])),
            }
            for p in phases
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    team = estimation.get("team_composition", [])
    if team:
        with st.expander("Composición del equipo"):
            st.dataframe(
                [
                    {
                        "Rol": m.get("role"),
                        "Nº": m.get("count"),
                        "Dedicación": m.get("dedication"),
                    }
                    for m in team
                ],
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="SW Estimator",
        page_icon="📐",
        layout="wide",
    )

    # --- Init session on first load ---
    if "session_id" not in st.session_state:
        try:
            _init_session_state()
        except Exception as exc:
            st.error(f"No se pudo conectar con el backend ({API_BASE}): {exc}")
            st.stop()

    # -----------------------------------------------------------------------
    # SIDEBAR — metadata + controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("📐 SW Estimator")
        st.caption(f"Sesión: `{st.session_state.session_id[:8]}…`")

        if st.button("🔄 Nueva conversación", use_container_width=True):
            try:
                _init_session_state()
                st.rerun()
            except Exception as exc:
                st.error(f"Error al crear sesión: {exc}")

        st.divider()

        st.subheader("🧠 Contexto del proyecto")
        st.caption(f"Turno {st.session_state.turn_count} · ventana deslizante activa")
        _render_project_metadata(st.session_state.project_metadata)

    # -----------------------------------------------------------------------
    # MAIN — conversation history
    # -----------------------------------------------------------------------
    st.header("Conversación")

    for turn in st.session_state.conversation:
        with st.chat_message("user"):
            st.write(turn["transcript"])
        with st.chat_message("assistant"):
            _render_estimation_result(turn["result"])

    # -----------------------------------------------------------------------
    # INPUT FORM — transcript + attachments
    # -----------------------------------------------------------------------
    st.divider()

    with st.form("turn_form", clear_on_submit=True):
        transcript = st.text_area(
            "Transcripción / descripción del proyecto",
            height=160,
            placeholder=(
                "Pega la transcripción de la reunión o describe el proyecto.\n"
                "Mínimo 20 caracteres. Puedes añadir más detalle en turnos sucesivos."
            ),
        )

        uploaded_files = st.file_uploader(
            "Adjuntos opcionales (PDF, DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Los documentos se procesan localmente — no se suben al proveedor LLM.",
        )

        submitted = st.form_submit_button(
            "Estimar →", type="primary", use_container_width=True
        )

    # -----------------------------------------------------------------------
    # PROCESS SUBMISSION
    # -----------------------------------------------------------------------
    if submitted:
        if len(transcript.strip()) < 20:
            st.error("La descripción debe tener al menos 20 caracteres.")
            st.stop()

        with st.spinner("Generando estimación… puede tardar hasta 2 minutos."):
            try:
                result = _call_session_estimate(
                    st.session_state.session_id,
                    transcript.strip(),
                    uploaded_files,
                )
            except GuardrailError as err:
                st.error(f"Input rechazado ({err.reason}): {err.message}")
                st.stop()
            except UpstreamError:
                st.error("El servicio LLM no está disponible. Inténtalo de nuevo.")
                st.stop()
            except httpx.TimeoutException:
                st.error("Timeout tras 120 s. Inténtalo de nuevo.")
                st.stop()
            except httpx.HTTPStatusError as err:
                st.error(f"Error {err.response.status_code}: {err.response.text[:300]}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error inesperado: {exc}")
                st.stop()

        # Persist turn locally for display
        st.session_state.conversation.append(
            {"transcript": transcript.strip(), "result": result}
        )

        # Refresh metadata from backend (best-effort)
        try:
            info = _get_session_info(st.session_state.session_id)
            st.session_state.project_metadata = info["project_metadata"]
            st.session_state.turn_count = info["turn_count"]
        except Exception:  # noqa: BLE001
            pass  # metadata panel simply won't update — not critical

        st.rerun()


if __name__ == "__main__":
    main()
