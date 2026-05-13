"""Punto de entrada para la aplicación Streamlit de estimación de software.

Actúa como cliente HTTP externo de la API FastAPI: nunca importa código
de src/ directamente. Toda la comunicación es vía HTTP usando httpx.
"""

import json

import httpx
import streamlit as st

from src.core.config import get_settings
from src.schemas.estimation import DetailLevel, OutputFormat, ProjectType

API_BASE = get_settings().api_base_url

# Human-readable labels for enum values shown in the form selects
PROJECT_TYPE_LABELS = {
    ProjectType.MOBILE_APP: "Mobile App",
    ProjectType.WEB_SAAS: "Web SaaS",
    ProjectType.INTERNAL_TOOL: "Internal Tool",
    ProjectType.DATA_PIPELINE: "Data Pipeline",
}
DETAIL_LEVEL_LABELS = {
    DetailLevel.SUMMARY: "Summary",
    DetailLevel.MEDIUM: "Medium",
    DetailLevel.DETAILED: "Detailed",
}
OUTPUT_FORMAT_LABELS = {
    OutputFormat.PHASES_TABLE: "Phases Table",
    OutputFormat.LINE_ITEMS: "Line Items",
    OutputFormat.NARRATIVE: "Narrative",
}


def _get_system_prompt() -> str:
    """Fetches the active system prompt from the API."""
    try:
        response = httpx.get(f"{API_BASE}/api/v1/context", timeout=10)
        response.raise_for_status()
        return response.json()["system_prompt"]
    except Exception as exc:
        return f"Error al cargar el system prompt: {exc}"


def http_stream_generator(payload: dict):
    """Calls POST /api/v1/estimate/stream and yields text chunks.

    When the final [DONE] event arrives, stores the EstimationResponse
    metadata in st.session_state.last_estimation_response.
    """
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{API_BASE}/api/v1/estimate/stream",
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]  # strip "data: " prefix
                if data.startswith("[DONE]"):
                    metadata = json.loads(data[6:])
                    st.session_state.last_estimation_response = metadata
                else:
                    yield data


def main():
    """Ejecución principal de la app."""
    st.set_page_config(
        page_title="SW Estimator",
        page_icon="🤖",
        layout="wide",
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Contexto CAG")
        st.info(
            "Este contexto se inyecta de forma invisible en cada petición para guiar al modelo."
        )

        with st.expander("Ver System Prompt Activo"):
            st.code(_get_system_prompt(), language="markdown")

        st.divider()
        st.header("📊 Última Ejecución")

        if "last_estimation_response" in st.session_state:
            resp = st.session_state.last_estimation_response
            st.metric("Modelo", resp["model_used"])
            st.metric("Prompt Version", resp["prompt_version"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Tokens", resp["usage"]["input_tokens"])
            with col2:
                st.metric("Output Tokens", resp["usage"]["output_tokens"])
            st.metric("Coste Estimado", f"${resp['usage']['cost_usd']:.6f}")
        else:
            st.caption("Aún no se ha generado ninguna estimación en esta sesión.")

    st.title("Generador de Estimaciones de Software")
    st.markdown("Rellena el formulario para obtener una estimación de esfuerzo usando CAG.")

    # --- FORM ---
    with st.form("estimation_form"):
        description = st.text_area(
            "Descripción del proyecto",
            placeholder="Describe el proyecto con suficiente detalle (mínimo 20 caracteres)...",
            height=180,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            project_type = st.selectbox(
                "Tipo de proyecto",
                options=list(PROJECT_TYPE_LABELS.keys()),
                format_func=lambda x: PROJECT_TYPE_LABELS[x],
            )
        with col2:
            detail_level = st.selectbox(
                "Nivel de detalle",
                options=list(DETAIL_LEVEL_LABELS.keys()),
                format_func=lambda x: DETAIL_LEVEL_LABELS[x],
            )
        with col3:
            output_format = st.selectbox(
                "Formato de salida",
                options=list(OUTPUT_FORMAT_LABELS.keys()),
                format_func=lambda x: OUTPUT_FORMAT_LABELS[x],
            )

        submitted = st.form_submit_button("Generar estimación", use_container_width=True)

    # --- RESULT ---
    if submitted:
        if len(description.strip()) < 20:
            st.warning("La descripción debe tener al menos 20 caracteres.")
        else:
            payload = {
                "description": description,
                "project_type": project_type.value,
                "detail_level": detail_level.value,
                "output_format": output_format.value,
            }

            st.divider()
            st.subheader("Estimación generada")
            try:
                st.write_stream(http_stream_generator(payload))
            except httpx.HTTPStatusError as e:
                st.error(f"Error HTTP {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"No se pudo conectar con la API ({API_BASE}): {e}")


main()
