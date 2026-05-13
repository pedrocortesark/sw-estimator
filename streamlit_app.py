"""Punto de entrada para la aplicación Streamlit de estimación de software.

Actúa como cliente HTTP externo de la API FastAPI: nunca importa código
de src/ directamente. Toda la comunicación es vía HTTP usando httpx.
"""

import json

import httpx
import streamlit as st

from src.core.config import get_settings

API_BASE = get_settings().api_base_url


def _get_system_prompt() -> str:
    """Fetches the active system prompt from the API."""
    try:
        response = httpx.get(f"{API_BASE}/api/v1/context", timeout=10)
        response.raise_for_status()
        return response.json()["system_prompt"]
    except Exception as exc:
        return f"Error al cargar el system prompt: {exc}"


def http_stream_generator(transcript: str):
    """Calls POST /api/v1/estimate/stream and yields text chunks.

    When the final [DONE] event arrives, stores the EstimationResponse
    metadata in st.session_state.last_estimation_response.
    """
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{API_BASE}/api/v1/estimate/stream",
            json={"transcript": transcript},
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
        st.info("Este contexto se inyecta de forma invisible en cada petición para guiar al modelo.")

        with st.expander("Ver System Prompt Activo"):
            st.code(_get_system_prompt(), language="markdown")

        st.divider()
        st.header("📊 Última Ejecución")

        if "last_estimation_response" in st.session_state:
            resp = st.session_state.last_estimation_response
            st.metric("Modelo", resp["model_used"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Tokens", resp["usage"]["input_tokens"])
            with col2:
                st.metric("Output Tokens", resp["usage"]["output_tokens"])
            st.metric("Coste Estimado", f"${resp['usage']['cost_usd']:.6f}")
        else:
            st.caption("Aún no se ha generado ninguna estimación en esta sesión.")

    st.title("Generador de Estimaciones de Software")
    st.markdown("Pega aquí la transcripción de tu reunión para obtener una estimación usando CAG.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe o pega aquí la transcripción..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                estimation_text = st.write_stream(http_stream_generator(prompt))
                st.session_state.messages.append(
                    {"role": "assistant", "content": estimation_text}
                )
            except httpx.HTTPStatusError as e:
                st.error(f"Error HTTP {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"No se pudo conectar con la API ({API_BASE}): {e}")


if __name__ == "__main__":
    main()
