"""Punto de entrada para la aplicación Streamlit de estimación de software."""

import asyncio
import streamlit as st

# Importamos la función asíncrona de generación de estimaciones del backend
from src.services.llm_service import stream_estimation, _build_system_prompt


def sync_stream_generator(transcript: str):
    """Convierte el generador asíncrono en síncrono para st.write_stream."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async_gen = stream_estimation(transcript)
    try:
        while True:
            chunk = loop.run_until_complete(anext(async_gen))
            if isinstance(chunk, str):
                yield chunk
            else:
                # Es el EstimationResponse final con las métricas
                st.session_state.last_estimation_response = chunk
    except StopAsyncIteration:
        pass
    finally:
        loop.close()


def main():
    """Ejecución principal de la app."""
    # 1. Configuración de la página
    st.set_page_config(
        page_title="SW Estimator",
        page_icon="🤖",
        layout="wide"  # Usamos 'wide' para que el sidebar tenga más espacio
    )

    # --- INICIO NIVEL 3: BARRA LATERAL (SIDEBAR) ---
    with st.sidebar:
        st.header("⚙️ Contexto CAG")
        st.info("Este contexto se inyecta de forma invisible en cada petición para guiar al modelo.")
        
        with st.expander("Ver System Prompt Activo"):
            st.code(_build_system_prompt(), language="markdown")
            
        st.divider()
        st.header("📊 Última Ejecución")
        
        if "last_estimation_response" in st.session_state:
            resp = st.session_state.last_estimation_response
            st.metric("Modelo", resp.model_used)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Tokens", resp.usage.input_tokens)
            with col2:
                st.metric("Output Tokens", resp.usage.output_tokens)
            st.metric("Coste Estimado", f"${resp.usage.cost_usd:.6f}")
        else:
            st.caption("Aún no se ha generado ninguna estimación en esta sesión.")
    # --- FIN NIVEL 3 ---

    st.title("Generador de Estimaciones de Software")
    st.markdown("Pega aquí la transcripción de tu reunión para obtener una estimación usando CAG.")

    # 2. Inicializar el estado de la sesión (st.session_state)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. Renderizar el historial del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Entrada de usuario y llamada al backend
    if prompt := st.chat_input("Escribe o pega aquí la transcripción..."):
        # a) Añadimos el mensaje del usuario al historial y lo mostramos
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # b) Creamos la burbuja para el asistente
        with st.chat_message("assistant"):
            try:
                # c) Pasamos el generador síncrono a st.write_stream
                # Streamlit irá sacando los tokens e imprimiéndolos con efecto máquina de escribir
                estimation_text = st.write_stream(sync_stream_generator(prompt))
                
                # d) Añadimos la respuesta completa al historial para futuras recargas
                st.session_state.messages.append({"role": "assistant", "content": estimation_text})
                    
            except Exception as e:
                st.error(f"Ocurrió un error al generar la estimación: {str(e)}")

if __name__ == "__main__":
    main()
