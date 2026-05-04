"""Punto de entrada para la aplicación Streamlit de estimación de software."""

import asyncio
import streamlit as st

# Importamos la función asíncrona de generación de estimaciones del backend
from src.services.llm_service import generate_estimation


def main():
    """Ejecución principal de la app."""
    # 1. Configuración de la página
    st.set_page_config(
        page_title="SW Estimator",
        page_icon="🤖",
        layout="centered"
    )

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

        # b) Mostramos un indicador de "pensando"
        with st.spinner("Generando estimación... esto puede tardar un poco."):
            try:
                # c) Llamamos a nuestro backend. 
                response = asyncio.run(generate_estimation(transcript=prompt))
                
                # Extraemos el texto de la estimación del modelo de respuesta
                estimation_text = response.estimation
                
                # d) Añadimos la respuesta del asistente al historial y la mostramos
                st.session_state.messages.append({"role": "assistant", "content": estimation_text})
                with st.chat_message("assistant"):
                    st.markdown(estimation_text)
                    
            except Exception as e:
                st.error(f"Ocurrió un error al generar la estimación: {str(e)}")

if __name__ == "__main__":
    main()
