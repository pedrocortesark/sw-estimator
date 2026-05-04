# Plan de Implementación y Tutorial: Interfaz Conversacional con Streamlit

Este documento detalla el plan paso a paso para añadir una interfaz de usuario interactiva al Proyecto 1 (sw-estimator) utilizando Streamlit. 

El objetivo es implementar 3 niveles de funcionalidad y, al mismo tiempo, entender los conceptos subyacentes de Streamlit.

---

## 🛠️ Fase 0: Preparación y Dependencias

**Objetivo:** Preparar el entorno para ejecutar Streamlit.

1. **Instalar dependencias**: 
   Ejecutaremos `uv add streamlit` para añadir la librería al proyecto. Esto actualizará `pyproject.toml` y `uv.lock`.
2. **Crear el fichero base**: 
   Crearemos el archivo `streamlit_app.py` en la raíz del proyecto. Este será nuestro punto de entrada.
3. **Ejecución local**:
   Streamlit se ejecuta con su propio servidor. Para lanzarlo usaremos: `uv run streamlit run streamlit_app.py`.

---

## 🟢 Nivel 1: Chat Básico (Obligatorio)

**Objetivo:** Construir la interfaz conversacional y conectarla con el backend asíncrono existente.

### Conceptos Clave de Streamlit a aprender:
* **Recarga de página (Top-to-Bottom Execution):** Cada vez que un usuario interactúa con un widget (como enviar un mensaje), Streamlit vuelve a ejecutar todo el script `streamlit_app.py` desde la línea 1 hasta el final.
* **Manejo del Estado (`st.session_state`):** Como el script se recarga, las variables normales se reinician. Para guardar información entre recargas (como el historial del chat), usamos un diccionario especial llamado `st.session_state`.
* **Asincronía (`asyncio.run`):** Streamlit ejecuta el código de forma síncrona. Nuestro backend es asíncrono (FastAPI/Anthropic/OpenAI), por lo que deberemos crear un puente con la librería estándar `asyncio`.

### Pasos:
1. **Configuración Inicial:** Usar `st.set_page_config()` para definir el título y layout.
2. **Inicializar Estado:**
   ```python
   # Inicializar el historial de chat si no existe
   if "messages" not in st.session_state:
       st.session_state.messages = []
   ```
3. **Renderizar Historial:**
   ```python
   # Mostrar los mensajes anteriores
   for msg in st.session_state.messages:
       with st.chat_message(msg["role"]):
           st.markdown(msg["content"])
   ```
4. **Input y Llamada al Backend:**
   Usaremos `st.chat_input()` para recibir la transcripción, y llamaremos a nuestra función `generate_estimation()` envolviéndola en `asyncio.run()` ya que nuestro backend es asíncrono.

---

## 🟡 Nivel 2: Streaming (Obligatorio)

**Objetivo:** Conseguir que el modelo "escriba" la respuesta en tiempo real en la pantalla (token a token).

### Conceptos Clave a aprender:
* **Generadores Asíncronos (`AsyncGenerator`):** Las APIs de LLMs soportan devolver la respuesta poco a poco (Chunks). Para soportarlo, nuestro backend usará `yield` para ir devolviendo fragmentos.
* **Streamlit `st.write_stream` / Placeholders:** Streamlit ofrece formas de ir actualizando un bloque de texto sin recargar todo. Un "placeholder" (`st.empty()`) permite reservar un espacio y sobreescribirlo repetidamente con más texto.

### Pasos:
1. **Modificar Backend (`src/services/`):** 
   Añadiremos un nuevo método a nuestros providers (`BaseLLMProvider`, `AnthropicProvider`, `OpenAIProvider`) para soportar streaming real de la API.
   ```python
   async def stream_complete(self, system_prompt: str, user_message: str):
       # Ejemplo conceptual de cómo devolver tokens asíncronamente
       async with client.messages.stream(...) as stream:
           async for text in stream.text_stream:
               yield text
   ```
2. **Adaptar el Frontend (`streamlit_app.py`):**
   Debido a que `st.write_stream` en versiones de Streamlit más estándar (<= 1.36) espera un generador *síncrono*, a menudo se usa un patrón de placeholder para escribir iterativamente desde un bucle asíncrono, o se usa un adaptador síncrono.

   ```python
   # Patrón de Placeholder en Streamlit
   placeholder = st.empty()
   full_response = ""
   
   # Simulando recepción de tokens
   for chunk in generador_de_tokens:
       full_response += chunk
       placeholder.markdown(full_response + "▌") # ▌ Simula cursor parpadeando
       
   placeholder.markdown(full_response) # Resultado final sin el cursor
   ```

---

## 🔵 Nivel 3: Contexto CAG en la Interfaz (Opcional)

**Objetivo:** Exponer el funcionamiento interno (Prompt, Ejemplos, Métricas) en un panel lateral interactivo para mayor transparencia.

### Conceptos Clave a aprender:
* **Layouts (Sidebar, Columns, Expanders):** Streamlit permite estructurar la información usando barras laterales (`st.sidebar`), columnas (`st.columns`), y desplegables (`st.expander`) para no saturar la vista principal.
* **Métricas (`st.metric`):** Componentes visuales diseñados específicamente para mostrar KPIs (Key Performance Indicators).

### Pasos:
1. **Crear la Barra Lateral:**
   ```python
   with st.sidebar:
       st.header("⚙️ Configuración y Contexto")
   ```
2. **Mostrar el Contexto CAG:**
   Importaremos nuestro contexto estático desde `src.context.examples` y lo mostraremos dentro de elementos colapsables.
   ```python
   with st.expander("Ver System Prompt"):
       st.code(system_prompt_text, language="markdown")
   ```
3. **Métricas de Rendimiento:**
   Tras recibir la estimación, guardaremos el coste y los tokens generados en `st.session_state` y los renderizaremos en el sidebar.
   ```python
   col1, col2 = st.columns(2)
   with col1:
       st.metric(label="Coste", value=f"${coste:.4f}")
   with col2:
       st.metric(label="Tokens Output", value=tokens)
   ```
