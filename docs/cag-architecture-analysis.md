# Análisis de Arquitectura CAG en sw-estimator

## Conclusiones del mapeo arquitectural

Después de estudiar los componentes teóricos de una arquitectura CAG y contrastarlos con el código del proyecto, estas son las conclusiones principales:

### Lo que el proyecto implementa bien

**Separación de responsabilidades clara.** Cada componente CAG tiene su propio lugar en el código: la fuente de conocimiento vive en `src/context/`, el constructor de prompts y el agregador en `src/services/llm_service.py`, y el servicio de llamada al LLM está desacoplado por proveedor (`openai_provider.py`, `anthropic_provider.py`).

**El patrón Strategy para los proveedores LLM** (`BaseLLMProvider` → `OpenAIProvider` / `AnthropicProvider`) es una buena decisión de diseño. Permite añadir un tercer proveedor (Gemini, Mistral, etc.) sin tocar nada del resto del sistema.

**Los ejemplos few-shot están bien formateados.** No son un JSON crudo: son resúmenes narrativos + tablas Markdown, que es exactamente el formato que un LLM procesa mejor. La decisión de qué incluir y qué omitir ya fue tomada en el preprocesamiento manual.

**Los tests no dependen de APIs externas.** El uso de `unittest.mock.patch` + `AsyncMock` hace que la suite sea rápida, determinista y ejecutable en CI sin coste.

---

## Áreas susceptibles de mejora

### 1. Preprocesamiento — actualmente manual y estático

**Situación actual:** `ESTIMATION_EXAMPLES` es una lista hardcodeada en Python. Añadir un nuevo ejemplo requiere editar código y hacer deploy.

**Mejora:** Externalizar los ejemplos a archivos `.md` o `.yaml` en `src/context/examples/`. El módulo los cargaría dinámicamente al arrancar. Esto permite que alguien del equipo añada o retire ejemplos sin tocar código.

**Mejora mayor:** Una interfaz de administración (un endpoint `POST /admin/examples`) para añadir ejemplos desde la propia API.

---

### 2. Constructor de prompts — no hay selección de ejemplos relevantes

**Situación actual:** `_build_system_prompt()` inyecta **todos** los ejemplos en cada petición, siempre.

**Problema futuro:** Con 10-20 ejemplos, el prompt superará fácilmente los 20.000 tokens. Coste elevado y posible degradación de calidad (los modelos procesan peor contextos muy largos).

**Mejora:** Selección semántica — dado el transcript del usuario, elegir los 2-3 ejemplos más similares (por tipo de proyecto, tecnologías, etc.). Esto es un paso hacia RAG, pero sin necesitar una base de datos vectorial completa: con `sentence-transformers` y similitud coseno en memoria es suficiente para un volumen pequeño.

---

### 3. Servicio LLM — no hay reintentos ni gestión de rate limits

**Situación actual:** Si la API de Anthropic/OpenAI devuelve un error 429 (rate limit) o un error de red transitorio, el error llega directamente al cliente como HTTP 500.

**Mejora:** Añadir retry con backoff exponencial usando la librería `tenacity`:

```python
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def complete(...): ...
```

---

### 4. Postprocesamiento — actualmente inexistente

**Situación actual:** La respuesta del LLM se devuelve tal cual, sin ninguna validación. Si el modelo alucina y devuelve "10 horas de frontend" con "coste: 50.000€", el cliente recibe ese texto inconsistente.

**Mejoras posibles (de menor a mayor complejidad):**

- **Validación estructural básica:** verificar que la respuesta contiene secciones Markdown esperadas (al menos un `##` y una tabla `|`).
- **Extracción de datos:** parsear las horas totales del Markdown y exponerlas como campo numérico en `EstimationResponse` (además del texto).
- **Validación de coherencia:** detectar inconsistencias entre horas declaradas y costes si se añade ese campo.

---

### 5. Observabilidad — logging sin métricas

**Situación actual:** Loguru registra eventos pero no hay métricas de negocio.

**Mejora:** Añadir a cada llamada LLM el conteo de tokens consumidos (ambas APIs lo devuelven en la respuesta) y loguearlo. Esto permite monitorizar el coste por request a lo largo del tiempo.

---

## Registro de análisis

| Fecha | Tema | Archivo |
|---|---|---|
| 2026-04-29 | Mapeo de componentes CAG en el proyecto | este archivo |
