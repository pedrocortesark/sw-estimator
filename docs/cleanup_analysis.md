# Informe de Duplicados y Limpieza

## Resumen Ejecutivo

Se identificaron **3 categorías principales de duplicados** en el proyecto como consecuencia del trabajo de las sesiones 9 y 10:

1. **Configuración duplicada** (2 archivos)
2. **LLM Wrapper duplicado** (2 archivos idénticos)
3. **Módulos RAG paralelos** (no son duplicados exactos, sino evoluciones)

---

## 1. Configuración Duplicada

### Archivos afectados:
- `src/config.py` (100 líneas)
- `src/core/config.py` (89 líneas)

### Estado actual:
Ambos archivos definen una clase `Settings` pero con convenciones diferentes:

| Aspecto | `src/config.py` | `src/core/config.py` |
|---------|----------------|---------------------|
| Convención de nombres | MAYÚSCULAS (`OPENAI_API_KEY`) | minúsculas (`openai_api_key`) |
| Campos | 35+ campos (Sessions 2-6) | 15 campos (Sessions 1-8, 10) |
| Usado por | Módulos nuevos de Antonio (api/, generation/) | Módulos originales (main.py, routers/, services/) |

### Problema:
- Dos sistemas de configuración coexisten
- Algunos módulos usan uno, otros usan el otro
- Riesgo de inconsistencia si se cambia una variable en solo un archivo

### Recomendación:
**Consolidar en un solo archivo de configuración.** Opciones:

**Opción A (Recomendada)**: Migrar todo a `src/core/config.py`
- Ventaja: Sigue la convención moderna (minúsculas)
- Ventaja: Ya tiene los campos de Session 10
- Desventaja: Requiere actualizar 6+ archivos que usan `src/config.py`

**Opción B**: Migrar todo a `src/config.py`
- Ventaja: Tiene todos los campos de Sessions 2-6
- Desventaja: Usa convención antigua (MAYÚSCULAS)
- Desventaja: Requiere actualizar 10+ archivos que usan `src/core/config.py`

---

## 2. LLM Wrapper Duplicado

### Archivos afectados:
- `src/llm/wrapper.py` (9.5 KB)
- `src/services/llm_wrapper.py` (9.5 KB)

### Estado actual:
**Son idénticos** (diff sin diferencias).

### Uso actual:
- `src/llm/wrapper.py` → Usado por módulos nuevos:
  - `src/api/config.py`
  - `src/domain/estimation_service.py`
  - `src/generation/conversation/metadata_extractor.py`

- `src/services/llm_wrapper.py` → Usado por servicios originales:
  - `src/services/estimation.py`
  - `src/services/llm_service.py`
  - `src/services/summarizer.py`

### Problema:
- Código duplicado al 100%
- Mantenimiento doble innecesario
- Riesgo de divergencia futura

### Recomendación:
**Eliminar `src/llm/wrapper.py` y usar solo `src/services/llm_wrapper.py`.**

Pasos:
1. Actualizar 3 archivos que importan de `src/llm/wrapper.py`:
   - `src/api/config.py`
   - `src/domain/estimation_service.py`
   - `src/generation/conversation/metadata_extractor.py`
2. Eliminar `src/llm/wrapper.py`
3. Eliminar directorio `src/llm/` si queda vacío

---

## 3. Módulos RAG Paralelos

### Directorios afectados:
- `src/rag/` (Session 8 - básico)
- `src/generation/rag/` (Session 9/10 - extendido)

### Estado actual:
**NO son duplicados exactos**, sino evoluciones del mismo concepto:

| Componente | `src/rag/` | `src/generation/rag/` |
|------------|-----------|----------------------|
| **Propósito** | Búsqueda semántica básica (Session 8) | Pipeline completo RAG (Session 9/10) |
| **Embedder** | ✅ Básico | ✅ Extendido (mismos métodos) |
| **Repository** | ✅ `search()` básico | ✅ `search()`, `search_filtered()`, `search_lexical()` |
| **Retriever** | ✅ `SemanticRetriever` | ✅ `SemanticRetriever` + `search_chunks()` |
| **Schemas** | ✅ Modelos Session 8 | ✅ Modelos Session 8 + 9 (236 líneas adicionales) |
| **Chunking** | ✅ `structural.py` básico | ✅ `structural.py` + `strategies/` (7 estrategias) |
| **Retrieval** | ❌ No tiene | ✅ `retrieval/` (fusion, reranker, pipeline) |
| **Usado por** | `src/routers/embeddings.py`, `src/routers/search.py` | `src/api/routers/retrieval.py`, pipeline completo |

### Análisis de diferencias:

#### `embedder.py`
- **Diferencia**: Solo imports y un comentario
- **Conclusión**: Son funcionalmente idénticos

#### `repository.py`
- **Diferencia**: `src/generation/rag/` tiene 140 líneas adicionales:
  - `search_filtered()` con filtros estructurales
  - `search_lexical()` para búsqueda híbrida
  - Uso de `HALFVEC` para optimización
- **Conclusión**: `src/generation/rag/` es un superconjunto

#### `retriever.py`
- **Diferencia**: `src/generation/rag/` tiene 85 líneas adicionales:
  - Función `search_chunks()` para Session 9
  - Soporte para filtros estructurales
- **Conclusión**: `src/generation/rag/` es un superconjunto

#### `schemas.py`
- **Diferencia**: `src/generation/rag/` tiene 236 líneas adicionales:
  - Modelos de Session 9: `EstimationQuery`, `RetrievedChunk`, `RetrievalResult`, etc.
  - Modelos de Session 9: `SourceCitation`, `Assumption`, `TaskItem`, `WorkModule`, `Estimate`
  - Modelos HTTP: `RetrievalRequest`, `EstimateRequest`, etc.
- **Conclusión**: `src/generation/rag/` es un superconjunto

#### `structural.py`
- **Diferencia**: `src/generation/rag/` tiene funciones adicionales:
  - `serialize_budget()` para estrategias avanzadas
  - Soporte para campo `module` en componentes
- **Conclusión**: `src/generation/rag/` es un superconjunto

### Problema:
- Dos módulos RAG coexisten con funcionalidad superpuesta
- `src/rag/` es un subconjunto de `src/generation/rag/`
- Ambos están en uso activo por diferentes endpoints

### Recomendación:
**Migrar completamente a `src/generation/rag/` y eliminar `src/rag/`.**

Pasos:
1. Actualizar 2 archivos que usan `src/rag/`:
   - `src/routers/embeddings.py` → Cambiar imports a `src/generation/rag/`
   - `src/routers/search.py` → Cambiar imports a `src/generation/rag/`
2. Verificar que `src/dependencies.py` use los providers correctos
3. Eliminar directorio `src/rag/` completo
4. Ejecutar tests para validar

**Riesgo**: Bajo, porque `src/generation/rag/` es un superconjunto funcional.

---

## 4. Otros Archivos con Nombres Duplicados

Se encontraron archivos con el mismo nombre en diferentes directorios:

| Archivo | Ubicaciones | ¿Duplicado? |
|---------|-------------|-------------|
| `__init__.py` | Múltiples | ❌ Normal (paquetes Python) |
| `base.py` | `src/rag/chunking/`, `src/generation/rag/chunking/` | ⚠️ Ver arriba |
| `config.py` | `src/`, `src/core/`, `src/api/` | ⚠️ Ver sección 1 |
| `embedder.py` | `src/rag/embedding/`, `src/generation/rag/embedding/` | ⚠️ Ver sección 3 |
| `estimation.py` | `src/services/`, `src/routers/`, `src/schemas/` | ❌ Diferentes responsabilidades |
| `ingest_service.py` | `src/rag/`, `src/generation/rag/` | ⚠️ Ver sección 3 |
| `loader.py` | `src/ingest/catalog/`, `src/prompts/` | ❌ Diferentes responsabilidades |
| `metadata_extractor.py` | `src/services/`, `src/generation/conversation/` | ❌ Diferentes responsabilidades |
| `models.py` | `src/persistence/`, `src/rag/store/`, `src/generation/rag/store/` | ⚠️ Ver sección 3 |
| `policy.py` | `src/ingest/cleaning/`, `src/generation/conversation/compression/` | ❌ Diferentes responsabilidades |
| `repository.py` | `src/rag/store/`, `src/generation/rag/store/` | ⚠️ Ver sección 3 |
| `retriever.py` | `src/rag/`, `src/generation/rag/` | ⚠️ Ver sección 3 |
| `schemas.py` | `src/rag/`, `src/generation/rag/`, `src/schemas/` | ⚠️ Ver sección 3 |
| `semantic.py` | `src/cache/`, `src/generation/cag/` | ❌ Diferentes responsabilidades |
| `sessions.py` | `src/services/`, `src/routers/` | ❌ Diferentes responsabilidades |
| `structural.py` | `src/rag/chunking/`, `src/generation/rag/chunking/` | ⚠️ Ver sección 3 |
| `summarizer.py` | `src/services/`, `src/generation/conversation/compression/` | ❌ Diferentes responsabilidades |
| `tier_resolver.py` | `src/services/`, `src/generation/conversation/` | ❌ Diferentes responsabilidades |

---

## Plan de Limpieza Recomendado

### Prioridad Alta (duplicados exactos):

#### Tarea 1: Eliminar LLM Wrapper duplicado
- **Archivos**: `src/llm/wrapper.py`
- **Acción**: Eliminar y actualizar imports
- **Riesgo**: Bajo
- **Tiempo estimado**: 15 minutos

#### Tarea 2: Consolidar configuración
- **Archivos**: `src/config.py` o `src/core/config.py` (elegir uno)
- **Acción**: Migrar todo a un solo archivo
- **Riesgo**: Medio (requiere actualizar 10+ archivos)
- **Tiempo estimado**: 30 minutos

### Prioridad Media (módulos RAG paralelos):

#### Tarea 3: Migrar a `src/generation/rag/`
- **Archivos**: `src/rag/` completo
- **Acción**: Actualizar imports y eliminar `src/rag/`
- **Riesgo**: Medio-bajo (es un superconjunto funcional)
- **Tiempo estimado**: 45 minutos
- **Requiere**: Testing exhaustivo de endpoints afectados

---

## Impacto de la Limpieza

### Beneficios:
- ✅ Eliminar 2 archivos duplicados exactos
- ✅ Reducir complejidad del código base
- ✅ Facilitar mantenimiento futuro
- ✅ Evitar divergencias accidentales

### Riesgos:
- ⚠️ Cambios en imports pueden romper funcionalidad
- ⚠️ Requiere testing exhaustivo después de cada cambio
- ⚠️ Posibles dependencias externas no documentadas

### Recomendación final:
Ejecutar limpieza en **orden de prioridad**:
1. Tarea 1 (LLM Wrapper) - bajo riesgo, alto impacto
2. Tarea 2 (Configuración) - riesgo medio, alto impacto
3. Tarea 3 (RAG modules) - riesgo medio-bajo, impacto medio

**Total estimado**: 1.5 horas de trabajo + testing
