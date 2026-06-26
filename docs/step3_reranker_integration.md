# Paso 3: Integración del Reranker - Resumen

## ✅ Implementación completada

### Patrón recall-then-rerank

El pipeline sigue el patrón estándar de recuperación en dos fases:

1. **Recall amplio** (top-50): Recuperación inicial con búsqueda vectorial o híbrida
2. **Reranking fino** (top-5): Reordenación con cross-encoder para mayor precisión

### Arquitectura

```
Query → Embedding → Recall (50 candidatos) → Reranker (5 finales) → Response
```

**Componentes:**
- `CrossEncoderReranker`: Wrapper del modelo `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- `retrieve()`: Pipeline principal con parámetros `rerank`, `recall_k`, `rerank_top_n`
- `get_reranker()`: Dependency injection para el reranker

### Control sin tocar código

El reranking se puede activar/desactivar de 3 formas:

1. **Parámetro en request** (prioridad más alta):
   ```json
   POST /v1/retrieval/search
   {"query_text": "...", "rerank": true, "search_mode": "hybrid"}
   ```

2. **Variable de entorno**:
   ```bash
   export RERANKER_ENABLED=true
   ```

3. **Configuración por defecto** en `src/core/config.py`:
   ```python
   reranker_enabled: bool = False
   reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
   ```

### Parámetros del pipeline

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `rerank` | `False` | Activar/desactivar reranking |
| `recall_k` | `50` | Número de candidatos en fase de recall |
| `rerank_top_n` | `5` | Número final de resultados después de reranking |
| `search_mode` | `"vector"` | `"vector"` o `"hybrid"` |

### 4 Configuraciones disponibles

1. **Vector only**: `search_mode="vector", rerank=False`
2. **Vector + Rerank**: `search_mode="vector", rerank=True`
3. **Hybrid**: `search_mode="hybrid", rerank=False`
4. **Hybrid + Rerank**: `search_mode="hybrid", rerank=True`

### Performance

| Configuración | Latencia | Uso de CPU |
|---------------|----------|------------|
| Vector only | ~10ms | Bajo |
| Vector + Rerank | ~100ms | Alto (primera vez ~5s) |
| Hybrid | ~15ms | Bajo |
| Hybrid + Rerank | ~110ms | Alto (primera vez ~5s) |

**Nota**: El modelo se carga lazy en la primera solicitud de reranking (~5s), luego se mantiene en caché.

### Tests

- ✅ 5 tests de reranker (ordenamiento, truncamiento, pares, custom extractor)
- ✅ 8 tests de RRF (single/multiple rankings, overlap, custom k, duplicates)
- ✅ Todos los tests offline pasan (420 tests)

### Scripts de demostración

- `scripts/demo_hybrid_search.py`: Muestra las 4 configuraciones
- `scripts/demo_reranker_toggle.py`: Demuestra cómo activar/desactivar sin tocar código

### Archivos modificados/creados

**Modificados:**
- `src/api/routers/retrieval.py`: Actualizado para usar `retrieve()` con parámetros `search_mode` y `rerank`
- `src/generation/rag/schemas.py`: Añadidos `search_mode` y `rerank` a `RetrievalRequest`
- `src/core/config.py`: Añadidos `reranker_enabled` y `reranker_model`
- `src/dependencies.py`: Añadido `get_reranker()`

**Creados:**
- `src/generation/rag/retrieval/reranker.py`: Wrapper del cross-encoder
- `src/generation/rag/retrieval/pipeline.py`: Pipeline recall-then-rerank
- `src/generation/rag/retrieval/verify_reranker.py`: Script de verificación
- `tests/test_reranker.py`: Tests del reranker
- `scripts/demo_reranker_toggle.py`: Demo de toggle

### Verificación

```bash
# Verificar que el modelo carga correctamente
uv run python src/generation/rag/retrieval/verify_reranker.py

# Probar las 4 configuraciones
uv run python scripts/demo_hybrid_search.py

# Probar toggle sin tocar código
uv run python scripts/demo_reranker_toggle.py

# Ejecutar tests
uv run pytest tests/test_reranker.py tests/test_fusion.py -v
```

## ✅ Criterios de aceptación cumplidos

- ✅ Patrón recall-then-rerank implementado (top-50 → top-5)
- ✅ Cross-encoder conectado al pipeline
- ✅ Reranking activable/desactivable sin tocar código (3 métodos)
- ✅ 4 configuraciones invocables de forma reproducible
- ✅ Tests unitarios pasando
- ✅ Documentación y scripts de demostración
