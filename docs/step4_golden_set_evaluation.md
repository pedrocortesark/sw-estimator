# Paso 4: Golden Set y Medición - Análisis de Resultados

## Golden Set

Se construyó un conjunto de 5 consultas representativas del dominio, cada una con sus presupuestos relevantes anotados manualmente:

| Query ID | Descripción | Presupuestos Relevantes |
|----------|-------------|------------------------|
| Q1 | Mobile banking con OAuth y cumplimiento normativo | BUD-2024-001, BUD-2024-002, BUD-2024-003 |
| Q2 | Plataforma e-commerce con catálogo y checkout | BUD-2024-005, BUD-2024-006, BUD-2024-007, BUD-2024-017 |
| Q3 | Plataforma de telemedicina con videoconsultas | BUD-2024-009, BUD-2024-010, BUD-2024-011, BUD-2024-012 |
| Q4 | Sistema IoT industrial para mantenimiento predictivo | BUD-2024-013, BUD-2024-014, BUD-2024-015 |
| Q5 | Pasarela de pagos en tiempo real con detección de fraude | BUD-2024-003, BUD-2024-004, BUD-2024-016 |

## Resultados de Medición

### Tabla Comparativa Completa

| Configuración | Query | Precision@5 | Latencia |
|---------------|-------|-------------|----------|
| A: Vector + No Rerank | Q1 | 66.67% | 4285.7ms |
| A: Vector + No Rerank | Q2 | 75.00% | 9.1ms |
| A: Vector + No Rerank | Q3 | 50.00% | 10.6ms |
| A: Vector + No Rerank | Q4 | 66.67% | 6.0ms |
| A: Vector + No Rerank | Q5 | 33.33% | 10.5ms |
| **A: Vector + No Rerank** | **Promedio** | **58.33%** | **864.4ms** |
| B: Hybrid + No Rerank | Q1 | 66.67% | 18.3ms |
| B: Hybrid + No Rerank | Q2 | 75.00% | 11.5ms |
| B: Hybrid + No Rerank | Q3 | 50.00% | 10.3ms |
| B: Hybrid + No Rerank | Q4 | 66.67% | 10.4ms |
| B: Hybrid + No Rerank | Q5 | 33.33% | 8.6ms |
| **B: Hybrid + No Rerank** | **Promedio** | **58.33%** | **11.8ms** |
| C: Vector + Rerank | Q1 | 33.33% | 21791.5ms |
| C: Vector + Rerank | Q2 | 50.00% | 1071.9ms |
| C: Vector + Rerank | Q3 | 50.00% | 1084.1ms |
| C: Vector + Rerank | Q4 | 100.00% | 694.8ms |
| C: Vector + Rerank | Q5 | 33.33% | 621.9ms |
| **C: Vector + Rerank** | **Promedio** | **53.33%** | **5052.8ms** |
| D: Hybrid + Rerank | Q1 | 33.33% | 1621.7ms |
| D: Hybrid + Rerank | Q2 | 50.00% | 612.0ms |
| D: Hybrid + Rerank | Q3 | 50.00% | 215.2ms |
| D: Hybrid + Rerank | Q4 | 100.00% | 169.3ms |
| D: Hybrid + Rerank | Q5 | 33.33% | 415.1ms |
| **D: Hybrid + Rerank** | **Promedio** | **53.33%** | **606.7ms** |

### Resumen Ejecutivo

| Configuración | Precision@5 | Latencia |
|---------------|-------------|----------|
| **A: Vector + No Rerank** | 58.33% | 864.4ms |
| **B: Hybrid + No Rerank** | 58.33% | 11.8ms |
| **C: Vector + Rerank** | 53.33% | 5052.8ms |
| **D: Hybrid + Rerank** | 53.33% | 606.7ms |

## Análisis de Resultados

### Hallazgos Clave

1. **El reranking NO mejora la precisión en este dataset**
   - Sin reranking: 58.33% de precisión promedio
   - Con reranking: 53.33% de precisión promedio
   - El reranking empeora los resultados en 5 puntos porcentuales

2. **La búsqueda híbrida sin reranking es la ganadora**
   - Misma precisión que la vectorial (58.33%)
   - **73x más rápida** (11.8ms vs 864.4ms)
   - La búsqueda léxica aporta valor sin penalización

3. **El reranking añade mucha latencia sin beneficio**
   - C: 5052ms promedio (5.8x más lento que A)
   - D: 606ms promedio (51x más lento que B)
   - El cross-encoder es computacionalmente costoso

4. **Casos donde el reranking sí ayudó**
   - Q4 (IoT industrial): 100% de precisión con reranking vs 66.67% sin él
   - Esto sugiere que el reranking puede ser útil para dominios muy específicos

### Posibles Causas del Bajo Rendimiento del Reranking

1. **Dataset pequeño**: Solo 17 presupuestos, 60 chunks
   - El recall ya es bueno, no hay mucho que rerankear
   - El reranking brilla cuando hay muchos candidatos marginales

2. **Modelo multilingüe vs corpus en inglés**
   - Usamos `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingüe)
   - El corpus está en inglés
   - Un modelo especializado en inglés podría funcionar mejor

3. **Queries muy específicas**
   - Las queries del golden set son muy descriptivas
   - La búsqueda vectorial ya encuentra los chunks correctos
   - No hay "candidatos marginales" que el reranking pueda reordenar

4. **Primer query lento**
   - El primer query en cada configuración tarda mucho (4-21s)
   - Esto es por el lazy loading del modelo de reranking
   - En producción, el modelo debería pre-cargarse

### Recomendaciones

1. **Para este dataset**: Usar **Configuración B (Hybrid + No Rerank)**
   - Misma precisión que las otras
   - Latencia mínima (11.8ms)
   - Sin overhead del cross-encoder

2. **Para mejorar la precisión**:
   - Aumentar el tamaño del dataset (más presupuestos)
   - Mejorar los embeddings (modelos más grandes o fine-tuned)
   - Añadir más contexto a los chunks (contextual retrieval)

3. **Para evaluar el reranking correctamente**:
   - Dataset más grande (100+ presupuestos)
   - Queries más ambiguas que requieran discriminación fina
   - Modelo de reranking especializado en el dominio

## Conclusiones

El ejercicio demuestra que **más técnicas ≠ mejores resultados**. La configuración más simple (híbrida sin reranking) es la que ofrece el mejor balance precisión/latencia para este dataset.

El reranking es una técnica poderosa, pero su valor depende del contexto:
- **Dataset pequeño + queries específicas**: No aporta valor, añade latencia
- **Dataset grande + queries ambiguas**: Puede mejorar significativamente la precisión

La lección clave es **medir antes de optimizar**. Sin el golden set, habríamos asumido que el reranking mejoraría los resultados, pero los datos muestran lo contrario.
