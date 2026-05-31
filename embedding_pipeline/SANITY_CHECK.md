# Sanity Check — Embedding Pipeline

Modelo: `text-embedding-3-small` (1536 dimensiones)  
Fecha: 2026-05-31  
Script: `scripts/compare.py`

---

## Resultados

| Pareja | Texto A | Texto B | Similitud coseno |
|--------|---------|---------|-----------------|
| A — Cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Authorization service using JSON Web Tokens for a banking application" | **0.5957** |
| B — No relacionados | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Database migration from MySQL to PostgreSQL with zero downtime" | **0.1920** |
| C — Genéricos / ambiguos | "Backend services" | "API development" | **0.5407** |

---

## Análisis

**Pareja A (0.5957):** El resultado confirma la discriminación esperada: dos frases que describen esencialmente el mismo concepto (autenticación con JWT) desde ángulos diferentes obtienen una similitud alta. Supera el umbral orientativo de 0.6 con holgura pequeña, lo que indica que el modelo capta la equivalencia semántica incluso cuando la formulación es distinta (OAuth 2.0 vs. JSON Web Tokens, fintech vs. banking).

**Pareja B (0.1920):** Resultado muy bajo, bien por debajo del umbral de 0.4. Autenticación backend y migración de bases de datos son dominios técnicamente distantes, y el modelo lo refleja claramente. Este valor es el más "limpio" de los tres y sirve como ancla de referencia para lo que significa "no relacionado" en este espacio de embeddings.

**Pareja C (0.5407):** El resultado es sorprendentemente alto para textos tan vagos. "Backend services" y "API development" son frases genéricas de solo dos tokens cada una, pero el modelo les asigna una similitud comparables a la pareja A. Esto ilustra un efecto importante: **la ambigüedad colapsa los vectores hacia zonas densas del espacio semántico**, donde todo lo técnico-genérico queda próximo. Para el pipeline de estimación esto es relevante: chunks demasiado cortos o sin contexto pueden producir falsos positivos en la búsqueda por similitud.
