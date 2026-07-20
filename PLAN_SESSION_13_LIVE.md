# Plan de Implementación: Session 13 Live

## Resumen de Diferencias

Mi implementación actual (pre-exercise) vs la implementación de Antonio (session_13_live):

### Estructura de Directorios
- **Mío**: `src/graph/`
- **Antonio**: `src/domain/graph/` (conductor territory)

### Arquitectura del Grafo
- **Mío**: 5 nodos secuenciales simples
- **Antonio**: Multi-agente con 8 nodos especializados + 2 human gates + fan-out paralelo

### Agentes
- **Mío**: extract_requirements, classify_components, search_budgets, generate_estimate, validate_and_consolidate
- **Antonio**: 
  - classifier_agent (complexity + reformulation)
  - structure_agent (reuses S12 run_structure_agent)
  - human_gate_structure (interrupt #1)
  - estimate_task_hours × N (fan-out paralelo con Send API)
  - recover_and_handover (agentic recovery)
  - analysis_agent (reliability report)
  - human_gate_analysis (interrupt #2)
  - proposal_agent (bonus)

### Características Avanzadas
- **Mío**: Checkpointer básico, spans simples
- **Antonio**:
  - AsyncConnectionPool para pausas largas
  - Activity log en tiempo real (Redis/dict)
  - Streaming endpoints con astream
  - Resume endpoints con Command(resume=...)
  - Personas temáticas (Matrix characters)
  - Reducer personalizado (merge_task_hours)
  - Command(goto) handovers explícitos

### Endpoints
- **Mío**: POST /v1/estimate/graph
- **Antonio**:
  - POST /v1/estimate/graph (START)
  - POST /v1/estimate/graph/{id}/resume (RESUME)
  - GET /v1/estimate/graph/{id}/state (READ)
  - POST /v1/estimate/graph/stream (START background)
  - POST /v1/estimate/graph/{id}/resume-stream (RESUME background)
  - GET /v1/estimate/graph/{id}/progress (POLL)
  - POST /v1/estimate/graph/{id}/proposal (generate proposal)

### Configuración
- **Mío**: Sin configuración específica
- **Antonio**: Modelos diferentes por agente, personas habilitables, effort mapping

---

## Plan de Implementación Paso a Paso

### Fase 1: Reestructuración de Directorios
1. Mover `src/graph/` → `src/domain/graph/`
2. Actualizar todos los imports
3. Actualizar `__init__.py`

### Fase 2: Estado Tipado Mejorado
1. Actualizar `state.py` con:
   - Campos para multi-agente flow
   - Reducer personalizado `merge_task_hours`
   - Campos para human gates (gate1_decision, gate2_decision)
   - Campos para proposal

### Fase 3: Agentes Especializados
1. Crear `agents/` subdirectorio
2. Implementar cada agente:
   - `classifier.py` - complexity + reformulation
   - `structure.py` - reuses run_structure_agent
   - `hours.py` - estimate_task_hours (fan-out) + recover_and_handover
   - `analysis.py` - reliability report
   - `gates.py` - human_gate_structure + human_gate_analysis
   - `proposal.py` - commercial proposal
   - `_common.py` - helpers compartidos
   - `__init__.py` - exports

### Fase 4: Schemas Internos
1. Crear `schemas.py` con modelos LLM I/O:
   - ComplexityClassification
   - ReliabilityReport
   - CommercialProposal
   - WeakPoint

### Fase 5: Personas Temáticas
1. Crear `personas.py` con Matrix characters
2. Implementar `persona_for()` helper

### Fase 6: Checkpointer Mejorado
1. Actualizar `checkpointer.py` con:
   - AsyncConnectionPool
   - Context manager pattern
   - saver_conninfo() helper

### Fase 7: Activity Log
1. Crear `activity.py` con:
   - GraphActivityLog class
   - describe_node() helper
   - Redis/dict backends

### Fase 8: Observabilidad
1. Crear `observability.py` con:
   - configure_logfire()
   - Singleton pattern

### Fase 9: Build del Grafo
1. Actualizar `build.py` con:
   - fan_out_hours() conditional edge
   - route_after_gate2() conditional edge
   - 8 nodos especializados

### Fase 10: Schemas de Dominio
1. Crear `src/domain/schemas/graph_estimation.py` con:
   - GraphEstimateRequest
   - GraphResumeRequest
   - PendingGate
   - GraphRunState
   - ActivityEntry
   - GraphProgress
   - GraphProposalResponse

### Fase 11: Router Mejorado
1. Actualizar `estimate_graph.py` con:
   - START endpoint
   - RESUME endpoint
   - STATE endpoint
   - STREAM endpoints (background)
   - PROGRESS endpoint
   - PROPOSAL endpoint

### Fase 12: Configuración
1. Actualizar `src/core/config.py` con:
   - GRAPH_CLASSIFIER_MODEL
   - GRAPH_ANALYSIS_MODEL
   - GRAPH_PROPOSAL_MODEL
   - GRAPH_PROPOSAL_ENABLED
   - GRAPH_PERSONAS_ENABLED
   - GRAPH_STRUCTURE_EFFORT_BY_COMPLEXITY

### Fase 13: Main.py
1. Actualizar inicialización del grafo:
   - AsyncExitStack pattern
   - Error handling graceful
   - app.state.graph

### Fase 14: Dependencies
1. Actualizar `src/dependencies.py` con:
   - get_graph_activity()

### Fase 15: Tests
1. Actualizar tests existentes
2. Crear tests para nuevos agentes
3. Tests para human gates
4. Tests para streaming

---

## Archivos a Crear/Modificar

### Nuevos (15 archivos)
1. `src/domain/graph/agents/__init__.py`
2. `src/domain/graph/agents/_common.py`
3. `src/domain/graph/agents/classifier.py`
4. `src/domain/graph/agents/structure.py`
5. `src/domain/graph/agents/hours.py`
6. `src/domain/graph/agents/analysis.py`
7. `src/domain/graph/agents/gates.py`
8. `src/domain/graph/agents/proposal.py`
9. `src/domain/graph/schemas.py`
10. `src/domain/graph/personas.py`
11. `src/domain/graph/activity.py`
12. `src/domain/graph/observability.py`
13. `src/domain/schemas/graph_estimation.py`

### Modificados (8 archivos)
1. `src/domain/graph/state.py`
2. `src/domain/graph/build.py`
3. `src/domain/graph/checkpointer.py`
4. `src/domain/graph/__init__.py`
5. `src/api/routers/estimate_graph.py`
6. `src/core/config.py`
7. `src/main.py`
8. `src/dependencies.py`

### Eliminados (0 archivos)
- Mantener `nodes.py` como referencia del pre-exercise

---

## Orden de Implementación

1. **Fase 1-2**: Reestructuración y estado (base)
2. **Fase 3-5**: Agentes y schemas (core)
3. **Fase 6-8**: Infraestructura (checkpointer, activity, observability)
4. **Fase 9-10**: Build y schemas de dominio
5. **Fase 11-13**: Router, config, main
6. **Fase 14-15**: Dependencies y tests

---

## Estimación de Tiempo

- **Fase 1-2**: 30 min
- **Fase 3-5**: 2 horas
- **Fase 6-8**: 1 hora
- **Fase 9-10**: 45 min
- **Fase 11-13**: 1.5 horas
- **Fase 14-15**: 1 hora

**Total**: ~7 horas

---

## Notas Importantes

1. **Mantener nodes.py**: Los nodos del pre-exercise quedan como referencia
2. **Backward compatibility**: El endpoint original debe seguir funcionando
3. **Error handling**: El grafo es optional infrastructure (503 si falla)
4. **Streaming**: Los endpoints de streaming son aditivos, no reemplazan los blocking
5. **Personas**: Son didácticas, no cambian el output shape
6. **Tests**: Priorizar tests de agentes individuales antes del grafo completo
