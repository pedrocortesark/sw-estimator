"""Ejemplos de estimación para inyección de contexto estático (CAG).

Este módulo contiene los ejemplos de few-shot learning que se inyectan
directamente en el system prompt del LLM en cada petición.

Arquitectura CAG: no hay base de datos ni retrieval. Todos los ejemplos
viajan en el contexto de cada llamada. Esto es válido mientras el volumen
de ejemplos quepa en la ventana de contexto del modelo.
"""

ESTIMATION_EXAMPLES: list[dict[str, str]] = [
    {
        "meeting_summary": (
            "Un cliente del sector arquitectura quiere una web donde los usuarios "
            "manipulen parámetros de una fachada paramétrica en tiempo real. "
            "En el background, las peticiones deben viajar a un servidor Rhino.Compute "
            "que ejecute definiciones de Grasshopper (.gh) y devuelva la geometría "
            "resultante para ser renderizada en un canvas de Three.js en el frontend. "
            "El cliente quiere además que los usuarios puedan exportar la geometría "
            "final en formato OBJ/STL."
        ),
        "estimation": """\
# Estimación: Visualizador Geométrico Paramétrico (Rhino.Compute + Three.js)

## Resumen ejecutivo
Proyecto de alta complejidad técnica que combina computational design, infraestructura
especializada Windows/Rhino y desarrollo frontend 3D. Requiere perfiles híbridos poco
comunes en el mercado.

---

## Desglose por módulo

### 1. Computational Design — Scripts Grasshopper
Desarrollo y preparación de las definiciones `.gh` para exposición vía Hops/API.

| Tarea | Horas | Perfil |
|---|---|---|
| Modelado paramétrico de la fachada en Grasshopper | 24h | Computational Design Specialist |
| Exposición de parámetros de entrada/salida vía Hops | 8h | Computational Design Specialist |
| Validación de rangos y manejo de geometría inválida | 6h | Computational Design Specialist |
| Pipeline de exportación OBJ/STL desde GH | 6h | Computational Design Specialist |
| **Subtotal** | **44h** | |

### 2. Infraestructura — Windows Server + Rhino.Compute
Provisionamiento y configuración del servidor de cómputo especializado.

| Tarea | Horas | Perfil |
|---|---|---|
| Provisionamiento Windows Server (Azure/AWS EC2) | 4h | DevOps / Cloud Engineer |
| Instalación y licenciamiento de Rhino 7/8 + Compute | 4h | DevOps + Computational Design |
| Configuración de Rhino.Compute como servicio (IIS/NSSM) | 6h | DevOps / Cloud Engineer |
| HTTPS reverse proxy (nginx/Caddy) + firewall rules | 4h | DevOps / Cloud Engineer |
| Health monitoring y auto-restart del servicio | 3h | DevOps / Cloud Engineer |
| **Subtotal** | **21h** | |

### 3. Frontend — Three.js + UI de parámetros
Interfaz web para manipulación de parámetros y visualización 3D.

| Tarea | Horas | Perfil |
|---|---|---|
| Setup proyecto (React/Vite + Three.js / react-three-fiber) | 4h | Frontend Engineer |
| Componentes de sliders/inputs para parámetros paramétricos | 10h | Frontend Engineer |
| Renderer 3D: carga de mallas, materiales, cámara orbital | 14h | Frontend Engineer (3D) |
| Gestión de estados de carga/error durante el cómputo | 6h | Frontend Engineer |
| Exportación OBJ/STL desde el cliente | 4h | Frontend Engineer |
| **Subtotal** | **38h** | |

### 4. Comunicación Frontend ↔ Rhino.Compute
Lógica de peticiones asíncronas con feedback al usuario (el cómputo puede tardar 5-30s).

| Tarea | Horas | Perfil |
|---|---|---|
| API REST intermediaria (FastAPI/Node) como proxy a Compute | 8h | Backend Engineer |
| Long Polling o WebSocket para notificar fin de cómputo | 10h | Backend Engineer |
| Serialización/deserialización de geometría (Rhino3dm.js) | 6h | Backend Engineer |
| Rate limiting y queue de peticiones (evitar sobrecarga) | 6h | Backend Engineer |
| **Subtotal** | **30h** | |

### 5. QA, integración y documentación
| Tarea | Horas | Perfil |
|---|---|---|
| Testing E2E del flujo completo | 10h | QA / Full-stack |
| Documentación técnica de la infraestructura | 4h | Tech Lead |
| **Subtotal** | **14h** | |

---

## Resumen de horas y coste estimado

| Módulo | Horas |
|---|---|
| Computational Design (GH/Hops) | 44h |
| Infraestructura Windows/Rhino.Compute | 21h |
| Frontend Three.js | 38h |
| Capa de comunicación async | 30h |
| QA + Documentación | 14h |
| **TOTAL** | **147h** |

**Rango estimado:** 140–165 horas (±10% por incertidumbre en licenciamiento y geometría).

## Perfiles necesarios
- Computational Design Specialist (Grasshopper/Hops) — 44h
- DevOps / Cloud Engineer (Windows Server) — 21h
- Frontend Engineer con experiencia en Three.js — 38h
- Backend Engineer (API, async, WebSocket) — 30h
- QA — 14h

## Riesgos principales
1. **Licencias Rhino en servidor**: requiere licencia flotante (Zoo) o Cloud Zoo — coste adicional ~€1.000/año.
2. **Latencia de cómputo**: definiciones GH complejas pueden superar 30s; el UX debe gestionarlo explícitamente.
3. **Geometría inválida**: Grasshopper puede producir mallas no cerradas; necesita validación antes de enviarla al frontend.
""",
    },
    {
        "meeting_summary": (
            "El cliente gestiona un sistema de inventarios propio (stack Python/Django) "
            "y necesita integrarlo con dos sistemas externos: una pasarela de pagos "
            "(Stripe) para procesar cobros automáticos cuando el stock baja de un umbral, "
            "y un CRM externo (HubSpot) para sincronizar el estado de los pedidos. "
            "La sincronización con HubSpot debe ser bidireccional. "
            "El equipo actual no tiene experiencia con ninguna de las dos APIs."
        ),
        "estimation": """\
# Estimación: Integración Inventario ↔ Stripe + HubSpot (Bidireccional)

## Resumen ejecutivo
Integración de complejidad media-alta. El volumen de código no es grande, pero la
fiabilidad es crítica (pagos + datos de negocio). Requiere manejo robusto de errores,
idempotencia en los webhooks y sincronización bidireccional consistente.

---

## Desglose por módulo

### 1. Integración con Stripe
Procesamiento de pagos automáticos ante eventos de inventario.

| Tarea | Horas | Perfil |
|---|---|---|
| Estudio de la API Stripe (Payment Intents / Invoices) | 4h | Backend Engineer |
| Lógica de trigger: umbral de stock → crear Payment Intent | 8h | Backend Engineer |
| Webhooks de Stripe → actualizar estado de pago en inventario | 8h | Backend Engineer |
| Idempotencia: evitar dobles cobros ante reintentos | 6h | Backend Engineer |
| Tests con Stripe CLI (modo sandbox) | 6h | Backend Engineer / QA |
| **Subtotal** | **32h** | |

### 2. Integración bidireccional con HubSpot CRM
Sincronización de pedidos entre el inventario y HubSpot Deals/Contacts.

| Tarea | Horas | Perfil |
|---|---|---|
| Estudio API HubSpot (Deals, Contacts, Properties) | 4h | Backend Engineer |
| Inventario → HubSpot: crear/actualizar Deal al procesar pedido | 10h | Backend Engineer |
| HubSpot → Inventario: webhook al cambiar etapa del Deal | 8h | Backend Engineer |
| Mapeo y transformación de campos entre ambos sistemas | 6h | Backend Engineer |
| Gestión de conflictos en sincronización bidireccional | 8h | Backend Engineer |
| **Subtotal** | **36h** | |

### 3. Backend transversal — Autenticación, cola y resiliencia

| Tarea | Horas | Perfil |
|---|---|---|
| Gestión segura de API Keys (Stripe + HubSpot) con secretos cifrados | 3h | Backend Engineer |
| Cola de tareas async para operaciones no críticas en tiempo real (Celery/RQ) | 8h | Backend Engineer |
| Retry logic con backoff exponencial para llamadas a APIs externas | 4h | Backend Engineer |
| Logging estructurado de todas las integraciones (auditoría) | 4h | Backend Engineer |
| **Subtotal** | **19h** | |

### 4. Testing y QA

| Tarea | Horas | Perfil |
|---|---|---|
| Tests unitarios de la lógica de transformación de datos | 6h | Backend Engineer / QA |
| Tests de integración con mocks de Stripe y HubSpot | 8h | QA Engineer |
| Tests E2E en entorno staging con cuentas sandbox reales | 6h | QA Engineer |
| **Subtotal** | **20h** | |

### 5. Documentación y entrega

| Tarea | Horas | Perfil |
|---|---|---|
| Documentación de los flujos de integración (diagramas + runbook) | 4h | Tech Lead |
| Formación al equipo cliente sobre gestión de webhooks | 2h | Tech Lead |
| **Subtotal** | **6h** | |

---

## Resumen de horas

| Módulo | Horas |
|---|---|
| Integración Stripe | 32h |
| Integración HubSpot (bidireccional) | 36h |
| Backend transversal (cola, resiliencia, auth) | 19h |
| Testing y QA | 20h |
| Documentación y entrega | 6h |
| **TOTAL** | **113h** |

**Rango estimado:** 110–125 horas (±10% por curva de aprendizaje en las APIs nuevas).

## Perfiles necesarios
- Backend Engineer (Python/Django, experiencia en integraciones) — 95h
- QA Engineer — 14h
- Tech Lead (revisión de arquitectura + documentación) — 6h

## Riesgos principales
1. **Curva de aprendizaje**: el equipo no conoce Stripe ni HubSpot; se ha añadido buffer de estudio (8h total).
2. **Sincronización bidireccional**: los conflictos de datos entre sistemas son el punto más delicado; requiere una estrategia de "source of truth" definida con el cliente antes de empezar.
3. **Rate limits de HubSpot**: el plan gratuito/starter tiene límites agresivos (100 req/10s); puede requerir upgrade.
""",
    },
]
