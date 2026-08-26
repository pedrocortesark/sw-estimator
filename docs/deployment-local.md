# Despliegue Local con Docker Compose

Este documento describe cómo levantar el sistema completo de estimación usando Docker Compose.

## Arquitectura

El sistema se compone de 4 servicios orquestados:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
│                                                                 │
│  ┌─────────────┐      ┌─────────────────┐      ┌────────────┐ │
│  │  Streamlit  │─────▶│   AI Service    │─────▶│  Postgres  │ │
│  │   :8501     │      │   (FastAPI)     │      │   :5432    │ │
│  │  (public)   │      │   (internal)    │      │            │ │
│  └─────────────┘      └────────┬────────┘      └────────────┘ │
│                                │                                │
│                                ▼                                │
│                       ┌─────────────────┐                      │
│                       │  Redis Stack    │                      │
│                       │    :6379        │                      │
│                       └─────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         ▲                                          
         │ Port 8501                                
         │                                          
    ┌────┴────┐                                     
    │  Host   │                                     
    │ Browser │                                     
    └─────────┘                                     
```

**Puntos clave de seguridad:**
- **AI Service** NO expone puertos al host — solo es accesible desde la red interna de Docker
- **Streamlit** es el único punto de entrada público (puerto 8501)
- Las bases de datos exponen puertos solo para desarrollo/inspección

## Requisitos

- Docker Engine 20.10+
- Docker Compose 2.0+
- Clave API de OpenAI o Anthropic

Verificar instalación:
```bash
docker --version
docker compose version
```

## Configuración

### 1. Variables de entorno

Copiar el archivo de ejemplo y editar con tus valores:

```bash
cp .env.example .env
```

Editar `.env` y configurar como mínimo:

```bash
# Obligatorio — al menos una clave API
OPENAI_API_KEY=sk-...
# o
ANTHROPIC_API_KEY=sk-ant-...

# Recomendado — token para autenticación entre servicios
# Generar con: openssl rand -hex 32
SERVICE_TOKEN=tu-token-seguro-aqui
```

**Nota:** El archivo `.env` NUNCA se sube al repositorio (está en `.gitignore`).

### 2. Modelo LLM

Por defecto usa `gpt-4o-mini`. Para cambiar:

```bash
LLM_PROVIDER=openai        # o "anthropic"
PRIMARY_MODEL=gpt-4o       # modelo principal
FALLBACK_MODEL=claude-haiku-4-5-20251001  # fallback
```

## Arranque

### Construir y levantar

```bash
docker compose build
docker compose up
```

O en modo detachado (background):

```bash
docker compose up -d
```

### Verificar servicios

```bash
docker compose ps
```

Deberías ver:

```
NAME                SERVICE       STATUS      PORTS
sw-estimator-ai-service-1     ai-service    running     
sw-estimator-postgres-1       postgres      running     0.0.0.0:5432->5432/tcp
sw-estimator-redis-1          redis         running     0.0.0.0:6380->6379/tcp
sw-estimator-streamlit-1      streamlit     running     0.0.0.0:8501->8501/tcp
```

## Verificaciones

### 1. Servicios arriba y healthy

```bash
docker compose ps
```

Todos los servicios deben mostrar `running` o `healthy`.

### 2. Interfaz web accesible

Abrir en el navegador: **http://localhost:8501**

Deberías ver la interfaz de Streamlit del estimador.

### 3. AI Service NO es accesible directamente

```bash
curl http://localhost:8000/health
# Debe fallar con "Connection refused" — el puerto no está expuesto
```

El AI Service solo es accesible internamente vía `http://ai-service:8000`.

### 4. Health endpoint interno

Verificar desde dentro del contenedor de Streamlit:

```bash
docker compose exec streamlit curl http://ai-service:8000/health
```

Debe responder:
```json
{"status":"ok","env":"production","llm_models":["anthropic/claude-haiku-4-5-20251001","openai/gpt-4o-mini"]}
```

### 5. Estimación end-to-end

1. Abrir **http://localhost:8501** en el navegador
2. Escribir o pegar una transcripción de reunión
3. Enviar para estimación
4. Verificar que se recibe una estimación estructurada

El flujo completo es:
```
Browser → Streamlit (:8501) → AI Service (:8000 interno) → Redis/Postgres → Respuesta
```

## Autenticación entre servicios

Cuando `SERVICE_TOKEN` está configurado en `.env`:

- El AI Service requiere el header `X-Service-Token` en endpoints de estimación
- Streamlit incluye automáticamente el token en todas sus peticiones
- El endpoint `/health` NO requiere token (para healthchecks de Docker)

Probar sin token (debe fallar 401):
```bash
docker compose exec streamlit curl -X POST http://ai-service:8000/api/v1/sessions
# {"detail":"Invalid or missing service token."}
```

Probar con token:
```bash
docker compose exec streamlit curl -X POST http://ai-service:8000/api/v1/sessions \
  -H "X-Service-Token: tu-token-aqui"
# {"session_id":"..."}
```

## Parar y limpiar

### Parar servicios (mantener datos)

```bash
docker compose down
```

### Parar y eliminar volúmenes (borra datos)

```bash
docker compose down -v
```

### Reconstruir desde cero

```bash
docker compose down -v
docker compose build --no-cache
docker compose up
```

## Persistencia de datos

Los volúmenes de Docker mantienen los datos entre reinicios:

- `postgres_data` — base de datos PostgreSQL
- `redis_data` — cache y datos de Redis

Para ver volúmenes:
```bash
docker volume ls | grep sw-estimator
```

## Troubleshooting

### El AI Service no arranca

Ver logs:
```bash
docker compose logs ai-service
```

Problemas comunes:
- Falta `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` en `.env`
- PostgreSQL o Redis no están healthy (esperar unos segundos)

### Streamlit no conecta con AI Service

Verificar que `ESTIMATOR_API_URL` está correcto en docker-compose.yml:
```yaml
environment:
  - ESTIMATOR_API_URL=http://ai-service:8000
```

El nombre `ai-service` debe coincidir con el nombre del servicio en docker-compose.yml.

### Error 401 en estimaciones

Si `SERVICE_TOKEN` está configurado, verificar que Streamlit lo tiene:
```bash
docker compose exec streamlit env | grep SERVICE_TOKEN
```

Debe mostrar el mismo valor que en `.env`.

### Puertos en uso

Si los puertos 8501, 5432 o 6380 están ocupados:

```bash
# Editar docker-compose.yml y cambiar los puertos del host
ports:
  - "8502:8501"  # Cambiar 8501 por otro puerto libre
```

## Desarrollo local (sin Docker)

Para desarrollo rápido sin contenedores:

```bash
# Terminal 1 — Redis
redis-stack-server

# Terminal 2 — PostgreSQL
# (usar instancia local o Docker separado)

# Terminal 3 — AI Service
uv run uvicorn src.main:app --reload --port 8000

# Terminal 4 — Streamlit
ESTIMATOR_API_URL=http://localhost:8000 uv run streamlit run app/streamlit_app.py
```

Nota: En desarrollo local, dejar `SERVICE_TOKEN=` vacío en `.env` para deshabilitar autenticación.

## Estructura de archivos

```
sw-estimator/
├── Dockerfile              # Multi-stage build (builder + runtime)
├── docker-compose.yml      # Orquestación de servicios
├── .env.example            # Plantilla de variables (SÍ se versiona)
├── .env                    # Valores reales (NO se versiona)
├── .dockerignore           # Archivos excluidos de la imagen
├── src/                    # Código FastAPI
├── app/                    # Streamlit frontend
└── docs/
    └── deployment-local.md # Este documento
```

## Siguientes pasos

- Configurar un backend de negocio (Rails/Django) como punto de entrada público
- Añadir HTTPS con nginx/caddy como reverse proxy
- Implementar logging centralizado
- Configurar backups automáticos de PostgreSQL
