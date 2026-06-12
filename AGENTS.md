# sw-estimator — Agent instructions

## Quick start

```bash
uv sync --group dev           # install project + dev deps
cp .env.example .env          # fill in API keys
./dev.sh                      # starts Redis Stack + FastAPI + Streamlit
uv run pytest -v              # run all tests
uv run pytest -v -m "not golden and not hard and not soft and not judge"  # offline tests only
```

## Package manager

Use `uv` exclusively. No pip, no poetry. Lockfile is `uv.lock`.

## Python

3.13 locally (`uv run` / `.python-version`), 3.12 in Docker. `src/` is the package root (`[tool.hatch.build.targets.wheel.packages]`).

## Architecture

```
Streamlit (:8501) → FastAPI (:8000) → Guardrails → Semantic Cache → Jinja2 prompts → Instructor/LiteLLM → Guardrails
```

Source tree:
- `src/` — Python package (FastAPI app at `src/main.py:65 app`)
- `app/streamlit_app.py` — Streamlit frontend
- `frontend/SwEstimator.Client/` — Blazor WASM frontend (separate Docker service)
- `src/routers/` — API endpoints (`estimation.py`, `sessions.py`, `health.py`)
- `src/services/estimation.py` — pipeline orchestrator
- `src/core/config.py` — pydantic-settings (reads `.env`)
- `src/cache/semantic.py` — RedisVL-based semantic cache (requires Redis Stack, not vanilla Redis)
- `src/ingest/` — data ingestion subsystem (catalog, loaders, parsers, PII)

Sessions are in-memory (`src/services/sessions.py:session_store`), not persistent.

## Commands

| Command | What |
|---|---|
| `uv run uvicorn src.main:app --reload` | Backend dev server |
| `ESTIMATOR_API_URL=http://localhost:8000 uv run streamlit run app/streamlit_app.py` | Frontend dev server |
| `docker compose up --build` | Production-like stack |
| `./dev.sh stop` | Stop all services |
| `uv run pytest -v` | All tests |
| `uv run pytest -v tests/api/test_sessions.py` | Single file |
| `uv run pytest -v -k "test_health"` | Single test |
| `uv run pytest -v -m "golden"` | Golden dataset (requires live LLM) |
| `python -m evals.run --mode actor` | Eval suite against golden dataset |
| `python -m evals.run --mode acb` | Eval suite (exit 1 on failure, for CI) |

## Test quirks

- `asyncio_mode = "auto"` in pyproject.toml — no `@pytest.mark.asyncio` needed for top-level async tests.
- **Live LLM tests** are tagged `golden`, `hard`, `soft`, or `judge`. They require `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` and are throttled to 1 call per 20s (see `tests/conftest.py:_llm_throttle`). Skip them with `-m "not golden and not hard and not soft and not judge"`.
- Mock LLM calls with FastAPI `dependency_overrides` (see tests/api/test_sessions.py). No monkey-patching.
- The `client` fixture uses `ASGITransport` (no TCP needed).
- Test paths: only `tests/` is scanned.

## CI

`.github/workflows/ci.yml`: `uv sync --group dev` → `uv run pytest -v`. No typecheck, no linter currently configured.

## Environment / Configuration

- `.env` is never committed (see `.gitignore`). `.env.example` documents all vars.
- `pydantic-settings` loads from `.env` at `src/core/config.py:12`.
- Prompts live in `src/prompts/estimation/v{1,2,3}/`. Active version set via `PROMPT_VERSION` env var.
- `PRESIDIO_SPACY_MODEL=es_core_news_md` — download with `python -m spacy download es_core_news_md`.
- LLM model fallback chain: `PRIMARY_MODEL` → `FALLBACK_MODEL` with `LLM_RETRIES` retries and `LLM_TIMEOUT` deadline.

## Notable conventions

- **No generated code** anywhere. Everything is hand-written.
- **No pre-commit hooks.**
- **No Alembic / migrations yet** (Postgres is Session 6+ infra, pgvector extension not activated).
- Docker images use multi-stage builds (builder → runtime).
- `src/config.py` exists alongside `src/core/config.py`. The former is a duplicate/legacy; prefer `src.core.config.get_settings()`.
