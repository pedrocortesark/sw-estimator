# =============================================================================
# Stage 1 — builder
# Install uv and resolve/install all dependencies into a virtual environment.
# This stage is discarded after the build; only its output (.venv) is kept.
# =============================================================================
FROM python:3.12-slim AS builder

# Install uv (the fast Python package manager)
# We pin the version for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

# Set the working directory
WORKDIR /app

# Copy dependency manifests first — Docker caches layers, so if these haven't
# changed, the expensive uv sync is skipped on subsequent builds.
COPY pyproject.toml uv.lock ./

# Install dependencies into a local .venv inside /app.
# --no-install-project: only install deps, not our own package yet.
# UV_COMPILE_BYTECODE: pre-compile .pyc files for faster container startup.
RUN uv sync --no-install-project --compile-bytecode

# Now copy the source code and install our package too
COPY src/ src/
RUN uv sync --compile-bytecode


# =============================================================================
# Stage 2 — runtime
# Minimal image that only contains the app and its installed dependencies.
# No build tools, no uv, no cache — just what's needed to run.
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the source code
COPY --from=builder /app/src /app/src

# Copy the Streamlit frontend
COPY app/ app/

# Copy the data catalog (YAML) and seed data for ingestion
COPY data/ data/

# Copy prompts (Jinja2 templates)
COPY src/prompts/ src/prompts/

# Add the .venv binaries to PATH so we can call `uvicorn` directly
ENV PATH="/app/.venv/bin:$PATH"

# Download the spaCy Spanish model required by Presidio for PII detection.
# Without this, Presidio defaults to English and silently misses Hispanic names.
RUN python -m spacy download es_core_news_md

# Run as a non-root user — best practice for container security
RUN useradd --create-home appuser
USER appuser

# Expose the port uvicorn listens on
EXPOSE 8000

# Health check: Docker calls this every 30s to verify the container is alive.
# The /health endpoint is lightweight — it does NOT call the LLM or external services.
# If it fails 3 times in a row, Docker marks the container as unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
# --host 0.0.0.0: listen on all interfaces (required inside Docker)
# --workers 1: single worker for now; increase in production based on CPU cores
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
