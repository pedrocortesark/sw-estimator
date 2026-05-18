#!/usr/bin/env bash
# dev.sh — start all three services for local development
#
# Usage: ./dev.sh [stop]
#
# Starts:
#   1. Redis Stack  (docker, port 6380)
#   2. FastAPI      (uvicorn --reload, port 8000)
#   3. Streamlit    (port 8501)
#
# Logs go to:  logs/api.log  and  logs/streamlit.log
# PIDs stored: .pids/api  and  .pids/streamlit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs .pids

# ── helpers ────────────────────────────────────────────────────────────────

green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }

stop_services() {
    yellow "Stopping services..."

    for svc in api streamlit; do
        pidfile=".pids/$svc"
        if [[ -f "$pidfile" ]]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" && yellow "  stopped $svc (pid $pid)"
            fi
            rm -f "$pidfile"
        fi
    done

    docker compose stop redis 2>/dev/null && yellow "  stopped redis" || true
    green "Done."
}

# ── stop mode ──────────────────────────────────────────────────────────────

if [[ "${1:-}" == "stop" ]]; then
    stop_services
    exit 0
fi

# ── trap Ctrl-C ────────────────────────────────────────────────────────────

trap 'echo; stop_services; exit 0' INT TERM

# ── 1. Redis Stack ─────────────────────────────────────────────────────────

yellow "Starting Redis Stack (port 6380)..."
docker compose up redis -d
green "  Redis Stack ready"

# ── 2. FastAPI (uvicorn) ───────────────────────────────────────────────────

yellow "Starting FastAPI backend (port 8000)..."
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload \
    > logs/api.log 2>&1 &
echo $! > .pids/api

# Wait until /health responds
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        green "  FastAPI ready  →  http://localhost:8000"
        break
    fi
    sleep 1
    if [[ $i -eq 20 ]]; then
        red "  FastAPI did not start in time. Check logs/api.log"
        stop_services; exit 1
    fi
done

# ── 3. Streamlit ───────────────────────────────────────────────────────────

yellow "Starting Streamlit frontend (port 8501)..."
ESTIMATOR_API_URL=http://localhost:8000 \
    uv run streamlit run app/streamlit_app.py \
        --server.port 8501 \
        --server.headless true \
    > logs/streamlit.log 2>&1 &
echo $! > .pids/streamlit
green "  Streamlit ready  →  http://localhost:8501"

# ── done ───────────────────────────────────────────────────────────────────

echo
green "All services running. Press Ctrl-C to stop."
echo "  API logs:       tail -f logs/api.log"
echo "  Streamlit logs: tail -f logs/streamlit.log"
echo

# Keep script alive so trap fires on Ctrl-C
wait
