#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load secrets/config from repo root if present (API keys, etc.)
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
	set -a
	# shellcheck disable=SC1090
	source "$ENV_FILE"
	set +a
fi

# SQLite will be created under ./db/sqlite/users.db
export CLIMB_DB_DIR="./db"

# Default FastAPI port in this repo
PORT="${1:-9000}"

DEFAULT_PY="$ROOT_DIR/env/bin/python"
if [[ -x "$DEFAULT_PY" ]]; then
	PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PY}"
else
	PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "Starting FastAPI on port $PORT (db at $CLIMB_DB_DIR/sqlite/users.db)"

"$PYTHON_BIN" -m uvicorn main:app --reload --port "$PORT"
