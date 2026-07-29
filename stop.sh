#!/usr/bin/env bash
# stop.sh — Stop all climBright services
# Usage: ./stop.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT/.pids"

echo "climBright — Stopping services..."

if [ -f "$PIDFILE" ]; then
  while read -r pid; do
    [ -z "$pid" ] && continue
    kill "$pid" 2>/dev/null && echo "  Stopped PID $pid" || echo "  PID $pid already stopped"
  done < "$PIDFILE"
  rm -f "$PIDFILE"
else
  echo "  No .pids file. Killing by name..."
  pkill -f "mongod.*2701" 2>/dev/null || true
  pkill -f "uvicorn main:app" 2>/dev/null || true
  pkill -f "node server.js" 2>/dev/null || true
fi

echo "  Done."
