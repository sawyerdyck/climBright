#!/usr/bin/env bash
# start.sh — Start all climBright services (MongoDB, FastAPI, Express)
# Usage: ./start.sh
# Stop:  ./stop.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT/.pids"

echo "climBright — Starting services..."

# --- MongoDB ---
mkdir -p "$ROOT/db/mongo"
echo "  [1/3] MongoDB (port 2701)..."
mongod --dbpath "$ROOT/db/mongo" --bind_ip 127.0.0.1 --port 2701 --fork --logpath "$ROOT/db/mongo/mongod.log" 2>/dev/null \
  || mongod --dbpath "$ROOT/db/mongo" --bind_ip 127.0.0.1 --port 2701 &
MONGO_PID=$!
sleep 2

# --- FastAPI ---
echo "  [2/3] FastAPI (port 9000)..."
cd "$ROOT"
source env/bin/activate 2>/dev/null || true
uvicorn main:app --port 9000 &
FASTAPI_PID=$!
sleep 3

# --- Express ---
echo "  [3/3] Express (port 3000)..."
cd "$ROOT/frontend"
[ ! -d node_modules ] && npm install
node server.js &
EXPRESS_PID=$!
sleep 2

# Save PIDs
echo "$MONGO_PID" > "$PIDFILE"
echo "$FASTAPI_PID" >> "$PIDFILE"
echo "$EXPRESS_PID" >> "$PIDFILE"

echo ""
echo "  All services running!"
echo "  App:     http://127.0.0.1:3000"
echo "  FastAPI: http://127.0.0.1:9000/docs"
echo ""
echo "  Run ./stop.sh to shut everything down."
