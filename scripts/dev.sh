#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env if present
if [ -f "$ROOT/.env" ]; then
  set -a; source "$ROOT/.env"; set +a
fi

# Prompt for password if not set
if [ -z "$APP_PASSWORD" ]; then
  read -rsp "Set APP_PASSWORD for this session: " APP_PASSWORD; echo
  export APP_PASSWORD
fi

# Auto-generate SESSION_SECRET if not set
if [ -z "$SESSION_SECRET" ]; then
  SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export SESSION_SECRET
fi

echo "Starting specGPT at http://localhost:8000"
echo "Press Ctrl+C to stop"

trap 'kill $BACKEND_PID 2>/dev/null' EXIT

(cd "$ROOT" && DEBUG_PIPELINE=1 python -m src.pipeline.app) &
BACKEND_PID=$!
wait $BACKEND_PID

