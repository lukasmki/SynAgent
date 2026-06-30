#!/usr/bin/env bash
# Smoke test for SynAgent web server.
# Launches the server, runs a health check + one chat round-trip, then stops.
# Exit 0 = healthy. Logs at /tmp/synagent.log.

set -e

PORT=${PORT:-8000}
BASE=http://localhost:$PORT

# Kill any stale instance on this port
pkill -f "synagent serve" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 0.3

# Launch in background
uv run synagent serve --port "$PORT" &>/tmp/synagent.log &
SERVER_PID=$!

# Wait for health endpoint (up to 15 s)
for i in {1..30}; do
  curl -sf "$BASE/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done

# Health check
HEALTH=$(curl -sf "$BASE/api/health")
echo "Health: $HEALTH"

# One-shot chat (non-streaming, just the first line proves it accepted the request)
FIRST=$(curl -sf -X POST "$BASE/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "trigger": "submit-message",
    "id": "smoke-1",
    "messages": [
      {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "ping"}]}
    ]
  }' | head -3)
echo "Chat response (first lines):"
echo "$FIRST"

# Clean up
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
echo "Done."
