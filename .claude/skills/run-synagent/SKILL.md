---
name: run-synagent
description: Build, run, and drive SynAgent. Use when asked to start SynAgent, run its server, smoke-test it, send a chat message, or interact with the running app.
---

SynAgent is a retrosynthesis assistant served as an HTTP API (Pydantic AI `agent.to_web()` over uvicorn). Drive it via `curl` against the SSE chat endpoint at `/api/chat`. The smoke script at `.claude/skills/run-synagent/smoke.sh` launches, verifies, and stops the server in one command.

## Prerequisites

Python 3.13 and `uv` must be available. No additional system packages required beyond the Python virtualenv.

```bash
uv sync     # installs all deps into .venv
```

## Environment

Create a `.env` file at the repo root and fill in:

```
GOOGLE_API_KEY=...          # required — Gemini API key
CHEMSPACE_API_KEY=...       # optional — needed for Chemspace search tools
```

The server loads `.env` automatically at startup via `python-dotenv`.

## Run (agent path)

### Smoke test (launch + verify + stop)

```bash
bash .claude/skills/run-synagent/smoke.sh
```

Exit 0 means the server started, `/api/health` returned `{"ok":true}`, and a chat request streamed at least a `start` event. Logs at `/tmp/synagent.log`.

### Manual background launch

```bash
uv run synagent serve --port 8000 &>/tmp/synagent.log &
SERVER_PID=$!

# Wait for readiness
for i in {1..30}; do
  curl -sf http://localhost:8000/api/health >/dev/null && break
  sleep 0.5
done
```

### Health check

```bash
curl http://localhost:8000/api/health
# → {"ok":true}
```

### Send a chat message

The chat endpoint uses the [Vercel AI SDK streaming format](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol) (SSE). The required `trigger` field is the discriminator:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "trigger": "submit-message",
    "id": "req-1",
    "messages": [
      {
        "id": "msg-1",
        "role": "user",
        "parts": [{"type": "text", "text": "Validate this SMILES: CCO"}]
      }
    ]
  }'
```

Response is an SSE stream of JSON events: `start`, `start-step`, `text-start`, `text-delta` (repeated), `text-end`, `finish-step`, `finish`.

To wait for the full response, pipe through `grep "text-delta"` and extract `delta` fields:

```bash
curl -sN -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"trigger":"submit-message","id":"r1","messages":[{"id":"m1","role":"user","parts":[{"type":"text","text":"What tools do you have?"}]}]}' \
  | grep '"text-delta"' | sed 's/.*"delta":"\(.*\)","id.*/\1/'
```

### Stop

```bash
kill $SERVER_PID
# or, if PID is lost:
pkill -f "synagent serve"
```

## Run (human path)

```bash
uv run synagent serve    # starts on http://localhost:8000, opens chat UI in browser
```

Ctrl-C to stop. The UI is served from CDN (pydantic/ai-chat-ui) and cached on first load — requires network access the first time.

## Gotchas

- **Missing `trigger` field crashes the request** — The `/api/chat` endpoint requires `"trigger": "submit-message"` at the top level. Sending just `{"messages": [...]}` returns a `422`/`500` with a Pydantic discriminator error. This is a Vercel AI SDK format requirement, not standard OpenAI chat format.
- **`GOOGLE_API_KEY` must be set** — the server starts fine without it, but every chat request immediately returns a 500. The error appears only in `/tmp/synagent.log`, not in the HTTP response body.
- **First load requires network** — `agent.to_web()` fetches the chat UI HTML from CDN on first request and caches it. In a network-restricted environment, pass a local HTML file via `--html-source`.
- **Port 8000 may already be in use** — pass `--port <other>` to `synagent serve`; update the `PORT` env var when running `smoke.sh`.
