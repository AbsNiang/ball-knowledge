# Whack

NBA research assistant with deterministic statistical checks and an inspectable research graph.

## Quick start

Requires Python 3.12+ and Node.js 20.19+ (or Node.js 22.12+).
Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m uvicorn backend.api.main:app --reload
```

The ASGI app is defined in `backend/api/main.py`, so use `backend.api.main:app`.
The older `backend.main:app` command is incorrect. Run the command from the
repository root; virtual environment activation is optional when using the explicit
Python path above.

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. API documentation is at http://127.0.0.1:8000/docs.
Keep both terminals running. The Vite development server proxies HTTP and WebSockets
under `/api` to the backend. Scripted demo needs no API key. Unchecking it runs the
research agent using `LLM_MODE` from the server environment or `.env` (default `fake`).
Run the backend from the checkout so the fixture file is available.

## Structure

```text
contracts/       Shared Python and TypeScript event envelopes
backend/
  api/          FastAPI health and fixture endpoints
  data/         Data preparation and feature engineering
  tools/        Deterministic statistical tools
  gate/         Statistical gates, ledger, verdict rules
  llm/          Shared LLM gateway
  agent/        Orchestrator and critic
  events/       Event storage and graph projection
  tests/        Backend tests
frontend/src/   React + TypeScript + React Flow
replays/        Mock and future recorded JSONL event logs
```

The directory layout follows `TASKS.md`; role ownership follows the addendum in
`plan.md` section 7.1: A owns data/stats, B owns the agent, and C owns product/API and
contracts. The server includes typed run/intervention/report endpoints, an in-memory
event log, WebSocket backfill/live streaming, and timed JSONL replay. The default app
connects `AgentService` to `POST /runs`, streaming the standalone agent's events and
exposing its reports. `create_app()` without an injected agent remains available for
isolated API tests. Durable API storage remains planned work. The API agent uses the
CLI's default synthetic stub tools, judged by the real gate; its results are not validated NBA findings.
Replay reports currently return 409; replay streams graph events.

See [contracts/README.md](contracts/README.md) for endpoint shapes, handoffs, schema
generation, error semantics, and agent integration. Python contracts are authoritative;
JSON Schema and TypeScript are generated from them. Coordinate changes with all owners.

To try replay without an agent, `POST /replay` with
`{"log_file":"mock.jsonl","speed":1}`, then connect to
`ws://127.0.0.1:8000/runs/{run_id}/events` using the returned ID. Events remain available
at `GET /runs/{run_id}/events`. Use a single server process; the development log resets
on restart.

## Agent loop

The agent loop (`backend/agent/`) runs against a scripted fake LLM and stub tools
(`backend/agent/stubs.py`), or Person A's mock tools with `--tools mock`. Results are judged by
the stats gate (`backend/gate/`: G1-G7, the run's test ledger, and the verdict table), which
re-runs the G4 multiple-testing correction over the whole ledger before the report. It writes
`contracts/events.py` events and connects to FastAPI through `backend/api/agent.py`.
The orchestrator is a LangGraph stateful router. A deterministic `router` node folds
finished gate/critic checks into state and dispatches one specialist per step
(`premise`, `hypothesize`, `plan_test`, `execute_test`, `choose_test`, `interpret`,
`verdict`, `report`). When a test passes its gates, the LLM chooses the next follow-up
(`out_of_sample`, `market_test`, `placebo`, or `holdout_confirm` once G1-G6 pass) or
concludes. When a gate fails, testing that hypothesis stops and the failure decides its
verdict; G6 (market) only locks the holdout. State is checkpointed, so approval mode can
pause on each decision and resume. Run it with the fake LLM (no API key needed):

```powershell
.\.venv\Scripts\python -m backend.agent                         # live mode
.\.venv\Scripts\python -m backend.agent --mode approval         # approve each decision
.\.venv\Scripts\python -m backend.agent --out replays\fake.jsonl # save the event log
.\.venv\Scripts\python -m backend.agent --tools mock            # Person A's mock tools
.\.venv\Scripts\python -m backend.agent --llm gemini --tools mock # real Gemini decisions
```

`--llm gemini` reads `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env` and sends at most
`GEMINI_RPM` (default 5) requests per minute, so a full run of about 35 calls takes ~7 minutes.

A saved log is valid replay input: with the server running, `POST /replay` with
`{"log_file":"fake.jsonl","speed":1}` streams the agent's graph to the UI.

## Checks and build

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
cd frontend
npm run build
```

The frontend build checks TypeScript and writes `frontend/dist/`. `npm run preview`
previews that build, but API calls require a production reverse proxy for `/api`;
use `npm run dev` for the integrated local workflow. There is no coverage threshold yet.

## Configuration and plans

`.env.example` documents gateway settings. The CLI and API agent load `.env`; existing
environment variables take precedence. `LLM_MODE=fake` uses scripted decisions without
network calls. `LLM_MODE=gemini` requires `GEMINI_API_KEY` and `GEMINI_MODEL` on the server.
Keep API keys, local environments, dependencies, and generated data out of Git.
See [plan.md](plan.md) for requirements, [TASKS.md](TASKS.md) for ownership,
[HOSTING_PLAN.md](HOSTING_PLAN.md) for the proposed deployment architecture and launch
checklist, and [AGENTS.md](AGENTS.md) for contributor guidance. This is a research tool,
not betting advice.

## Research workspace MVP

The frontend now includes collapsible session tabs, a downward Dagre research graph,
Markdown node inspection, evidence charts, and bidirectional WebSocket controls.
Start the backend and frontend using the quick-start commands above. Leave **Scripted
 demo** checked to try the full interaction without API keys. Demo results are synthetic;
submit any question to watch the scripted branching sequence, pause/resume it, or redirect
from the active step. A redirect retains the interrupted node and adds a labeled branch.
Uncheck demo to run the configured research agent. Pause takes effect at the next router
boundary after the current step finishes; resume continues that run. Redirect is currently
supported only by Scripted demo. Approval/rejection is supported through the API for
agent runs created in approval mode; the current UI starts runs in live mode.

Only one session can run at a time in this single-process MVP. Pause it before starting
or resuming another. Session IDs are stored in this browser; graph evidence reloads from
the server log on refresh. The server log is in memory and disappears on backend restart.
Sessions confirmed missing by the backend are removed from the browser list with an
explanation. Other sessions stay intact. Temporary disconnects retry automatically from
the last applied event, up to five times with backoff; failed attempts offer **Retry
connection**. Commands are never resent automatically. A heartbeat detects connections
left open by a proxy after the backend stops. Production must proxy HTTP and WebSockets
under `/api`.

For local connection troubleshooting, open http://127.0.0.1:5173/api/health; it should
return `{"status":"ok"}`. Connection refused on port 5173 means Vite is not running.
A proxy error with FastAPI stopped means the backend must be restarted. To use a different
backend port, set `$env:WHACK_API_TARGET="http://127.0.0.1:8001"` before `npm run dev`.

Frontend verification:

```powershell
cd frontend
npm run test
npm run build
npm run test:integration
```

The integration test starts temporary Vite/FastAPI servers on free ports, uses the fake
LLM, and checks streaming, controls, reconnects, reports, and backend restarts. It uses
the repository `.venv` Python; override with `WHACK_TEST_PYTHON` if needed. Node's built-in
WebSocket client is enabled for this test; no browser or external API key is needed.

The frontend uses React, Tailwind CSS, React Flow, Dagre, react-markdown with GFM,
and Recharts. Node selection opens the inspector; its Evidence tab accepts chart data
from `node.tool.output.charts` and always exposes the original tool JSON.
