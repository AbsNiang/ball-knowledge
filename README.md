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

Open http://localhost:5173. API documentation is at http://127.0.0.1:8000/docs.
The Vite development server proxies `/api` to the backend. The starter loads
`replays/mock.jsonl`; it performs no statistical tests or LLM calls and needs no API key.
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
event log, WebSocket backfill/live streaming, and timed JSONL replay. Live run creation
returns `503 agent_unavailable` until B supplies `AgentPort` to `create_app(agent=...)`.
Statistical tools, durable storage, graph/report projection, and Gemini integration
remain planned work. Replay reports currently return 409; replay streams graph events.

See [contracts/README.md](contracts/README.md) for endpoint shapes, handoffs, schema
generation, error semantics, and agent integration. Python contracts are authoritative;
JSON Schema and TypeScript are generated from them. Coordinate changes with all owners.

To try replay without an agent, `POST /replay` with
`{"log_file":"mock.jsonl","speed":1}`, then connect to
`ws://127.0.0.1:8000/runs/{run_id}/events` using the returned ID. Events remain available
at `GET /runs/{run_id}/events`. Use a single server process; the development log resets
on restart.

## Agent loop

The agent loop (`backend/agent/`) runs against a scripted fake LLM and stub tools/gate
(`backend/agent/stubs.py`). It is not yet connected to `AgentPort` and still emits the
earlier untyped event shape, so it needs porting to `contracts/events.py` before it runs.
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
```

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

`.env.example` documents reserved gateway settings; this starter does not load them.
Keep API keys, local environments, dependencies, and generated data out of Git.
See [plan.md](plan.md) for requirements, [TASKS.md](TASKS.md) for ownership, and
[AGENTS.md](AGENTS.md) for contributor guidance. This is a research tool, not betting advice.
