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

The layout follows `TASKS.md`; it takes precedence over alternate directory names
in `plan.md`. Implemented so far: the health and fixture endpoints, the event envelope,
the mock graph, an in-memory event log with its graph projection (`backend/events/`), and
the agent loop (`backend/agent/`). The agent runs against a scripted fake LLM and stub
tools/gate (`backend/agent/stubs.py`) until the real tools, gate and Gemini gateway land.
Node/tool contracts, WebSocket streaming and the API routes remain planned work.
Coordinate contract changes across backend and frontend owners.

## Agent loop

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
