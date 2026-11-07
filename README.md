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
in `plan.md`. Only the health endpoint, fixture endpoint, event envelope, and mock
graph are implemented. The other backend packages are placeholders. Full node/tool
contracts, WebSocket streaming, interventions, statistical tools, and Gemini integration
remain planned work. Coordinate contract changes across backend and frontend owners.

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
