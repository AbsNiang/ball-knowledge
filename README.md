# Whack

NBA research assistant with deterministic statistical checks and an inspectable research graph.

## Hosting

Test deployment: https://whack-production.up.railway.app

The Railway project is [whack](https://railway.com/project/d5cd55e4-f2ed-47f5-88bd-5c686c909d21).
The production entrypoint serves React and `/api` HTTP/WebSockets on one origin.
The Docker image pins Linux runtime dependencies in `requirements-runtime.txt`,
builds the frontend with `npm ci`, and runs one Uvicorn worker as a non-root user.
Railway uses one replica and `/api/health` for deployment health checks.

Build and launch locally with Docker Desktop's Linux engine running:

```powershell
docker build -t whack:local .
docker run --rm -p 8000:8000 -e PORT=8000 -e LLM_MODE=fake whack:local
```

Open http://localhost:8000. To exercise the production container or hosted URL:

```powershell
$env:WHACK_SMOKE_URL="http://localhost:8000"
npm --prefix frontend run test:hosted
```

This check creates synthetic sessions, checks assets, HTTP and WebSocket routing,
pause/resume, redirect, cursor reconnect, and completion/report generation through
the fake agent. It requires the frontend development dependencies and an idle app.
Set `WHACK_SMOKE_URL` to the HTTPS Railway URL to check the deployed service.
Use supported Node 22.12+ for this check. To use the pinned Node version in Docker:

```powershell
docker build --target frontend-build -t whack:checks .
docker run --rm -e WHACK_SMOKE_URL=http://host.docker.internal:8000 whack:checks npm run test:hosted
```

The deployment branch is `deploy/railway-demo`. Changes can be reviewed and tested
there before merging into `main`. If Railway's GitHub app has repository access,
select that source branch. Otherwise upload the clean branch with the CLI:

```powershell
npx --yes @railway/cli up --project d5cd55e4-f2ed-47f5-88bd-5c686c909d21 --environment production --service whack --detach
```

The CLI respects `.gitignore`; keep the bundled `replays/mock.jsonl` exception.
The current deployment was uploaded from commit `27f18fa` on `deploy/railway-demo`.
It uses CLI uploads because Railway's GitHub source connection still rejects this
private repository. Pushing the branch does not automatically redeploy the service.

Deployment uses `LLM_MODE=fake`, `LANGSMITH_TRACING=false`, and `PORT=8000`.
No Gemini key is needed. Sessions, evidence, and checkpoints are in memory and
are lost on restart/deploy. One non-replay run may be active globally. Synthetic
tools/gates do not produce validated NBA research. Finish active sessions before
deploying. To roll back, select the previous successful deployment in Railway,
restore compatible variables, and repeat the hosted check; sessions are not restored.
Trial credit is consumed by running compute; stop the service when the demo is no
longer needed and monitor remaining credit in Railway Billing. No paid extras are
required. See [HOSTING_PLAN.md](HOSTING_PLAN.md) for later persistence and access work.

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

The research loop (`backend/agent/`) is a LangGraph stateful router with four LLM roles:

1. **Premise check** (mandatory). The question is restated as a measurable claim and tested
   on the explore seasons. If the pattern isn't there (or can't be tested), the run ends.
2. **Seeding.** A Google Search-grounded web search (`source` node), then up to 3 seeds:
   competing ideas that would answer the question, citing the search results.
3. **Research, one seed at a time.** Each step: the **researcher** calls one toolkit tool
   (`backend/agent/toolkit.py`), the **judge** reviews the step and updates the seed's running
   synthesis (`judgment` node), and the **supervisor** decides to continue with guidance or
   conclude the seed. A seed gets at most 8 steps.
4. **Aggregation.** The final answer covers every seed (`report` node and `GET /runs/{id}/report`).

The researcher chooses tools freely, but the rules are code, not prompts. It must
pre-register a primary test (`declare_primary_test`) before any statistical test on the
seed. The confirmatory tools (`run_primary_test`, `replicate_primary`, `market_check`,
`holdout_confirm`) run that frozen spec once each, and they alone set the verdict through
the stats gate (`backend/gate/`: G1-G7 and the verdict table). Exploratory tests and
`web_search` inform the judge. Every exploratory p-value enters the run's ledger, and G4's
multiple-testing correction is re-run over the whole ledger before the report. The judge's
output is audited: claims must cite evidence ids, and it can't call a seed supported after
a stopping gate failed. In approval mode a human reviews the premise, each seed and each tool
call before it runs. Events follow `contracts/events.py` and reach FastAPI through
`backend/api/agent.py`. Run it with the fake LLM (no API key needed):

```powershell
.\.venv\Scripts\python -m backend.agent                         # live mode
.\.venv\Scripts\python -m backend.agent --mode approval         # approve each decision
.\.venv\Scripts\python -m backend.agent --out replays\fake.jsonl # save the event log
.\.venv\Scripts\python -m backend.agent --tools mock            # Person A's mock tools
.\.venv\Scripts\python -m backend.agent --llm gemini --tools mock # real Gemini decisions
```

`--llm gemini` reads `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env` and sends at most
`GEMINI_RPM` (default 5) requests per minute. Tool choices use Gemini function calling and
searches use Google Search grounding, so the model must support both. Every search counts
as a call: the scripted run makes 36 calls, about 7 minutes at 5 per minute. The CLI default
for `--llm` is `LLM_MODE` from `.env`, so pass `--llm fake` explicitly to avoid spending quota.

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
