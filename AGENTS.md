# Repository Guidelines

## Project Structure & Module Organization

Whack is a planned NBA research assistant using deterministic statistical tools, a Gemini orchestrator, and a live research graph. Currently, the repository contains documentation only: `README.md` introduces the project, `plan.md` details requirements and architecture, and `TASKS.md` assigns ownership and milestones.

The proposed stack is Python/FastAPI with React, Vite, and React Flow. `TASKS.md` proposes `backend/` for data, tools, gates, LLM integration, agents, events, API, and tests; `frontend/` for the UI; `contracts/` for shared Python/TypeScript schemas; and `replays/*.jsonl` for recorded events. `plan.md` uses some different paths; reconcile these before scaffolding. No source, tests, or asset directories exist yet.

## Build, Test, and Development Commands

No dependency manifests, build scripts, or runnable application commands are configured. Do not assume `npm test`, `pytest`, or a server command works yet. When adding scaffolding, document exact installation, development, build, and test commands in `README.md`. Develop the pipeline with scripted fake LLM outputs before using live Gemini calls.

## Coding Style & Naming Conventions

No formatter or linter is configured. For new code, use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Use `PascalCase` React components and `camelCase` TypeScript functions/variables. Keep statistical tools deterministic and seeded. Coordinate shared contract changes with all owners and update both Python and TypeScript definitions together.

## Testing Guidelines

No testing framework or coverage threshold has been selected. Place planned backend tests in `backend/tests/`; use descriptive names such as `test_holdout_second_call_fails`. Prioritize spread signs and pushes, no look-ahead, join integrity, deterministic tools/reducers, global multiple-testing correction, and sealed-holdout access. Include planted-noise and planted-signal evaluations, plus replay, approval, and taint-propagation checks.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as `Add team task plan and ownership split`; no strict commit format is established. Use concise imperative subjects, short-lived branches, and small changes. PRs should describe behavior, validation performed, linked tasks, and contract changes; include screenshots for UI changes. Keep main usable as implementation begins.

## Security & Configuration

Keep Gemini keys out of commits and replay logs. Use individual development keys and configurable model IDs. Route LLM calls through one gateway; preserve append-only evidence logs and gate authority over LLM judgments.
