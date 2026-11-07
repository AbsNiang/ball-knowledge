# Product and agent contracts

Person C owns this boundary. Python Pydantic models are authoritative; coordinate changes
with A and B before merging. Dataset schemas and tool-specific arguments/results belong
to A and are intentionally outside this change. `ToolRecord.output` carries their JSON
without interpreting it. The existing `TASKS.md` ownership table is superseded by the
role addendum in `plan.md` section 7.1.

## Ownership and integration

- A → B: statistical tool registry, tool results, gates, ledger and verdict labels.
- B → C: typed nodes, edges, checks, budget updates, verdicts and narratives via `EventSink`.
- C → B: `AgentPort.start`, `intervene`, and `report` (`ports.py`). Pass an implementation
  to `backend.api.main.create_app(agent=...)`. `start`/`intervene` enqueue and return
  promptly. B owns workers, failures, terminal status events and intervention semantics.
- Browser → C: HTTP commands and WebSocket subscription. The browser does not write to
  a database. C owns storage, transport, graph/report projection and rendering.

The default app injects `backend.api.agent.AgentService`. Calling `create_app()` without
an agent still returns `503` for live run creation; fixtures/replay work without an agent.
The memory log is for one worker only, resets on restart and is not durable. Do not use
multiple Uvicorn workers until C supplies shared durable storage and subscriptions.

For example, B publishes a complete node snapshot through the supplied run-scoped sink:

```python
from contracts.events import NodeCreated

await events.emit("orchestrator", NodeCreated(node=node), node_id=node.id)
```

`node.run_id` must match the active run; C owns the envelope, not B. Use the same
`Node`/`CheckRecord`/`Verdict` types for fake and real adapters. Until A's result schemas
are agreed, use clearly labelled canned `ToolRecord.output` JSON for integration only.

## HTTP API

All routes use JSON except Markdown export. IDs are opaque strings. Run creation and replay
return `202 {"run_id": "..."}`; interventions return `202 {"intervention_id": "...",
"status": "accepted"}`. Acceptance is not successful completion.

| Route | Request / response |
|---|---|
| `POST /runs` | `CreateRun`: question, mode (`live`/`approval`), optional budget (60 calls, 40 tests, depth 8 by default) |
| `GET /runs/{id}/events?from_seq=1` | Ordered `Event[]`; cursor is inclusive and must be >= 1 |
| `WS /runs/{id}/events?from_seq=1` | Same individual events: backfill then live; reconnect at last applied seq + 1 |
| `POST /runs/{id}/interventions` | Discriminated `Intervention` union below |
| `GET /runs/{id}/report` | `Report`: question, answer, verdict rows with clickable node-ID paths |
| `GET /runs/{id}/report?format=md` | `text/markdown` export |
| `POST /replay` | `{"log_file":"mock.jsonl","speed":1}`; paths relative to `replays/`, positive speed <= 100 |
| `GET /fixtures/events` | Validated fixture for frontend development |
| `GET /health` | Process liveness |

Unknown runs return 404. Reports not ready and invalid-state interventions return 409.
Missing agent integration returns 503. Domain errors use `{"detail":{"code":"...",
"message":"..."}}`. Invalid request shapes use FastAPI's standard 422 validation response.
For an unknown run/invalid WS cursor, the socket sends a domain error then closes with
1008. Sockets remain open after terminal events until the client disconnects.

## Interventions

Every command has `type`, `node_id`, `payload`. `node_id` is required for node commands
and null/omitted for run commands. Unknown fields are rejected.

| Type | Payload |
|---|---|
| `prune`, `approve` | `{ "note": "optional" }` or `{}` |
| `rerun` | `{}` |
| `reject` | `{ "note": "at least 10 characters" }` |
| `override` | `{ "check_id": "...", "note": "at least 10 characters" }` |
| `set_mode` | `{ "mode": "approval" }` or `live` |
| `add_hypothesis` | `{ "statement": "..." }` (P2) |

B validates node/check existence, approval mode and pending state, G7 prohibition, and
run lifecycle before acceptance; raise `ValueError` for 409. B owns subtree effects,
taint/revisions and mode changes at decision boundaries, and emits their resulting events.
C records accepted human commands. Authentication and durable command/outbox transactions
are follow-on work; this skeleton is a local development integration boundary.

## Events and replay

Wire envelope: `{seq, run_id, ts, actor, type, node_id, payload}`. `seq` starts at 1 per run;
timestamps are UTC; `actor` identifies the source. Both envelope `type` and `payload.type`
must match. The payload discriminator makes validation and generated TS narrowing explicit.
The old `kind`/`data` fixture format is retired.

Payloads cover `run_created`, `run_status`, `node_created`, `node_updated`, `edge_created`,
`check_result`, `ledger_entry`, `verdict_computed`, `intervention`, `budget_update`,
`mode_changed`, and `error`. `node_updated.node` is a complete snapshot, not a partial patch.
Node lifecycle, check outcome and flags are separate fields. Revisions get new IDs;
`revises` links new → old. `derives_from` links child → parent (plan.md section 5.3).
Consumers deduplicate by `(run_id, seq)` and apply in sequence. Only canonical backend
snapshots determine graph state; clients do not compute gate verdicts or taint.

Replay validates an ordered, single-run JSONL log, returns a new run ID, and publishes
events with original relative delays divided by speed. Node run IDs are remapped;
node/edge IDs and payload meaning are preserved. No agent calls or interventions occur.
Replay report projection is not implemented yet (409); replay is currently graph-only.

## Schema generation

From the repository root, after installing Python/frontend development dependencies:

```powershell
.\.venv\Scripts\python -m contracts.export_schema
npm --prefix frontend run contracts
```

Commit `schema.json` and `generated.ts` with model changes. `events.ts` re-exports the
generated event type for existing consumers. HTTP OpenAPI is at `/openapi.json` and
interactive documentation at `/docs`; WebSocket messages use the same `Event` model.

## Research workspace transport additions

`POST /demo/runs` accepts `CreateRun` and starts a clearly labeled synthetic demo through
the real event stream. It never calls an LLM or statistical tool. Live creation remains
`POST /runs`. A single-process lock prevents starting/resuming a second active run.
Paused sessions may coexist. This is workspace-wide enforcement, not multi-user isolation.

The event socket also accepts:

```json
{"request_id":"client-generated-id","command":{"type":"pause","node_id":null,"payload":{}}}
```

Commands use the existing `Intervention` union, extended with `pause`, `resume` (run
commands) and `redirect` (`node_id` required, payload `{ "statement": "new question" }`).
Replies are `{type:"command_ack",request_id,intervention_id,status:"accepted"}` or
`{type:"command_error",request_id,detail:{message,...}}`, separate from sequenced events.
The same commands are accepted over the existing HTTP intervention endpoint.
Acknowledgement means accepted; run/node events determine actual UI state. `RunStatus`
now includes `paused`; edges accept optional `label` text. Python models, JSON Schema,
and generated TypeScript are updated together.

Transport liveness uses `{ "type": "ping" }` and `{ "type": "pong" }` frames outside
the sequenced event log. The browser sends a ping every 15 seconds and reconnects if
there is no pong within 5 seconds. Handshakes time out after 8 seconds. Unknown-run
errors remove only that saved session; other errors and event gaps remain visible.

The default agent adapter forwards the standalone log in order through `EventSink`,
skipping its duplicate creation/intervention envelopes because the API assigns those.
It supports pause/resume, approve/reject, and set-mode. Unsupported commands return
409; redirect remains demo-only. It uses the CLI's default stub tools/gates and the
configured fake/Gemini gateway. No event model or generated contract changed.

Agent pause/resume must emit `paused` only after the current step has stopped advancing,
preserve the checkpoint, and resume from it. For a
running external model/tool call, cancellation or boundary pausing is the agent's
responsibility; the frontend cannot suspend an external model mid-token. Redirect implementations must
validate the active node, preserve prior nodes, and emit the new branch with a decision
label. The scripted adapter implements these semantics for demonstration.

Evidence charts use optional `node.tool.output.charts` entries:

```json
{"type":"bar","title":"Effect by sample","xKey":"sample","yKey":"effect","data":[{"sample":"Explore","effect":-2.4}]}
```

`type` supports `bar` and `line`. Invalid charts are skipped; raw tool JSON remains
inspectable. Gate outcomes and verdicts are rendered from backend events, never inferred
from chart values. Layout reverses child-to-parent `derives_from`/`revises` edges for
presentation while retaining the original records. Failed/pruned nodes are never removed.
