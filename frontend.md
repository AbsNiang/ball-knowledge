# Frontend MVP: implementation and design review

This document describes the implementation reviewed on 7 November 2026. The result is a functional research workspace with a scripted demo streamed over real WebSockets. Live autonomous research still requires an injected backend agent. The improvements below are recommendations, not completed changes.

## What was implemented

| Area | Current behavior |
| --- | --- |
| Workspace | Dark, viewport-filling interface with a collapsible session sidebar, graph canvas, question composer, and right-hand inspector. |
| Sessions | Create sessions and switch between running, paused, and completed sessions. Browser local storage retains session IDs, questions, and demo flags. Graph contents reload from server events. |
| Graph | React Flow with Dagre top-to-bottom layout, labeled edges, pan/zoom controls, and a minimap. Node deletion, manual connections, and dragging are disabled. Failed and interrupted nodes remain visible. |
| Node cards | Show the step title, type, status, and aggregate check outcome. Running nodes have a pulsing border; reduced-motion preferences disable animation. |
| Inspector | Clicking a node opens Research, Evidence, and Checks tabs. Research renders Markdown and GFM tables. Evidence supports JSON-fed bar and line charts through Recharts, plus raw tool output. Checks show backend check records and explanations. |
| Controls | Natural-language session creation, pause, resume, and redirect from the active node. Buttons wait for command acknowledgement or a timeout; backend events determine displayed research state. |
| Connection loss | Socket errors or closure dim and block the workspace with “Refresh the page to reconnect.” There is no automatic retry. A secondary action clears the browser's session list when the server no longer has those sessions. |
| Styling | Tailwind is integrated through Vite. Most detailed component styling currently lives in custom CSS. Lucide supplies icons. Responsive rules adapt the sidebar and inspector. |

The backend was also extended to make the interaction demonstrable: a synthetic demo adapter, a demo creation endpoint, WebSocket command handling, and single-process active-session checks. Shared Python models, JSON Schema, and generated TypeScript were updated together.

## How it works

1. The composer creates a session over HTTP: `POST /demo/runs` for synthetic research or `POST /runs` for an injected live agent.
2. The browser opens `/runs/{id}/events?from_seq=1` through the `/api` proxy. The socket sends existing events, then live updates.
3. `frontend/src/state.ts` reduces events into session state. It ignores duplicate sequence numbers and events for other sessions, and rejects sequence gaps. Node snapshots replace prior snapshots; revisions with new IDs remain separate nodes.
4. The graph renders that state. Existing `derives_from` and `revises` records point child-to-parent; their displayed direction is reversed for downward layout. Server edge labels are preferred, with the relationship type as a fallback.
5. Pause/resume/redirect travel back over the socket as `{request_id, command}`. The server validates the intervention, delegates to the agent, records acceptance, and sends an acknowledgement or error. Acceptance alone does not mean execution has finished.
6. On refresh, session metadata is restored locally and evidence is reconstructed from the server log. That log is currently in memory and disappears on backend restart.

The important architectural boundary is that the frontend displays backend evidence. It does not calculate statistical significance, verdicts, or taint. Retaining failed nodes makes prior work inspectable; it does **not** itself prevent the agent from repeating a failed hypothesis.

The demo uses a fixed NBA-themed sequence regardless of the submitted question, with canned Markdown, checks, and chart values. Pause stops its progress counter. Redirect retains the active node with a `pruned` flag, inserts a new child at the front of the queue, and resumes execution. The remaining original queue is retained. This demonstrates interaction mechanics, not question-dependent research.

## What I understood, and what remained unclear

I understood the central requirement as an inspectable, growing research history: users should see successful and unsuccessful paths, examine supporting evidence, and intervene without destroying that history. The implementation supports that model.

I made the following assumptions where the requirements or existing backend did not establish complete semantics:

| Unresolved point | Assumption or limitation in this implementation |
| --- | --- |
| “Freeze the AI mid-thought” | The demo can pause its own execution. A real model request cannot be frozen by the browser. The live agent must define cancellation, checkpointing, and whether pause takes effect immediately or at a safe execution boundary. |
| Redirect's effect on old work | The demo interrupts the current step, runs the new branch next, then continues queued work. Whether redirect should abandon, suspend, or preserve that queue was not established. |
| Gate versus confidence | No calibrated confidence score or explicit continuation permission exists in the current contract. The card labels `check_outcome` as “Gate,” although that aggregate can include critic/human outcomes. It is not a probability or a definitive “allowed to continue” field. |
| One active session | Enforcement is server-process-wide, rather than per user or workspace. An asynchronous live agent must publish state promptly for this event-based check to remain reliable. |
| Past sessions | Browser-local metadata plus an in-memory server log was treated as sufficient for the MVP. Durable history, accounts, and cross-device discovery are not implemented. |
| Comprehensive node details | The inspector exposes rationale, verdict, notes, checks, charts, and tool output. It does not yet expose every provenance field, such as tool arguments, data hashes, seed, or model metadata. |
| Click/zoom expansion | Clicking opens details; zooming only magnifies the canvas. Automatic detail expansion at a zoom threshold is not implemented. |
| Visual direction | The dark research-console design was my choice. It was not a confirmed user preference, and browser visual validation was unavailable. |

## Improvements I would prioritize

### 1. Make controls and evidence labels precise

- Replace the ambiguous “Gate” label with separate backend-authored gate status, critic status, and continuation permission. Show a numeric confidence value only if the backend defines its meaning.
- Add explicit requested/accepted/applied states for pause and redirect. Preserve redirect text until acceptance so rejected commands do not discard the user's question.
- Define whether redirect replaces or suspends queued work. Use a dedicated interrupted/superseded state instead of displaying every `pruned` node as “interrupted.”
- Make the demo's fixed scenario explicit in question intake. Arbitrary input currently changes the root title without changing the research sequence.

### 2. Strengthen session and transport behavior

- Extract a connection manager that retains existing sockets when a session is added. Currently, changing the session-ID list closes and reopens all sockets and replays every log from sequence 1.
- Track connection and pending-command state per session/request. The current global `connected` and `busy` flags can misrepresent concurrent session activity; acknowledgements are not fully isolated by pending request.
- Validate complete incoming event and acknowledgement shapes at runtime, rather than relying mainly on TypeScript and shallow envelope checks. Add request deduplication server-side for safe retries.
- Add heartbeat/liveness detection for silent network failures. The overlay appears immediately after a socket error/close, but a silent lost connection may take time to generate either event. Preserve the requested manual-refresh recovery behavior.
- Persist events and session metadata server-side. Use an authoritative active-run reservation for asynchronous agents and scope it to the intended user/workspace. The current lock serializes requests, but does not reserve a pending resume while an agent delays publishing its new status.

### 3. Improve graph readability and navigation

- Measure actual node sizes for layout. Dagre currently assumes 288 × 158 pixels, while longer titles can produce taller cards. Account for edge-label dimensions and wrap long redirect labels.
- Recalculate layout only when topology or dimensions change. The current reducer clones node/edge maps for every event, which triggers layout even for unrelated status updates and can move established branches.
- Preserve each session's viewport and add “Focus active step,” visible node selection, and optional follow mode. Do not force users back to a new camera position while they inspect history.
- Add distinct visuals for flagged, checking, approval-required, overridden, and stale states. The current card presentation compresses several states and styles flagged checks like failures.
- Allow reversible branch collapsing for large graphs while keeping the evidence and node count accessible. Collapsing must never delete failed work.

### 4. Improve maintainability, accessibility, and validation

- Split `App.tsx` into the workspace shell, session sidebar, graph, inspector, composer, and transport hook. Format the compressed JSX/CSS and centralize repeated design tokens and state-to-label mappings.
- Lazy-load the inspector/chart libraries. The last build emitted an approximately 1.04 MB JavaScript bundle before gzip, with a bundle-size warning.
- Complete keyboard behavior for inspector tabs and dialogs, add textual chart alternatives and explicit active-state announcements, and test long questions on narrow screens.
- Add browser tests for start → pause → redirect → resume, switching sessions, rejected commands, and disconnect recovery. Also test measured node spacing, table overflow, reduced motion, and keyboard focus. The current tests do not establish visual or accessibility correctness.

## Validation and development

The implementation run completed **20 backend tests**, **3 frontend reducer tests**, a TypeScript/Vite production build, and Ruff checks on the touched Python areas. The new backend test exercises WebSocket pause/redirect, retained history, rejection of a second active demo, and invalid-command errors. Reducer tests cover deterministic replay, deduplication, sequence gaps, session isolation, and check-record replacement without deriving verdicts.

Browser visual testing was not performed. This document was produced by reviewing the source; it does not claim that the recommendations above have been implemented or tested.

From the repository root, start the backend:

```powershell
.\.venv\Scripts\python -m uvicorn backend.api.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` and leave **Scripted demo** checked. Frontend verification commands are `npm run test` and `npm run build`. See `README.md` for environment setup and `contracts/README.md` for transport details.
