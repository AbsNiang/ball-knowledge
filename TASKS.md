# Build Plan & Ownership

Spec: [plan.md](plan.md). Times are hours from the start of the hackathon (H0–H24).

## Roles
| Role | Owns | In short |
|---|---|---|
| **A: Data & Stats** | `backend/data/`, `backend/tools/`, `backend/gate/` | Makes the numbers right. No LLM. |
| **B: Agent & Backend** | `backend/llm/`, `backend/agent/`, `backend/events/`, `backend/api/` | Makes the agent think, and streams everything it does. |
| **C: Frontend** | `frontend/` | Makes the reasoning visible and lets the user step in. |
| **Shared** | `contracts/` | Any change needs all three to agree, and both the Python and TS copies get updated. |

**With 2 people:** A also owns the critic prompt and the report generator. B owns the backend and frontend, with charts cut to the CI bar only.

## Repo layout
```
contracts/   events.py (pydantic) + events.ts   ← frozen at H2
backend/     data/ tools/ gate/ llm/ agent/ events/ api/ tests/
frontend/    Vite + React + React Flow
replays/     *.jsonl event logs (mock + recorded demo runs)
```

## H0–H2: everyone together (contracts)
1. **Event** `{seq, run_id, ts, kind, data}`. `kind` is one of `node_created | node_updated | edge_created | check_result | intervention | run_status`.
2. **Node** `{id, type, state, parent_ids, rationale, tool_call, tool_output, data_hash, checks[], verdict}`.
3. **ToolResult** `{tool, args, n, effect, ci_low, ci_high, p_value, slice_hash, extra}`. A produces it, and B and the gate consume it.
4. **CheckResult** `{node_id, checker: gate|critic|human, gate_id, status: pass|flag|fail, severity, rationale}`.
5. **Tool registry**: a list of `{name, description, json_schema}` that B passes to Gemini function calling. A owns it.

**Exit criteria (so nobody waits on anyone):**
- B commits `replays/mock.jsonl`, a hand-written run that C builds the UI against.
- A commits stub tools with the real signatures that return canned `ToolResult`s, which B builds against.
- Everyone clones `AbsNiang/whack` and has push access.

## Task lists
**A: Data & Stats**
- [ ] H2–5: Join games with SBR closing odds (2007→) and map team renames. Add a join-rate assertion. ⚠ **Critical path**
- [ ] H5–7: Features computed as-of tip-off: rest_days, b2b, travel_km, tz_shift, altitude
- [ ] H7–11: Tools `describe_dataset`, `define_slice`, `add_feature`, `compare_groups`, `regress`, `market_test`, `persistence`, `power`
- [ ] H11–15: Gate G1–G7 and the global BH test ledger
- [ ] H15–17: `out_of_sample`, `holdout_confirm` (sealed seasons, once per hypothesis), `placebo`
- [ ] Throughout: tests for the ATS sign convention on 5 known games, no look-ahead, and determinism
- [ ] H18–20: Planted-noise FDR script. Its output is the headline number for the pitch.

**B: Agent & Backend**
- [ ] H2–5: Gemini gateway with token bucket, prompt-hash cache, 429 backoff, priority queue and call budget
- [ ] H2–5: Append-only event log and graph projection
- [ ] H5–7: FastAPI routes `POST /runs`, `WS /runs/{id}/events`, `POST /interventions`, `GET /report`
- [ ] H7–11: Orchestrator loop: premise check → 3–5 hypotheses (including chance and confounder) → tools → interpretation → verdict
- [ ] H11–14: Async critic with an adversarial checklist and JSON output. It can only downgrade a node.
- [ ] H14–16: Taint propagation. Flagged nodes must be revised, pruned or justified.
- [ ] H16–18: Approval mode (blocking, switchable mid-run) and the Markdown report
- [ ] H20–22: Record 2 replays of the demo question

**C: Frontend**
- [ ] H2–6: React Flow with dagre layout, WS client, colour by node state, verdict badges (against `mock.jsonl`)
- [ ] H6–10: Inspector tabs: Reasoning, Tools, Evidence, Checks, Notes
- [ ] H10–14: Prune, re-run, override (note required), approve/reject
- [ ] H14–16: Top bar with mode toggle, LLM budget meter and tests-spent counter
- [ ] H14–18: Charts: CI bar, null distribution, in-sample/OOS/holdout bars
- [ ] H16–20: Report view where clicking a path highlights it in the graph
- [ ] H18–21: Replay mode with a speed control

## Sync points
| Time | Milestone | Done when |
|---|---|---|
| H2 | Contracts frozen | `contracts/`, `mock.jsonl` and the tool stubs are on main |
| H6 | Checkpoint | A has the table joined and 3 real tools. B has the orchestrator calling stubs and streaming over WS. C renders `mock.jsonl` live. |
| H10–14 | **Integration** | One question runs live, end to end, in the UI |
| H18 | Feature complete | Approval mode, taint, critic, all gates and the report work |
| H22 | Code freeze | 2 replays recorded and the FDR number computed |
| H22–24 | Pitch | Rehearse on a replay, and keep the live run as a bonus |

## Rules
- Main must always run. Each person works on short-lived branches and merges small changes often.
- Each person uses their own Gemini key for development. Nobody touches the demo key until H20.
- Gate thresholds belong to A, prompts belong to B, and the UI wording belongs to C.
- **Fallback:** if the odds join isn't done by H5, A ships one season, B and C carry on with stubs, and A finishes the join in parallel.
- Hold a 5-minute standup at each sync point to report what's blocked and any contract changes.
