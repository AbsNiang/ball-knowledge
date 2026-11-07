# Requirements: AI Researcher for NBA Betting Patterns (Hackathon MVP)

## Context
Hackathon brief: *"Finding patterns in data is easy; knowing which ones to trust is hard. Build an AI researcher that tests its own ideas and can tell a real discovery from a lucky one."*

We are building an agent that takes a user's **"why" question** about NBA games/betting (e.g. "Why do teams on a back-to-back underperform?"), generates hypotheses with an LLM, tests them with **hard-coded statistical tools**, and checks **every LLM decision asynchronously** with (a) a deterministic stats gate and (b) an LLM critic. The reasoning is shown as a **live graph** that a human can watch, inspect, and step into. The output is a verdict per hypothesis plus an evidence trail that a judge can follow and check.

Workspace `/Users/alexhossien/whack` is empty (greenfield, not a git repo yet).

## Decisions made
| Area | Decision |
|---|---|
| Domain | NBA basketball betting (regular season + playoffs) |
| Data | Game results/box scores + closing odds (spread, total, moneyline), 2007-08 onward |
| Team/time | ~24h, 2–3 people |
| Verification | Deterministic stats gate + LLM critic; human can override |
| Human role | Watch live + intervene by default; **toggle to full approval mode** |
| LLM | Gemini free tier (likely Flash-only; Pro is reportedly not free since Apr 2026) |
| Stack | FastAPI (Python) + WebSocket + React + React Flow |
| Agent freedom | Hard-coded tools only, no free-form code execution |

---

## 1. System overview

```
 User "why" question
        │
        ▼
 ┌──────────────┐  tool calls   ┌─────────────────┐
 │ Orchestrator │──────────────▶│ Tool executor    │  pandas / scipy / statsmodels
 │ (Gemini)     │◀──────────────│ (pure Python)    │  deterministic, seeded
 └──────┬───────┘   results     └────────┬────────┘
        │ every decision = event          │ every result = event
        ▼                                 ▼
 ┌─────────────────────────────────────────────────┐
 │ Append-only event log  →  Graph store (nodes)   │──WS──▶ React Flow UI
 └──────┬──────────────────────────────┬───────────┘        (live graph,
        ▼ async                        ▼ async               inspector,
 ┌──────────────┐               ┌──────────────┐             interventions,
 │ Stats gate   │               │ LLM critic   │             report)
 │ (no LLM)     │               │ (Gemini)     │
 └──────────────┘               └──────────────┘
        All LLM calls go through ONE rate-limited, cached LLM gateway
```

---

## 2. Functional requirements

### FR1: Question intake and premise check
- FR1.1 The user submits a free-text "why" question and picks a mode (Live / Approval).
- FR1.2 **The first node is always a Premise Check.** The agent restates the question as a measurable claim (e.g. "B2B 2nd-night teams have a lower ATS cover rate") and tests whether the pattern exists at all. If the premise fails, the run ends with the verdict *Premise false*, and the agent does not go on to explain noise.
- FR1.3 If the question can't be tested with the available data/tools (e.g. it needs player tracking), the agent returns *Untestable* and says what data would be needed. It must not substitute a proxy without labelling it as one.

### FR2: Research graph
- FR2.1 **Node types:** Question → Premise Check → Hypothesis → Test Plan → Test Result → Mechanism Prediction → Verdict. Human Note nodes can be attached to any of them.
- FR2.2 **Edge types:** `derives_from`, `tests`, `supports`, `refutes`, `revises` (agent rewrote a node after critique).
- FR2.3 **Node states:** `proposed → running → checking → passed | flagged | failed`, plus `tainted` (an ancestor later failed a check), `pruned` (by the human or agent), `awaiting_approval` (approval mode), and `overridden` (human overrode a check, with a mandatory note).
- FR2.4 Each node stores: the LLM prompt and response, the structured rationale, tool name, args and output, a hash of the data slice, the check records (gate results and critic review), the model id, timestamps, and human annotations. This is the provenance that makes the result checkable.
- FR2.5 Seed hypotheses: the orchestrator generates 3–5 competing hypotheses per question, including at least one "it's just chance" or confounder hypothesis.

### FR3: Orchestrator agent
- FR3.1 Gemini function-calling loop over the tool library (FR4). Outputs use a JSON schema (structured output), and every decision includes a `rationale` field.
- FR3.2 Each LLM decision (generate hypothesis, choose test, interpret result, assign verdict) emits a graph event and triggers the async checks (FR5).
- FR3.3 The orchestrator reads check outcomes on its next step. A flagged or failed node must be **revised, pruned, or justified** before it can be built on.
- FR3.4 There is a hard budget per run: max LLM calls, max tests spent, and max graph depth, all shown in the UI.
- FR3.5 Stopping rule: every hypothesis has a verdict, or the budget is exhausted. The agent then writes the report (FR7).

### FR4: Hard-coded tool library (Python, deterministic, seeded)
Data and features (all features are **as-of tip-off**, so there is no look-ahead):
| Tool | Purpose |
|---|---|
| `describe_dataset()` | Schema, seasons, row counts, available features |
| `define_slice(filters)` | Filter on whitelisted columns (season, team, home/away, rest_days, b2b, travel_km, tz_shift, altitude, month, is_bubble, is_playoffs). Returns slice_id and n |
| `add_feature(name, params)` | From a fixed menu: rest days, B2B flag, travel distance, time-zone shift, altitude, rolling ATS record, rolling margin, days into season |

Tests:
| Tool | Purpose |
|---|---|
| `compare_groups(metric, slice_a, slice_b)` | Effect size + bootstrap CI + **block permutation p-value** (permute within season) |
| `regress(metric, features, controls)` | OLS/logit with cluster-robust SEs (by team-season); **always offers closing spread as a control** |
| `out_of_sample(test_id)` | Re-runs the same spec on the validation seasons |
| `holdout_confirm(hypothesis_id)` | Runs on the **sealed holdout** seasons. Allowed **once per hypothesis** and logged |
| `persistence(metric, entity)` | Season t → t+1 correlation, for "Team X beat the odds" style questions (regression to the mean) |
| `market_test(slice, side)` | ATS cover rate vs the **52.38% break-even at -110**, ROI backtest, over/under vs the closing total |
| `power(n, effect)` | Minimum detectable effect, which flags underpowered tests |
| `placebo(test_id)` | The same test on a period or group where the effect shouldn't exist (e.g. the bubble effect tested on non-bubble months) |

Metrics: `home_win`, `margin`, `ats_cover` (= margin + spread > 0; push if = 0), `total_over`, `roi`.

### FR5: Verification layer (asynchronous)
**FR5.1 Stats gate.** Deterministic, no LLM, runs on every Test Result:
| Gate | Rule |
|---|---|
| G1 Sample/power | n ≥ min, and the effect is above the minimum detectable effect; otherwise *Inconclusive* |
| G2 Uncertainty | CI excludes 0 |
| G3 Null | Beats the block-permutation null (p < α) |
| G4 Multiple testing | Survives **Benjamini–Hochberg across every test run in the whole graph** (a global ledger, not per node) |
| G5 Replication | Same sign, and ≥ 50% of the in-sample magnitude, on the validation seasons |
| G6 Market | For a "beats the market" claim: ATS > 52.38% with CI, and the effect survives controlling for the closing spread |
| G7 Holdout | Final confirmation on the sealed seasons, run once only |

**FR5.2 LLM critic.** Runs on every LLM decision, adversarial, with a fixed checklist and JSON output:
- Does the test actually measure the hypothesis (construct validity)?
- Is a confounder missing (team quality, schedule, market line)?
- Is there method or selection bias (the hot-hand-style trap)?
- Does the interpretation overclaim, given the gate results?
- Does the proposed mechanism make a **testable secondary prediction**?
- Output: `approve | concerns | reject`, with a severity, a rationale, and a suggested fix.

**FR5.3 Authority rules.** The critic can *downgrade* a node but can never *upgrade* one that failed the stats gate. Only a human can override, and the override is shown on the node with the human's note.

**FR5.4 Async semantics.** In Live mode the orchestrator continues optimistically. If a check fails later, the node is marked failed, **all descendants are marked `tainted`**, and the orchestrator must handle this on its next step. In Approval mode, each LLM decision blocks until its checks complete **and** the human approves it.

### FR6: Verdict taxonomy
| Verdict | Meaning |
|---|---|
| Premise false | The pattern isn't in the data |
| Untestable | Can't be tested with the available data/tools |
| Inconclusive | Underpowered (G1 failed) |
| Luck | Fails G2–G5 (doesn't survive the null, the correction, or replication) |
| Real but priced in | Passes G1–G5 but fails G6, so there's no edge after the closing line and vig |
| Real and beats market (provisional) | Passes G1–G7. Shown with the strongest caveats |

Every verdict also carries an **explanation status**. *Mechanism supported* means a secondary prediction was tested and passed; *Mechanism plausible but untested* means it wasn't tested.

### FR7: Human harness (React Flow UI)
- FR7.1 Live graph with auto-layout (dagre/ELK). Node colour shows state and a badge shows the verdict. It updates over WebSocket.
- FR7.2 Node inspector side panel with tabs: **Reasoning** (LLM rationale), **Tools** (calls and outputs), **Evidence** (charts), **Checks** (each gate pass/fail plus the critic review), **Notes**.
- FR7.3 Evidence charts (minimum set): an effect-size CI bar, the null distribution with a line for the observed value, and in-sample vs out-of-sample vs holdout bars.
- FR7.4 Interventions: **prune** a branch, **re-run** a node, **add** a human hypothesis, **approve/reject** (approval mode), and **override** a check (a note is required). All are logged as events.
- FR7.5 Top bar: mode toggle (Live ⇄ Approval, switchable mid-run), LLM budget meter, "tests spent" counter (the multiple-testing cost), run status.
- FR7.6 Report view: an answer to the why question, a verdict table, and for each verdict the **path through the graph** with clickable nodes that highlight in the graph. Exportable as Markdown.
- FR7.7 **Replay mode:** load a recorded event log and replay it at chosen speed, with no LLM calls (the demo fallback).

### FR8: Backend API
- `POST /runs` {question, mode, budget} → run_id
- `WS /runs/{id}/events`: a stream of graph events (node_created, node_updated, check_result, edge_created, run_status)
- `POST /runs/{id}/interventions` {type, node_id, payload}
- `GET /runs/{id}/report`, `GET /runs/{id}/events` (for replay/export)

---

## 3. Non-functional requirements
- **NFR1 Rate limits.** The Gemini free tier is roughly 10–15 RPM and ~1,500 RPD for Flash (third-party figures; **check the real numbers in AI Studio → Rate limit page**). Design for this:
  - One LLM gateway with a token bucket matched to the project's RPM, a priority queue (in Live mode the orchestrator goes first; in Approval mode critics go first because they're blocking), and exponential backoff on 429.
  - Responses cached by prompt hash, so re-runs and dev iterations are free.
  - Budget of about 40–60 LLM calls per run, which is roughly 5–6 minutes per run at 10 RPM. Graph streaming keeps the UI busy while it waits.
  - Each teammate develops with their own key/project so development doesn't use up the demo key's daily quota.
- **NFR2 Reproducibility.** The tools are pure functions with fixed seeds. Any node can be re-executed from its stored args and gives the same result.
- **NFR3 Auditability.** The append-only event log is the source of truth, and the graph state is derived from it.
- **NFR4 Demo safety.** Record at least 2 full successful runs as event logs before the pitch (replay mode).
- **NFR5 Data privacy.** Free-tier prompts may be used by Google, which is fine because we only send public NBA data and never private data.

---

## 4. Data requirements
- **Sources:** game results/box scores from the Kaggle NBA games dataset (Nathan Lauga) or a pre-downloaded nba_api pull. Closing spread, total and moneyline from the Sportsbook Reviews Online NBA odds archive (2007-08 onward; **check where coverage ends when downloading**).
- **Canonical `games` table:** game_id, date, season, home, away, home_pts, away_pts, close_spread_home, close_total, ml_home, ml_away, is_playoffs, is_bubble, plus derived rest_days, b2b, travel_km, tz_shift, and arena altitude for each team.
- **Join validation:** team relocations and renames (Seattle→OKC 2008, NJ→Brooklyn 2012, New Orleans Hornets→Pelicans 2013, Bobcats→Hornets 2014), missing odds, and pushes. Write an assertion test for the join rate.
- **Time-based split (no look-ahead):** *explore* = 2007-08 to 2015-16, *validate* = 2016-17 to 2018-19, *special period* = 2019-20 bubble and 2020-21 (limited fans), *sealed holdout* = 2021-22 onward. The agent can't see holdout results except through `holdout_confirm`.
- **Do the data prep before the hackathon if the rules allow it.** If they don't, it's the first 3–4 hours of person A's time.

---

## 5. Oversights and risks (with mitigations)
1. **A "why" question assumes the pattern exists.** Without FR1.2, the agent will explain noise. *Mitigation:* mandatory Premise Check node.
2. **Forking paths / p-hacking by iteration.** An agent that tries a test, fails, tweaks it and retries is p-hacking, even when every individual test looks fine. *Mitigation:* a global test ledger with BH correction across the whole graph (G4), and a visible "tests spent" counter.
3. **Holdout leakage.** If the agent can look at the holdout repeatedly, it stops being a holdout. *Mitigation:* the sealed seasons are reachable only through `holdout_confirm`, once per hypothesis, and every use is logged.
4. **Look-ahead bias.** Rolling features computed with same-game data inflate results. *Mitigation:* tools compute features as-of tip-off, plus unit tests for this.
5. **Real ≠ profitable.** Most real NBA effects (rest, home court) are probably already priced into the line, and at -110 you need 52.38% to break even. *Mitigation:* the separate *Real but priced in* verdict, and spread-as-control in `regress`.
6. **An LLM checking an LLM.** On the free tier the critic is probably the same Gemini Flash model as the orchestrator, so they share blind spots. *Mitigation:* the deterministic gate has final authority (FR5.3), the critic uses an adversarial checklist rather than "is this OK?", the critic must cite specific evidence, and the critic gets a different system prompt and doesn't see the orchestrator's chain of reasoning.
7. **"Explainable" ≠ causal.** LLMs can produce a plausible story for any pattern. *Mitigation:* a mechanism only counts if it makes a secondary prediction that is then tested (e.g. "if it's fatigue, the effect should be larger after longer travel").
8. **Async consistency.** A check that fails *after* child nodes were built leaves invalid conclusions downstream. *Mitigation:* taint propagation (FR5.4), and the report excludes tainted nodes unless they've been re-validated.
9. **Non-independent observations.** Games share teams and seasons, so i.i.d. p-values are too optimistic. *Mitigation:* block permutation within season, and cluster-robust SEs.
10. **Small-n slices.** Altitude is n = 2 teams, and the bubble is a short period. *Mitigation:* power gate G1, the *Inconclusive* verdict, and `placebo`. The bubble finding can be checked again against 2020-21 (limited fans).
11. **Spread sign conventions and pushes** are a classic silent bug. *Mitigation:* one canonical definition (`ats_cover = margin + close_spread_home > 0`), with unit tests on known games.
12. **Rate limits during a live demo.** A 429 halfway through the pitch kills the demo. *Mitigation:* NFR1, NFR4 and replay mode.
13. **Scope creep in questions.** Users will ask player-level questions ("Why does LeBron…"). *Mitigation:* the *Untestable* verdict, and a curated list of example questions in the UI.
14. **Human overrides weaken trust if hidden.** *Mitigation:* overrides need a note and stay visible on the node and in the report.
15. **Proving the system works.** Judges will ask "how do you know it's right?" *Mitigation:* see Verification below (known-answer questions and a planted-noise false discovery rate test).
16. **Responsible gambling.** Add a short disclaimer: this is a research tool, not betting advice.
17. **Model availability.** Free-tier models and limits change with little notice. *Mitigation:* keep the model id in config, and have a paid key or a second provider ready as backup.

---

## 6. Scope: MVP vs stretch
**MVP (must demo):** one question end to end. That means a premise check, 3–5 hypotheses, tools FR4 (all), stats gate G1–G7, the critic, the live graph and inspector, prune/re-run/override, the approval-mode toggle, the report, and replay mode.

**Stretch:** the human adds a hypothesis, charts beyond the minimum set, a side-by-side comparison of runs, a critic on a different model family, play-by-play data (the hot-hand question), and a synthetic benchmark dataset.

## 7. 24h plan (2–3 people)
| Hours | Person A: data + tools + gate | Person B: agent + backend | Person C: frontend |
|---|---|---|---|
| 0–2 | **All three together:** agree the event schema, node schema and tool signatures (contracts first) | | |
| 2–10 | Data join and features, tools, stats gate, unit tests | LLM gateway, orchestrator loop, critic, event log, FastAPI + WS | React Flow graph, inspector, built against a mocked event stream |
| 10–14 | **Integrate end to end on one question** | | |
| 14–18 | Market/placebo tools, BH ledger | Approval mode, taint propagation, report | Report view, approval UI, charts |
| 18–22 | Known-answer runs, FDR test | Record replay logs | Replay mode, polish |
| 22–24 | **Pitch prep and rehearsal on a replay** | | |
With only 2 people, B also takes the inspector and report view, and C's charts are cut to the minimum set.

---

## 8. Verification (how we'll know it works)
- **Unit tests (A):** ATS/push sign conventions on 5 known games, no look-ahead in rolling features, join-rate assertion, and tool determinism (same args → same output).
- **Planted-noise FDR test (no LLM cost):** generate 100 random team-season "coin-flip" features and run each through the full stats gate. The share that passes G1–G6 should be ≈ ≤ 5%. Show this number in the pitch as evidence the gate works.
- **Known-answer questions (end to end):**
  - "Why did home-court advantage disappear in the 2020 bubble?" → expect *Real* (existing research)
  - "Why did [a team with a high ATS record in season t] cover so often?" → expect *Luck* (fails `persistence`)
  - "Why do teams on the 2nd night of a back-to-back underperform?" → expect *Real but priced in*
  - If our data disagrees with these expectations, that's fine as long as the evidence chain explains why.
- **Async/taint test:** force a critic reject on a mid-graph node and confirm its descendants turn `tainted` and the orchestrator revises them.
- **Approval-mode test:** toggle mid-run and # AI Researcher for NBA Betting Patterns: Implementation Spec (v2)

> Audience: the coding agent (and humans) implementing this project.
> This replaces the v1 requirements. Every rule here is meant to be implementable without guessing.
> Where v1 was ambiguous, this document makes a decision and marks it **[A#]** (assumption, change it if wrong). Unresolved items are **[Q#]** (see section 15).

**Keywords.** MUST / MUST NOT = non-negotiable. SHOULD = default unless there is a reason. MAY = optional.
**Priority tags.** **P0** = needed for the demo. **P1** = needed for the full pitch. **P2** = stretch.

---

## 1. Goal

Hackathon brief: *"Finding patterns in data is easy; knowing which ones to trust is hard. Build an AI researcher that tests its own ideas and can tell a real discovery from a lucky one."*

The system takes a user's "why" question about NBA games or betting (e.g. "Why do teams on the second night of a back-to-back underperform?"). It then:

1. Turns the question into a **measurable claim** and checks the pattern exists at all (premise check).
2. Generates 3-5 competing **hypotheses** (including a "chance" one) with an LLM.
3. Tests each with **hard-coded, deterministic statistical tools** (the LLM never computes statistics).
4. Runs a **deterministic stats gate** and an **LLM critic** on every result and every LLM decision.
5. Shows all of this as a **live graph** a human can watch, inspect and intervene in.
6. Outputs a **verdict per hypothesis** and an **evidence trail** a judge can follow.

### Non-goals
- No free-form code execution by the agent. No player-level or play-by-play questions (return *Untestable*). No live betting advice. No private data.
- This is a research tool, not betting advice (show a disclaimer in the UI).

---

## 2. Glossary (use these words exactly)

| Term | Definition |
|---|---|
| **Run** | One execution for one user question, with one mode and one budget. |
| **Claim** | The user's "why" question restated as a measurable statement: "*group A has a different value of metric M than group B*" (direction declared before testing). |
| **Premise** | The claim that the pattern exists at all, before any explanation is sought. |
| **Hypothesis** | A proposed *explanation* of a premise-passing pattern. Has a machine-readable `HypothesisSpec` (section 6). |
| **Primary test** | The single test per hypothesis that decides its verdict. Declared when the hypothesis is created and frozen before the holdout is used. |
| **Secondary test / mechanism prediction** | A further test of something the explanation implies (e.g. "if fatigue, the effect is larger after long travel"). Only affects the *explanation status*, never the verdict label. |
| **Ledger** | Global append-only list of every confirmatory p-value produced in the run (section 7.3). |
| **Gate** | A deterministic pass/fail rule (G1-G7) on a test result. No LLM involved. |
| **Critic** | An LLM reviewer that audits an LLM decision against a fixed checklist. It cannot override gates. |
| **Decision** | One LLM output of a defined type (D1-D6, section 8.2). Every decision becomes a graph node and triggers the critic. |
| **Research graph** | The *product artefact*: nodes and edges the user sees (section 5). Not the same thing as the LangGraph graph. |
| **LangGraph graph** | The *control flow* of the agent (section 4.2). Implementation detail. |
| **Slice** | A filtered subset of the data, defined by a `SliceSpec`. |
| **Seasons** | *explore* = 2007-08 to 2015-16. *validate* = 2016-17 to 2018-19. *special* = 2019-20 and 2020-21 (bubble, limited fans). *holdout* = 2021-22 onward (sealed). **[A1]** Exact boundaries depend on odds coverage **[Q4]**. |

---

## 3. Fixed decisions

| Area | Decision |
|---|---|
| Domain | NBA, regular season + playoffs, 2007-08 onward |
| Data | Game results + closing spread, total and moneyline |
| Team / time | 2-3 people, about 24h |
| LLM | Gemini free tier (Flash). Model id lives in config, never hard-coded. |
| Agent runtime | **LangGraph** (control flow, checkpointing, human-in-the-loop). **LangSmith** for tracing and evals (development only, section 12.3). |
| Backend | Python, FastAPI, WebSocket, SQLite |
| Frontend | React + React Flow. Wireframes deferred (section 11) but the data contract is not. |
| Agent freedom | Hard-coded tools only |
| Human role | Watch and intervene (Live mode), or approve each step (Approval mode). Switchable mid-run. |

---

## 4. Architecture

### 4.1 Components

| Component | Responsibility | Must NOT |
|---|---|---|
| **Data layer** | Builds the `games` and `team_games` tables, enforces season splits | Leak holdout rows to anything except `holdout_confirm` |
| **Tool library** | Pure, seeded Python functions (section 7) | Call an LLM |
| **Stats gate + ledger + verdict function** | Applies G1-G7, computes verdicts deterministically | Call an LLM |
| **Orchestrator** | LangGraph graph that makes LLM decisions and calls tools | Compute statistics or assign verdict labels |
| **Critic** | Audits each decision | Upgrade anything that failed a gate |
| **LLM gateway** | Rate limiting, caching, retries, model config | Be bypassed: all LLM calls go through it |
| **Event store** | Append-only log, the single source of truth | Be mutated (append only) |
| **Graph reducer** | `reduce(events) -> research graph state` (pure function) | Hold state not derivable from events |
| **API** | REST + WebSocket | Contain business logic |
| **Frontend** | Renders the reduced graph, sends interventions | Compute verdicts |

### 4.2 Three different "graphs": do not confuse them

1. **Research graph** (product). Nodes: Question, Premise Check, Hypothesis, Test Plan, Test Result, Mechanism Prediction, Verdict, Human Note. Stored as *derived state* of the event log. Rendered by React Flow.
2. **LangGraph graph** (control flow). The orchestrator's state machine (section 8.1). Its state holds **only pointers and counters** (ids, cursor, budget, mode), never the research graph itself.
3. **React Flow graph** (UI). A rendering of (1).

### 4.3 Source of truth

- The **event log** (SQLite table `events`) is the source of truth **[A2]**. The research graph is `reduce(events)`.
- The LangGraph **checkpointer** (SQLite) exists only so a run can pause (`interrupt`) and resume. It is not read by the API or UI.
- LangSmith traces are a development aid. The product MUST work with tracing off.

### 4.4 Async checks, concretely

After every decision the orchestrator spawns `asyncio` background tasks: `run_gate(node_id)` (if the node is a Test Result) and `run_critic(node_id)`. They append `check_result` events when finished. The orchestrator **does not await them in Live mode**. At the start of each orchestrator step a `collect_checks` LangGraph node drains new check events into state and applies section 9 rules (flagged, failed, taint). In Approval mode a `await_checks_and_human` node awaits them and then calls `interrupt()` for approval.

---

## 5. Research graph model

### 5.1 Node types and allowed parents

| Node | Created by | Parent type(s) |
|---|---|---|
| Question | system | none |
| Premise Check | D1 + tool result | Question |
| Hypothesis | D2 | Premise Check (one hypothesis may also derive from another via `revises`) |
| Test Plan | D3 | Hypothesis |
| Test Result | tool executor (not an LLM) | Test Plan |
| Interpretation | D4 | Test Result |
| Mechanism Prediction | D5 | Interpretation |
| Verdict | `compute_verdict()` + D6 narrative | Interpretation (+ Mechanism Prediction if any) |
| Human Note | human | any node |

**[A3]** v1 had no Interpretation node although FR3.2 lists "interpret result" as an LLM decision. It is added here so that every decision has a node.

### 5.2 Node fields (provenance)

```python
class Node(BaseModel):
    id: str                       # uuid
    run_id: str
    type: NodeType
    version: int = 1              # incremented when revised; old versions kept
    created_at: datetime
    lifecycle: Literal["proposed", "running", "checking", "complete"]
    check_outcome: Literal["pending", "passed", "flagged", "failed"]
    flags: set[Literal["tainted", "pruned", "awaiting_approval", "overridden", "stale"]]
    title: str                    # <= 80 chars, shown on the node
    llm: LLMRecord | None         # model_id, prompt, raw_response, parsed_output, rationale, from_cache
    tool: ToolRecord | None       # tool_name, args, args_hash, output, data_slice_hash, seed, duration_ms
    checks: list[CheckRecord]     # gate results and critic reviews
    human: list[HumanAnnotation]  # notes, approvals, overrides (with required note)
```

**[A4]** v1 mixed three different things into one "state" list. They are separated here:
- `lifecycle`: where the node is in its work.
- `check_outcome`: what the checks said.
- `flags`: independent booleans.

**UI colour precedence** (first match wins): `pruned` (grey, dashed) > `tainted` (hatched) > `awaiting_approval` (blue pulse) > `check_outcome=failed` (red) > `flagged` (amber) > `overridden` (purple outline) > `lifecycle=running|checking` (animated) > `passed` (green) > `proposed` (white).

### 5.3 Edge types

| Type | Meaning | From -> To |
|---|---|---|
| `derives_from` | Created because of | child -> parent |
| `tests` | Plan/result examines a hypothesis | Test Plan/Result -> Hypothesis |
| `supports` / `refutes` | Evidence direction, set from the gate outcome (not by the LLM) | Test Result -> Hypothesis |
| `revises` | New version of a node after critique or re-run | new -> old |

### 5.4 Taint rule

When a node's `check_outcome` becomes `failed`, every descendant (following `derives_from`, `tests`, `supports`, `refutes` forward) gets the `tainted` flag. Tainted nodes MUST be excluded from the report. Taint is cleared only if the failed ancestor is overridden by a human, or is replaced by a revised version that passes, and descendants are then re-run (`stale` until they are).

---

## 6. Data and types

### 6.1 Grain (important: v1 left this implicit)

Most effects are about *one team* in a game ("the team on a back-to-back"), but odds are per game. So there are two tables:

- `games`: one row per game (columns below).
- `team_games`: **view/table, two rows per game**, one per team, from that team's perspective.

```
games(game_id, date, season, home, away, home_pts, away_pts,
      close_spread_home, close_total, ml_home, ml_away,
      is_playoffs, is_bubble, is_neutral)

team_games(game_id, date, season, team, opp, is_home,
           pts, opp_pts, margin,                 # margin = pts - opp_pts
           spread,                               # spread from this team's perspective (negative = favourite)
           ats_margin,                           # margin + spread
           ats_cover,                            # 1 if ats_margin > 0, 0 if < 0, NULL if == 0 (push)
           win,
           rest_days, b2b, opp_rest_days, opp_b2b,
           travel_km, tz_shift, altitude_m,      # as of tip-off, this team's perspective
           games_into_season, rolling_ats_10, rolling_margin_10,   # as of tip-off, excluding this game
           close_total, total_over)              # total_over: 1 if pts+opp_pts > total, NULL on push
```

Canonical definitions (one place, with unit tests):
- `margin = home_pts - away_pts` for the home row, the negative for the away row.
- `close_spread_home`: negative means the home team is the favourite. For the away row, `spread = -close_spread_home`.
- `ats_margin = margin + spread`. `ats_cover = ats_margin > 0`. Pushes are `NULL` and dropped from rate calculations (never counted as 0).

### 6.2 Features
All features MUST be computed **as of tip-off** (no information from the game itself, no later games). `rest_days` at the start of a season is capped at 7 **[A5]**. Travel distance needs an arena coordinates table (lat/lon, tz, altitude) that is part of data prep.

### 6.3 Splits
Splits are by `season` as in the glossary. The data layer exposes `get_frame(split)` where `split in {"explore","validate","special"}` for general use. `holdout` rows are only readable inside `holdout_confirm`. A test asserts that no other code path can return them.

### 6.4 Data prep acceptance (P0, build first)
- Join rate of games to odds >= 98% for covered seasons, asserted in a test, with unmatched games listed to a file.
- Team relocations/renames mapped (Seattle->OKC 2008, NJ->Brooklyn 2012, New Orleans Hornets->Pelicans 2013, Bobcats->Hornets 2014).
- Unit tests: ats/push sign conventions on 5 hand-checked real games; no look-ahead in rolling features; `team_games` has exactly 2x rows of `games`.
- **Known data risk:** the odds archive is typically scraped with one row per team per game and quirky formats (pick'em lines, spread/total column order, abbreviations). Budget real time for it. If data is already prepared, say so **[Q1]**.

### 6.5 Slices and hypotheses (machine-readable)

```python
class Filter(BaseModel):
    col: Literal["season","team","opp","is_home","rest_days","b2b","opp_b2b","travel_km",
                 "tz_shift","altitude_m","month","is_bubble","is_playoffs","games_into_season"]
    op: Literal["eq","ne","in","gte","lte","between"]
    value: Any                         # validated against col type

class SliceSpec(BaseModel):
    split: Literal["explore","validate","special"] = "explore"   # holdout is never allowed here
    filters: list[Filter]

class HypothesisSpec(BaseModel):
    id: str
    statement: str                     # plain English
    kind: Literal["premise","explanation","confounder","chance","market"]
    metric: Literal["ats_margin","ats_cover","margin","win","total_over","roi"]
    group_a: SliceSpec
    group_b: SliceSpec | Literal["complement"]   # complement = same split, rows not in A
    direction: Literal["a_gt_b","a_lt_b","two_sided"]   # declared BEFORE testing
    primary_test: Literal["compare_groups","regress","persistence","market_test"]
    controls: list[str] = []           # feature names from the whitelist
    mechanism_prediction: MechanismSpec | None = None
```

`chance` hypotheses have no test of their own. **[A6]** A chance hypothesis is resolved by rule: *supported* if the premise's or any competing hypothesis's primary test fails G2-G5, *refuted* if all substantive hypotheses pass G2-G5. `confounder` hypotheses are tested with `regress` and a `controls` list; their claim is "the effect shrinks by at least 50% once controls are added".

"Underperform" is ambiguous. The orchestrator MUST pick a metric explicitly. **Default: `ats_margin`** (more statistical power than binary `ats_cover`, and measures performance against expectation). It MUST say which one it chose and why in the Premise Check rationale.

---

## 7. Tool library

All tools: pure functions, Pydantic input/output models, deterministic given `(args, data version, seed)`, return a `ToolResult` (never raise to the agent; return `status="error"` with a message). Every call is logged with `args_hash` and `data_slice_hash`. Tools return **summaries** (numbers, CIs, small arrays for charts), never raw dataframes, so the LLM context stays small.

### 7.1 Data and feature tools

| Tool | Input | Output |
|---|---|---|
| `describe_dataset()` | none | seasons, row counts, whitelisted columns with types, available metrics, splits (holdout row counts hidden) |
| `define_slice(spec: SliceSpec)` | SliceSpec | `slice_id` (hash of spec), `n_rows`, `n_games`, `n_teams` |
| `add_feature(name, params)` | `name` from fixed menu: `rest_days`, `b2b`, `travel_km`, `tz_shift`, `altitude_m`, `rolling_ats(k)`, `rolling_margin(k)`, `games_into_season` | confirms the feature exists and its coverage. **[A7]** Features are precomputed in the data layer; this tool only whitelists and parameterises them. |

### 7.2 Test tools

| Tool | Input | Output / behaviour |
|---|---|---|
| `compare_groups(spec: HypothesisSpec)` | metric, group_a, group_b, direction | `effect` (see below), `ci_low/high` (bootstrap, 95%), `p_perm`, `n_a`, `n_b`, `mde`, chart arrays (null distribution histogram, observed value) |
| `regress(spec)` | metric, features, controls | coefficient + cluster-robust CI and p for the feature of interest; OLS for continuous, logit for binary. **Always includes `spread` as a control** when the metric is `ats_*`/`win`/`roi` and the claim is a market claim, and offers it otherwise. |
| `out_of_sample(test_id)` | id of a prior test | re-runs the identical spec with `split` swapped to `validate`; returns effect, CI, and `same_sign`, `retention = effect_val / effect_explore` |
| `persistence(metric, entity)` | e.g. `ats_cover`, `team` | correlation of team-season value in season t vs t+1 (regression-to-the-mean test), with permutation p |
| `market_test(slice, side)` | slice + `side in {team, opp, over, under}` | cover or over rate with Wilson/bootstrap CI vs break-even 0.5238 (110/210), flat-stake ROI at -110 (or actual moneyline for ml bets), p vs break-even |
| `power(n_a, n_b, metric)` | sample sizes | minimum detectable effect at 80% power, alpha 0.05 |
| `placebo(test_id)` | id of a prior test + a declared placebo slice | same test where no effect should exist (e.g. bubble test on non-bubble months); returns effect and CI |
| `holdout_confirm(hypothesis_id)` | hypothesis id | runs that hypothesis's **frozen primary test spec** on the holdout seasons. **Once per hypothesis**; a second call returns `status="error", reason="already_used"`. Always logged. |

### 7.3 Statistical definitions (v1 left these open)

- **Effect size.** Binary metrics: difference in rates, in percentage points. Continuous: difference in means, in the metric's units. `effect = mean(A) - mean(B)`.
- **Unit of resampling.** **Game level**, blocked by season: bootstrap and permutation resample/permute whole games, never single team-rows, so the two rows of one game never separate. **[A8]** For regressions, cluster-robust SEs by `game_id`, and additionally by `team-season` where team effects are present.
- **Permutation p.** 10,000 permutations of group labels *within season blocks*, two-sided unless `direction` is one-sided and was declared before testing. Seed fixed.
- **Bootstrap CI.** 5,000 resamples (game-level, season-blocked), percentile CI.
- **Seed.** One global `SEED = 1729`, combined with `args_hash` per call so different tests do not share random streams, same args always give same output.
- **Ledger entry rule.** Every p-value from `compare_groups`, `regress`, `persistence` and `market_test` is a **ledger entry**, keyed by `spec_hash`. Re-running an *identical* spec does not add an entry. Changing *anything* in a spec adds one (this counts the forking paths). `out_of_sample`, `placebo`, `power`, `holdout_confirm` are diagnostics and are logged but are not ledger entries.

---

## 8. Orchestrator (LangGraph)

### 8.1 Control flow

```
intake -> premise_check -> [premise false/untestable? -> report]
       -> generate_hypotheses -> (for each hypothesis, in order):
            plan_test -> execute_test -> collect_checks -> interpret
              -> (if G1-G5 pass) out_of_sample -> collect_checks
              -> (if G1-G5 pass) market_test -> collect_checks
              -> (if G1-G6 pass) holdout_confirm
              -> (optional) mechanism_prediction -> secondary test
              -> compute_verdict + narrative
       -> write_report
```

Every `->` into an LLM node is followed by `collect_checks`. Conditional edges are decided by **gate results in state**, never by LLM opinion. Use `interrupt()` before executing a decision's effects when `mode == "approval"`. Checkpointer: SQLite, so `mode` can be flipped between steps **[A9]**.

LangGraph state (pointers only):
```python
class AgentState(TypedDict):
    run_id: str
    mode: Literal["live", "approval"]
    question: str
    hypothesis_queue: list[str]          # hypothesis ids
    current: str | None
    budget: Budget                       # remaining llm_calls, tests, depth
    pending_checks: list[str]            # node ids with unfinished checks
    revision_counts: dict[str, int]
```

### 8.2 LLM decisions (each = structured output + `rationale` + critic review)

| ID | Decision | Output schema (summary) |
|---|---|---|
| D1 | Restate question as claim | `Claim{metric, group_a, group_b, direction, rationale}` or `Untestable{missing_data}` |
| D2 | Generate hypotheses | 3-5 `HypothesisSpec`, **must include one `chance` and at least one `confounder`** |
| D3 | Plan test | `TestPlan{tool, args}` where `args` validate against the tool's Pydantic input. Invalid -> one automatic retry, then node `failed`. |
| D4 | Interpret result | `Interpretation{summary, claims_made[], limitations[]}`, cites gate results by id |
| D5 | Mechanism prediction | `MechanismSpec{statement, secondary_test: HypothesisSpec}` or `None` |
| D6 | Verdict narrative and report text | prose only. **The label is not an LLM output.** |

The LLM MUST NOT choose the verdict label, and MUST NOT see holdout numbers except as returned by `holdout_confirm`.

### 8.3 Premise check (v1's FR1.2, made precise)
1. D1 produces a `Claim` (a `HypothesisSpec` with `kind="premise"`).
2. `compare_groups` runs on **explore** seasons only.
3. If G1 fails -> run ends, verdict **Inconclusive**. If G2 or G3 fails -> run ends, verdict **Premise false** (the report says "the pattern is not in the data" and does not explain noise). If all pass -> continue.
4. If D1 returns `Untestable`, run ends with that verdict and `missing_data` filled. A proxy is allowed only if the node title says "PROXY:" and the report states what it stands in for.

### 8.4 Reacting to checks (v1's FR3.3, made precise)
At each `collect_checks`:
- `check_outcome=flagged` (critic `concerns`): the orchestrator MUST either **revise** (new node version via `revises`) or **justify** (a new D4-type decision that answers each concern; the critic reviews it once). Maximum `max_revisions_per_node = 2` **[A10]**; after that the node is set to `failed` unless a human overrides.
- `check_outcome=failed` (gate failure, or critic `reject`): apply taint (5.4). The orchestrator MUST NOT build new nodes on it.
- Building on a node with `check_outcome=pending` is allowed in Live mode only.

### 8.5 Budget and stopping
Defaults: `max_llm_calls=60`, `max_tests=40` (ledger entries), `max_depth=8`, all shown in the UI. Stop when every hypothesis has a verdict, or any budget hits 0. **On budget exhaustion, unfinished hypotheses get verdict *Inconclusive (budget)*** **[A11]**, then the report is written (the report step reserves 2 LLM calls).

---

## 9. Verification layer

### 9.1 Stats gate (deterministic)

Parameters (all in one `config.py`, **[A12]** defaults are starting points to tune): `ALPHA=0.05`, `BH_Q=0.05`, `N_MIN=200` per group, `RETENTION=0.5`, `BREAKEVEN=0.5238`, `SESOI` (smallest effect of interest): `ats_margin/margin: 1.0 pts`, all binary/ROI metrics: `0.03`.

| Gate | Applies to | Rule |
|---|---|---|
| **G1 Power** | every primary test | `n_a >= N_MIN and n_b >= N_MIN` **and** `MDE(80% power) <= SESOI`. Otherwise **Inconclusive**. *(v1 compared the observed effect to the MDE; that is circular, because it is nearly the same as significance.)* |
| **G2 Uncertainty** | primary test | 95% CI excludes 0 and the sign matches the declared `direction` |
| **G3 Null** | primary test | `p_perm < ALPHA` |
| **G4 Multiple testing** | primary test | The test's p-value is rejected by Benjamini-Hochberg at `BH_Q` over **the whole ledger at the time of evaluation**. Because BH on a growing ledger can flip earlier results, G4 is evaluated *provisionally* on arrival and **re-evaluated for every hypothesis when the report is built**; the final value is the one that counts, and any flip is logged as a `check_result` event. |
| **G5 Replication** | primary test | `out_of_sample` has the same sign and `retention >= RETENTION`. (The validation CI is reported but not required to exclude 0, as validation is smaller.) |
| **G6 Market** | any hypothesis that passes G1-G5 | `market_test` on the hypothesis's slice: lower CI bound of cover/over rate `> BREAKEVEN`, **and** the effect survives with `spread` as a control in `regress`. Market tests are ledger entries. |
| **G7 Holdout** | hypotheses that pass G1-G6 | `holdout_confirm`: same sign, `retention >= RETENTION`, and the same G6 rule on the holdout. Run once. **G7 cannot be overridden.** |

### 9.2 Verdict function (v1's FR6, now a deterministic table)

`compute_verdict(hypothesis) -> Verdict` is a pure function over gate results. First matching row wins:

| # | Condition | Verdict |
|---|---|---|
| 1 | Hypothesis flagged *Untestable* by D1/D3 and no whitelisted tool can express it | **Untestable** |
| 2 | G1 fails | **Inconclusive** |
| 3 | Any of G2-G5 fails | **Luck** |
| 4 | G1-G5 pass, G6 fails or not applicable | **Real but priced in** |
| 5 | G1-G6 pass, G7 fails | **Real but priced in**, with flag `holdout_failed` **[A13]** |
| 6 | G1-G7 pass | **Real and beats market (provisional)** |
| - | Premise stage: G2 or G3 fails | **Premise false** |

Each verdict also carries `explanation_status`: **Mechanism supported** (a secondary prediction was tested and passed its own G2-G3), **Mechanism tested and failed**, or **Mechanism plausible but untested**. It never changes the verdict label.

### 9.3 LLM critic

Runs on every decision (D1-D6). It gets: the decision output, its `rationale`, the tool output and gate results for the node, and a fixed checklist. It does **not** get the orchestrator's system prompt or hidden reasoning. It uses a different system prompt.

```python
class CriticReview(BaseModel):
    verdict: Literal["approve", "concerns", "reject"]
    checklist: dict[str, Literal["ok", "problem"]]   # keys below
    severity: Literal["low", "medium", "high"]
    evidence: list[str]            # MUST cite specific numbers/ids from the node; empty evidence => review downgraded to "approve with low severity"
    rationale: str
    suggested_fix: str | None
```
Checklist keys: `construct_validity`, `missing_confounder`, `selection_or_method_bias`, `overclaim_given_gates`, `testable_mechanism`.

Mapping: `approve` -> `check_outcome=passed`; `concerns` -> `flagged`; `reject` -> `failed` (taint).

### 9.4 Authority rules
1. Gate results are final. The critic can set `flagged` or `failed` (downgrade), but can never change a failed gate to passed, and cannot change a verdict label.
2. Only a human can override, via an intervention with a mandatory note (min 10 chars). An override sets the `overridden` flag; the node and the report show who overrode what and why. G7 is not overridable.
3. In the report, any verdict with an overridden check is labelled `(human-overridden)`.

### 9.5 Mode semantics
- **Live:** decisions apply immediately; checks arrive later; section 8.4 and taint handle late failures.
- **Approval:** after each decision the graph awaits its checks, sets `awaiting_approval`, calls `interrupt()`, and resumes on the `approve` or `reject` intervention. `reject` -> node `failed` with the human's note, orchestrator must revise.
- Mode can be toggled at any time. The change takes effect at the next decision boundary.

---

## 10. Events and API

### 10.1 Event envelope (the contract between backend and frontend: define in hour 0-2)

```python
class Event(BaseModel):
    seq: int                       # monotonic per run
    run_id: str
    ts: datetime
    actor: Literal["orchestrator","tool","gate","critic","human","system"]
    type: Literal["run_created","run_status","node_created","node_updated","edge_created",
                  "check_result","ledger_entry","verdict_computed","intervention",
                  "budget_update","mode_changed","error"]
    node_id: str | None
    payload: dict                  # typed per `type` in shared/schemas
```
Reducing the same events twice MUST give an identical graph. Replay mode feeds a recorded event file through the same WebSocket path with a speed control and **no LLM calls**.

### 10.2 Endpoints

| Endpoint | Behaviour |
|---|---|
| `POST /runs` `{question, mode, budget?}` -> `{run_id}` | Starts a run |
| `WS /runs/{id}/events?from_seq=N` | Streams events; on connect sends all events from `from_seq`, then live |
| `GET /runs/{id}/events` | Full log (for export/replay) |
| `POST /runs/{id}/interventions` `{type, node_id, payload}` | See below |
| `GET /runs/{id}/report` | Report JSON (and `?format=md` for Markdown) |
| `POST /replay` `{log_file, speed}` | Replay a recorded run |

### 10.3 Interventions

| Type | Payload | Effect |
|---|---|---|
| `prune` | `{note?}` | Sets `pruned` on the node and all descendants; excluded from report and budget |
| `rerun` | `{}` | Re-executes the node (cache bypassed for LLM nodes), creates a new version with `revises`; descendants become `stale` |
| `override` | `{check_id, note}` (note required) | Section 9.4 |
| `approve` / `reject` | `{note?}` (note required on reject) | Approval mode only |
| `set_mode` | `{mode}` | Section 9.5 |
| `add_hypothesis` (P2) | `{statement}` | Orchestrator converts to a `HypothesisSpec` (D2-style) and it goes through the normal pipeline |

---

## 11. Frontend contract (wireframes deferred)

The wireframes can wait, but a frontend built against a mocked event stream needs this much fixed now:

- **Layout (P0):** top bar | graph canvas (centre) | inspector (right) | report view (separate route).
- **Top bar:** mode toggle, LLM budget meter, "tests spent" (ledger size), run status, question text.
- **Graph:** auto-layout (dagre). Node shows icon by type, `title`, a state colour (section 5.2 precedence), and a verdict badge. Edges styled by type.
- **Inspector tabs:** Reasoning, Tools, Evidence, Checks, Notes. Evidence charts (P0 minimum): effect-size CI bar, null distribution with observed line, explore vs validate vs holdout bars.
- **Report view:** answer to the question, verdict table, per-verdict clickable path through the graph, Markdown export.
- **Replay:** a speed control and play/pause when a recorded log is loaded.
- **Rule:** the frontend renders `reduce(events)` and posts interventions. It MUST NOT compute verdicts, taint or colours from raw stats.
- Visual design (look and feel) is **TBD [Q5]**. Do not block backend work on it. A fixture event log is delivered at the end of milestone M0 so the frontend can start immediately.

---

## 12. Cross-cutting requirements

### 12.1 LLM gateway
- One gateway for every LLM call (orchestrator and critic). It wraps the Gemini chat model via LangChain (structured output via the model's `with_structured_output`) and provides: token-bucket rate limiting at the project's real RPM **[Q3]** (LangChain core has an in-memory rate limiter you can pass to the model; verify against the version you pin), exponential backoff on 429, an on-disk cache keyed by `hash(model_id, system, prompt, schema)`, and `call counter -> budget_update` events.
- `LLM_MODE = live | fake | cached-only`. **`fake` returns scripted, schema-valid outputs so the whole pipeline (including UI) can be developed with zero quota.** This is a P0 requirement.
- Priority (P1): in Live mode the orchestrator first; in Approval mode critics first. If this costs time, use a single FIFO queue.
- Each teammate uses their own API key in dev. Model id is config.

### 12.2 Reproducibility and auditability
- Tools are pure and seeded: re-executing a node's stored `args` gives the same output (tested).
- The event log is append-only. Every node keeps its provenance (5.2).

### 12.3 LangSmith (development only)
- Tag every LLM call with `run_id`, `node_id`, `decision_type` as metadata so traces can be filtered per run.
- Create datasets: (a) **known-answer questions** (section 13.3), (b) **critic test cases** (hand-written decisions with planted flaws: wrong metric, missing confounder, overclaim vs gates; the critic should flag them), (c) **D3 argument validity** (does the planned tool call validate against the schema).
- Use LangSmith evaluators to track: schema-valid-output rate, critic recall on planted flaws, verdict agreement with expected on known-answer questions.
- Optional (P2): store the LangSmith trace URL on the node and link it from the inspector.
- The product MUST run with tracing disabled.

---

## 13. Testing and evaluation

### 13.1 Unit tests (P0)
Sign conventions and pushes; no look-ahead; join rate; `team_games` = 2x `games`; tool determinism; holdout inaccessible except via `holdout_confirm`; `holdout_confirm` second call fails; `compute_verdict` table (one test per row); BH ledger behaviour (G4 re-evaluation flips correctly); reducer determinism.

### 13.2 Planted-noise FDR test (P0, no LLM, build before the orchestrator)
Generate 100 random "coin-flip" features (random group labels at team-season level and at game level), run each through the full pipeline (primary test -> G1-G6 -> verdict). The share reaching "Real" (rows 4 or 6) SHOULD be <= about 5%. Report the number. This is the headline evidence the gate works. Also run a **planted-signal** test (inject a known +3pp effect) and check it is detected.

### 13.3 Known-answer questions (end to end)
| Question | Expected | Note |
|---|---|---|
| Why did home-court advantage disappear in the 2020 bubble? | Real (existing research) | Small-n: could legitimately be Inconclusive; the evidence chain must explain |
| Why did a team with a high ATS record in season t cover so often? | Luck (fails `persistence`) | |
| Why do teams on the 2nd night of a back-to-back underperform? | Real but priced in | Headline demo **[Q2]** |

If our data disagrees with an expectation, that is acceptable provided the evidence chain explains why.

### 13.4 Behaviour tests
- **Taint:** force a critic `reject` on a mid-graph node; its descendants become `tainted`; the orchestrator revises.
- **Approval:** toggle mid-run; the next decision blocks until approved.
- **Replay:** a recorded log replays to an identical final graph and report.

---

## 14. Build order and milestones

Build **vertical slices with the LLM faked first**, so the hard-to-replace parts (data, stats, gate) are right before spending quota.

| Milestone | Content | Done when |
|---|---|---|
| **M0 Contracts** (hours 0-2, all together) | `shared/schemas` (Pydantic: Node, Event, HypothesisSpec, tool I/O, CriticReview) exported as JSON Schema + TS types; config constants; **a fixture event log of one full fake run** | Frontend can render the fixture; backend can validate against the schemas |
| **M1 Data** | `games`, `team_games`, splits, unit tests | 6.4 acceptance passes |
| **M2 Stats core** | tools 7.1-7.2 (except holdout/placebo), gate G1-G6, ledger, `compute_verdict`, planted-noise test | 13.1 + 13.2 pass; **no LLM yet** |
| **M3 Pipeline with fake LLM** | LangGraph graph, event store, reducer, FastAPI + WS, gateway in `fake` mode | A full run for the B2B question completes using scripted decisions; UI shows it live |
| **M4 Real LLM** | Gemini decisions D1-D6, schemas, retries, cache | Same question runs live on Gemini within budget |
| **M5 Critic and harness (P1)** | critic, taint, approval mode, interventions, `holdout_confirm` + G7, `placebo` | 13.4 tests pass |
| **M6 Demo** | known-answer runs, 2 recorded successful logs, replay mode, report export, pitch | Rehearsal on replay only |

### Priority tiers
- **P0:** M0-M4, report view, replay, FDR test, critic in simple form (approve/concerns/reject recorded on nodes).
- **P1:** taint, approval mode, interventions (prune, rerun, override), holdout and G7, placebo, priority queue.
- **P2:** human-added hypotheses, extra charts, side-by-side run comparison, a critic from a different model family, play-by-play data, LangSmith trace links.

### Repo layout
```
shared/schemas/        # Pydantic models -> JSON Schema -> TS types
backend/
  data/                # prep scripts, arena table, odds join
  stats/               # tools.py gates.py ledger.py verdict.py config.py
  agent/               # graph.py (LangGraph) decisions.py critic.py gateway.py prompts/
  store/               # events.py reducer.py
  api/                 # main.py
  tests/
frontend/              # Vite + React + React Flow
recordings/            # event logs for replay
evals/                 # LangSmith datasets and evaluators
```

---

## 15. Open questions (answer early; none block M0-M2)

- **[Q1]** Is the data already downloaded/joined, or is data prep part of the clock? (Determines M1 length.)
- **[Q2]** Is the headline demo the B2B question?
- **[Q3]** Real Gemini RPM/RPD for the demo project (AI Studio rate-limit page)? This sets the per-run budget.
- **[Q4]** Where does odds coverage end? This fixes the holdout seasons (if only 1-2 seasons remain, G7 has little power; consider treating holdout as "confirmation, not proof").
- **[Q5]** What should the UI look and feel like? (Wireframes after M3.)
- **[Q6]** Judging criteria weighting (technical depth vs demo vs impact)? Decides time on the FDR proof versus polish.

## 16. Risks carried over from v1 (kept, condensed)
Premise-assumes-pattern (mitigated by 8.3); forking paths (ledger, 7.3); holdout leakage (6.3, `holdout_confirm`); look-ahead bias (6.2); real != profitable (G6); LLM checking LLM (gate authority, 9.4); explainable != causal (mechanism rule, 9.2); async inconsistency (taint, 5.4); non-independence (game-level blocks, 7.3); small-n slices (G1, `placebo`); spread sign bugs (6.1); live-demo rate limits (gateway, replay); out-of-scope questions (*Untestable*); hidden overrides (9.4); responsible gambling (UI disclaimer); model availability (config, backup key).confirm the next decision blocks until it's approved.
- **Replay test:** a recorded log replays to an identical final graph and report.

## 9. Still open (decide early, not blocking)
- Hackathon rules: may we prepare the data or code beforehand?
- Judging criteria weighting (technical depth vs demo vs impact). This affects how much time goes into polish vs the FDR proof.
- Which question is the headline demo? (Recommended: the B2B question, because it shows all three verdict types in one graph.)
- The actual Gemini RPM/RPD for our project (from AI Studio), which sets the per-run budget.
- Exact boundaries of the holdout seasons once SBR coverage is confirmed.

## Next step after approval
Save this as `/Users/alexhossien/whack/REQUIREMENTS.md` so the team has it in the repo, then (if wanted) scaffold the repo: `backend/` (FastAPI, tools, gate, orchestrator), `frontend/` (Vite + React Flow), `data/` (prep scripts), along with the shared schemas from hour 0–2.
