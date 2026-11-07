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
- **Approval-mode test:** toggle mid-run and confirm the next decision blocks until it's approved.
- **Replay test:** a recorded log replays to an identical final graph and report.

## 9. Still open (decide early, not blocking)
- Hackathon rules: may we prepare the data or code beforehand?
- Judging criteria weighting (technical depth vs demo vs impact). This affects how much time goes into polish vs the FDR proof.
- Which question is the headline demo? (Recommended: the B2B question, because it shows all three verdict types in one graph.)
- The actual Gemini RPM/RPD for our project (from AI Studio), which sets the per-run budget.
- Exact boundaries of the holdout seasons once SBR coverage is confirmed.

## Next step after approval
Save this as `/Users/alexhossien/whack/REQUIREMENTS.md` so the team has it in the repo, then (if wanted) scaffold the repo: `backend/` (FastAPI, tools, gate, orchestrator), `frontend/` (Vite + React Flow), `data/` (prep scripts), along with the shared schemas from hour 0–2.
