# Explore, commit, confirm, judge: agent redesign

Status: agreed by A, B and C on 2026-11-08. This is the implementation spec.

The agent explores the data freely with our tools, commits to findings as frozen specs, and each
finding is then confirmed, without changes, on seasons the agent never saw. A judge reviews the
analysis trail. The gate still assigns every verdict label.

This replaces the fixed pipeline in `backend/agent/` (premise, then 3-5 hypotheses, then one
primary test each, then LLM-chosen follow-ups).

## 1. Why the split matters

When the agent tries many slices, the p-values on the data it searched are not evidence: with 20
tries, about one hits p < 0.05 by luck. So:

- **Exploration results are descriptive only.** They never go through the gate or the ledger.
- **Only confirmation on the validation seasons decides the verdict.** The agent's searching could
  not have touched that data, so its p-values mean what they say.
- **The spec is frozen at commit.** Nothing about a finding changes after the agent sees any
  validation number.

Seasons (unchanged, `backend/data/seasons.py`):

| Split | Seasons | Who reads it |
|---|---|---|
| explore | 2007-08 to 2015-16 | exploration and the discovery estimate |
| validate | 2016-17 to 2018-19 | confirmation only, after commit |
| special | 2019-20, 2020-21 | nobody by default |
| holdout | 2021-22 onward | `holdout_confirm` only, once per finding |

## 2. Run flow

```
intake
  -> explore  (loop: one LLM decision = one tool call, explore split only)
  -> commit   (LLM: 1-5 findings, each a frozen HypothesisSpec citing explore steps)
  -> judge    (LLM: approve, or send back to explore with feedback; max 2 send-backs)
  -> confirm  (no LLM: fixed test sequence per finding, gated)
  -> verdict  (gate label per finding; judge may downgrade)
  -> report   (LLM: answer from the verdict table)
```

### 2.1 Explore

Each turn the LLM returns one `ExploreStep`. Code runs the tool through `dispatch_explore` and
appends the result to the trail. The loop ends when the LLM chooses `commit`, or when
`MAX_EXPLORE_CALLS` is reached (then commit is forced).

Allowed tools: `describe_dataset`, `define_slice`, `add_feature`, `compare_groups`, `regress`,
`persistence`, `market_test`, `power`, `placebo`.
Not allowed: `out_of_sample`, `holdout_confirm`.

Any slice or split other than `explore` comes back as `status="error",
reason="explore_split_only"`. The LLM sees the error and adjusts. Nothing raises.

### 2.2 Commit

The LLM states 1-5 findings. Each is a full `HypothesisSpec` (metric, group_a, group_b, direction,
primary_test, controls) plus the explore step ids it rests on. Code:

1. Assigns ids `f1`, `f2`, ... (overwriting whatever the LLM wrote).
2. Forces every slice's `split` to `explore`.
3. Checks every cited step id exists. If not, the commit is rejected back to the LLM once, with
   the problem listed.

If the question presupposes an effect ("why do X cover less?"), the commit must include that
effect as a finding with `kind="premise"`. That gives the report a yes or no on the question
itself.

A run where nothing is worth committing returns zero findings. The report then says so.

### 2.3 Judge (before confirmation)

One LLM call that sees the question, the whole explore trail (compacted) and the commit. It
checks:

- Does each finding follow from the cited steps?
- Is the spec the one the evidence supports, or a cherry-picked variant (narrower slice, different
  metric) of something that looked weaker in general?
- Obvious confounders the agent saw but did not control for.
- Whether the premise finding is missing.

Output: `approve`, or `revise` with feedback. A `revise` must cite step ids; one with none is
discarded (as the critic does today). On `revise`, the run returns to explore with the feedback
in context and the trail kept. After 2 send-backs the judge's objections are attached as flags
and the run proceeds.

The send-back is only allowed here, before any validation number exists. After confirmation the
judge can only downgrade (2.5).

### 2.4 Confirm (deterministic, per finding)

No LLM. For each finding, in order, stopping at the first failed stop-gate:

| Step | Call | Data | Gates | Stops on fail |
|---|---|---|---|---|
| 1. discovery | `<primary_test>(spec)` | explore | none (the estimate G5 compares against) | tool error only |
| 2. confirmation | `out_of_sample(test_id from step 1)` | validate | G1-G5 | yes |
| 3. placebo | `placebo(test_id, placebo_slice)` if the commit gave one | explore | none (shown to the judge) | no |
| 4. market | `market_test(slice=group_a on validate, side)` | validate | G6 | no; it only blocks step 5 |
| 5. holdout | `holdout_confirm(finding id)` if G1-G6 all pass | holdout | G7 | n/a |

- `side` for step 4 is `opp` when `direction == "a_lt_b"`, else `team` (as in today's
  `_follow_up_args`).
- The spec passed to the gate for step 4 has every slice's split set to `validate`, so G6's
  spread-control `regress` also runs on validation data.
- The ledger now holds only confirmation and market p-values, so BH (G4) corrects across the
  findings, not across everything explored.
- Each `out_of_sample` call is one ledger entry. Re-running an identical finding adds nothing.

### 2.5 Verdict

The label is `compute_verdict` over the finding's gates. The table is unchanged; the gates now
come from validation data:

| Gates | Label |
|---|---|
| G1 fails | Inconclusive |
| any of G2-G5 fails | Luck |
| G6 fails or never ran | Real but priced in |
| G7 fails or never ran | Real but priced in (`holdout_failed` when G7 failed) |
| G1-G7 pass | Real and beats market (provisional) |

A `kind="premise"` finding uses the premise rule instead: G1 fails gives Inconclusive, G2 or G3
fails gives Premise false, otherwise the premise holds and its row shows its label from the
table above.

Before the report, `gate.reevaluate()` re-runs BH over the final ledger and G4 flips are logged
and applied, as today.

**Judge after confirmation.** One more judge call sees every finding's results and labels. It may
downgrade any label to `Inconclusive` with a rationale citing node ids, for example when the
placebo showed the same effect. It can never upgrade, and it can never touch G7. Downgrades are
logged as `check_result` events with `checker="critic"`.

Chance and confounder hypotheses no longer exist as separate tracks. A confounder the judge
raises is the agent's to test during exploration, with `regress` and controls.

## 3. Budgets and constants

Agent constants (B), in `backend/agent/state.py`:

| Name | Value | Meaning |
|---|---|---|
| `MAX_EXPLORE_CALLS` | 30 | tool calls in explore, across all send-backs |
| `MAX_FINDINGS` | 5 | findings per commit |
| `MAX_JUDGE_SENDBACKS` | 2 | revise rounds before proceeding |
| `REPORT_RESERVE` | 2 | LLM calls kept for the report (unchanged) |

LLM calls per run, worst case: 30 explore + 3 commits + 3 judge + 1 final judge + 1 report = 38.
At `GEMINI_RPM=5` that is about 8 minutes. Confirmation adds no LLM calls.

Gate thresholds stay in `backend/gate/config.py` (A).

## 4. Schemas

### 4.1 LLM outputs (B, `backend/agent/schemas.py`)

```python
ExploreTool = Literal[
    "describe_dataset", "define_slice", "add_feature", "compare_groups", "regress",
    "persistence", "market_test", "power", "placebo",
]

class ExploreStep(BaseModel):
    thought: str              # why this call, in one or two sentences
    action: Literal["call_tool", "commit"]
    tool: ExploreTool | None = None
    args_json: str | None = None   # a JSON object matching the tool's schema
    builds_on: list[str] = []      # explore step ids this call follows from

class Finding(BaseModel):
    statement: str
    kind: Literal["premise", "explanation", "market"]
    spec: HypothesisSpec           # backend/tools/schemas.py; id and splits are overwritten
    evidence: list[str]            # explore step ids, at least one
    placebo_slice: SliceSpec | None = None

class Commit(BaseModel):
    rationale: str
    findings: list[Finding]        # 0 to MAX_FINDINGS

class JudgeReview(BaseModel):
    decision: Literal["approve", "revise"]
    rationale: str
    objections: list[JudgeObjection] = []

class JudgeObjection(BaseModel):
    finding_id: str | None         # None = about the exploration as a whole
    problem: str
    evidence: list[str]            # explore step or node ids; required for it to count

class JudgeDowngrade(BaseModel):
    finding_id: str
    rationale: str
    evidence: list[str]
```

`args_json` is a string on purpose: one schema covers every tool, and Gemini's structured output
handles it reliably. Code parses it with `json.loads`. A parse failure goes back to the LLM as a
tool error (`reason="invalid_json"`) and costs a call.

### 4.2 What the explore LLM sees each turn

```json
{
  "question": "...",
  "tools": [{"name": "...", "description": "...", "json_schema": {...}}],
  "trail": [
    {"step": "x3", "tool": "compare_groups", "args": {...}, "builds_on": ["x1"],
     "status": "ok", "n": 4120, "effect": -0.031, "ci": [-0.05, -0.012], "p_value": 0.004}
  ],
  "calls_left": 22,
  "judge_feedback": null
}
```

`tools` is `TOOL_REGISTRY` filtered to `ExploreTool`. Trail rows drop `extra` except `n_a`,
`n_b`, `test_id` (needed for `placebo`) and `reason`/`problems` on errors.

## 5. Graph and contract

One contract change, in `contracts/models.py` and the generated TypeScript together:

```python
class Node(Contract):
    ...
    phase: Literal["explore", "confirm"] | None = None
```

Node mapping (no new node types):

| Thing | Node type | Phase | Parents (edge type) |
|---|---|---|---|
| the question | `question` | none | none |
| one explore call | `test_result`, with `llm` = the step's decision and `tool` = the result | explore | `builds_on` steps, else the question (`derives_from`) |
| the commit's reasoning | `interpretation` | explore | the cited steps (`supports`) |
| one finding | `hypothesis` | confirm | the commit node (`derives_from`) |
| each confirm step | `test_result` | confirm | the finding (`tests`) |
| the finding's verdict | `verdict` | confirm | its last confirm result (`derives_from`) |

Judge reviews are `check_result` events with `checker="critic"`: the pre-confirm review on the
commit node, and downgrades on the verdict node. `Node.title` is still at most 80 characters.

## 6. Tasks by owner

### A (data and stats): `backend/tools/`, `backend/gate/`

- [ ] `registry.dispatch_explore(name, args) -> ToolResult`: allowlist from 2.1; reject any slice
      or `split` other than `explore` (in `spec.group_a`, `spec.group_b`, `slice`,
      `placebo_slice`, and `persistence.split`) with `reason="explore_split_only"`; overwrite
      `spec.id` with `x-<args_hash[:8]>` so explore calls never register a finding id.
- [ ] `StatsGate.evaluate(node_id, result, spec, role)`: take the role explicitly instead of
      inferring it. Roles: `discovery` (no gates, no ledger), `confirmation` (G1-G5, one ledger
      entry), `market` (G6), `holdout` (G7), `diagnostic` (none).
- [ ] G1-G4 on `out_of_sample` output: use `extra["original_tool"]` wherever the code now checks
      `result.tool` (`_power`'s n_a/n_b branch, `sesoi`, the market-test branch of
      `_uncertainty`). Check `out_of_sample` passes through `n_a`, `n_b` and `mde`.
- [ ] Premise rule as a function next to `compute_verdict` (`premise_verdict(gates)`), so B no
      longer encodes it.
- [ ] Remove `resolve_chance` from the verdict path once B drops chance tracks (keep it until
      then).
- [ ] Tests: `test_explore_rejects_validate_split`, `test_explore_rejects_holdout_tools`,
      `test_explore_overwrites_spec_id`, `test_confirmation_role_gets_g1_to_g5`,
      `test_discovery_role_adds_no_ledger_entry`, `test_ledger_counts_only_confirmations`.
- [ ] No-LLM runner (commit, then confirm, then verdict) for the planted-noise FDR and
      planted-signal checks, reusing B's confirm step as a plain function.

### B (agent): `backend/agent/`, `backend/llm/`

- [ ] Schemas in 4.1 and prompts for `explore`, `commit`, `judge`, `judge_final`, `report`.
- [ ] New LangGraph nodes in the order of section 2; drop `premise`, `hypothesize`, `plan_test`,
      `choose_test`, `interpret` and the chance logic. Keep checkpointing, pause and set-mode.
- [ ] `confirm(finding, tools, gate) -> list[node ids]` as a plain function with no LLM, so A's
      no-LLM runner can call it.
- [ ] Explore tool calls go through `dispatch_explore`; confirm calls through `dispatch`.
- [ ] Approval mode pauses at commit (approve, or reject with a note that becomes judge
      feedback) and before each judge downgrade, not on every explore step.
- [ ] Emit nodes as in section 5, with `phase` set.
- [ ] `FakeLLM` script for the new flow, so `python -m backend.agent` runs with no key.
- [ ] `AgentPort` adapter around `AgentRun` (still needed before the UI can run the agent live).
- [ ] Remove the `print(... # DEBUG)` in `execute_test`.

### C (product): `backend/api/`, `backend/events/`, `frontend/`

- [ ] `phase` in the contract (Python and TypeScript), coordinated with A and B.
- [ ] Graph layout: an explore lane (the trail, which can be 30 nodes, so collapsible) leading to
      the commit node, then a confirm lane with one column per finding.
- [ ] Inspector: for explore nodes, show the LLM's `thought` and the result summary; for confirm
      nodes, show the gates as they are today.
- [ ] Report view: the verdict table, plus "explored N analyses, committed K findings" from the
      run's nodes.
- [ ] Wire `create_app(agent=...)` to B's adapter.

## 7. Order of work

1. A: `dispatch_explore` and the gate `role` argument (B's explore and confirm depend on them).
2. C: `phase` in the contract.
3. B: explore loop and commit with `FakeLLM`; confirm as a plain function.
4. A: no-LLM runner and the planted-noise check on top of B's confirm.
5. B: judge, then `AgentPort` adapter. C: UI lanes and wiring.

## 8. Known limits

- **Validation is 3 seasons**, about a third the size of explore. Weak effects will often fail
  G1 and end Inconclusive. That is the honest outcome.
- **Tool state is per process** (`real._tests`, `_hypotheses`, `_holdout_used`). One run at a
  time, and `reset_tool_state()` at the start of each run.
- **The judge is an LLM.** It can miss things. Its only power is to send work back before
  confirmation or downgrade after, so a bad judge can cost a finding but never invent one.
