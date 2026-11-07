# A ↔ B interface: stats layer and agent

**Status: proposal, not agreed.** Nothing here is in `contracts/` yet. Section 3 becomes
`contracts/tools.py` and `contracts/tools.ts` once A, B and C sign off. Where this differs from
`plan.md` or `TASKS.md`, section 8 says so and why.

## 1. The boundary in one picture

```
             B: agent                               A: data and stats
┌───────────────────────────────┐        ┌───────────────────────────────────────┐
│ LLM decisions D1-D6           │        │ ResearchSession (one per run)         │
│ critic, taint, revisions      │  specs │  ├─ hypothesis registry (frozen specs)│
│ LangGraph control flow        │ ─────► │  ├─ global BH ledger                  │
│ budget, modes, interrupts     │        │  ├─ holdout seal                      │
│                               │ ◄───── │  ├─ gates G1-G7, compute_verdict      │
│ hands results to C's event    │ results│  └─ pure tools ──► data layer         │
│ layer as check_result etc.    │        │                                       │
└───────────────────────────────┘        └───────────────────────────────────────┘
```

Four principles:

1. **B says what to test, A says what it means.** B supplies hypotheses. A returns numbers, gate
   results and verdict labels. B never computes a statistic and never picks a label.
2. **B does not touch data.** `backend/agent/` imports only the two entry points in section 2. It
   never imports `backend/data/` and never sees a dataframe.
3. **Tools stay pure, and state lives in the session.** The ledger, frozen specs and holdout use
   can't live inside pure functions, so `ResearchSession` wraps the tools and owns that state.
4. **Nothing raises across the boundary.** Failures come back as `status="error"` plus a `reason`.

## 2. What B imports

```text
from backend.gate.session import open_session        # section 4
from backend.tools.registry import tool_declarations  # section 5
from contracts.tools import HypothesisSpec, ToolResult, CheckResult, Verdict, ...
```

A test will fail if `backend/agent/` imports anything else from `backend/data`, `tools` or `gate`.

## 3. Shared models (proposed `contracts/tools.py`)

```python
"""Models shared across the A/B boundary."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Metric = Literal["ats_margin", "ats_cover", "margin", "win", "total_over", "roi"]
Direction = Literal["a_gt_b", "a_lt_b", "two_sided"]
PrimaryTest = Literal["compare_groups", "regress", "persistence", "market_test"]
HypothesisKind = Literal["premise", "explanation", "confounder", "chance", "market"]
FilterColumn = Literal[
    "season", "team", "opp", "is_home", "rest_days", "b2b", "opp_b2b", "travel_km",
    "tz_shift", "altitude_m", "month", "is_bubble", "is_playoffs", "games_into_season",
]
ToolName = Literal[
    "describe_dataset", "define_slice", "add_feature", "power", "compare_groups", "regress",
    "market_test", "persistence", "placebo", "out_of_sample", "holdout_confirm",
]
GateId = Literal["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
TestPurpose = Literal["primary", "secondary", "g6_market", "g6_spread_control"]
ErrorReason = Literal[
    "invalid_args",        # args failed validation: message says why, the LLM may retry
    "unknown_hypothesis",  # hypothesis_id was never registered
    "not_eligible",        # the gates so far don't allow this step
    "already_used",        # holdout_confirm has been called for this hypothesis
    "holdout_forbidden",   # something tried to read holdout rows outside holdout_confirm
    "empty_slice",         # the slice matched no rows
    "session_finalized",   # finalize() has already run for this session
    "internal_error",      # a bug on A's side
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filter(Strict):
    col: FilterColumn
    op: Literal["eq", "ne", "in", "gte", "lte", "between"]
    value: bool | int | float | str | list[int] | list[float] | list[str]


class SliceSpec(Strict):
    split: Literal["explore", "validate", "special"] = "explore"  # holdout can't be named
    filters: list[Filter] = []


class StatSpec(Strict):
    """A testable claim, and everything that identifies its test."""

    statement: str
    kind: HypothesisKind
    metric: Metric
    group_a: SliceSpec
    group_b: SliceSpec | Literal["complement"]
    direction: Direction
    primary_test: PrimaryTest
    controls: list[str] = []


class MechanismSpec(Strict):
    statement: str
    secondary_test: StatSpec  # a StatSpec, so the schema isn't recursive


class HypothesisSpec(StatSpec):
    """D2's output. D1's Claim is the same model with kind="premise"."""

    mechanism_prediction: MechanismSpec | None = None


class Registration(Strict):
    hypothesis_id: str
    status: Literal["ok", "error"]
    spec_hash: str | None = None
    reason: ErrorReason | None = None
    message: str | None = None


class ToolResult(Strict):
    """Everything here is deterministic. B measures duration_ms itself."""

    tool: ToolName
    status: Literal["ok", "error"]
    reason: ErrorReason | None = None
    message: str | None = None
    args: dict[str, Any]
    args_hash: str
    data_version: str | None = None
    data_slice_hash: str | None = None
    seed: int | None = None
    effect: float | None = None  # mean(A) - mean(B)
    effect_unit: Literal["points", "fraction"] | None = None  # 0.03 means 3 percentage points
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    n_a: int | None = None
    n_b: int | None = None
    mde: float | None = None
    extra: dict[str, Any] = {}  # tool-specific summary numbers
    charts: dict[str, Any] = {}  # small arrays for the UI, kept out of LLM prompts

    def for_llm(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"charts"}, exclude_none=True)


class LedgerEntry(Strict):
    seq: int = Field(ge=1)
    spec_hash: str
    hypothesis_id: str
    tool: ToolName
    p_value: float


class CheckResult(Strict):
    """One gate, critic or human check. `gate_id` is set only for gates."""

    node_id: str | None
    hypothesis_id: str | None = None
    checker: Literal["gate", "critic", "human"]
    gate_id: GateId | None = None
    status: Literal["pass", "flag", "fail", "pending", "not_applicable"]
    provisional: bool = False  # True for G4 until finalize()
    severity: Literal["low", "medium", "high"] = "low"
    rationale: str
    metrics: dict[str, float] = {}  # the numbers behind the result, for the critic to cite


class ToolCallOutcome(Strict):
    result: ToolResult
    ledger_entry: LedgerEntry | None = None  # None for diagnostics and identical re-runs
    checks: list[CheckResult] = []  # gates this result newly settled or changed


class NextStep(Strict):
    action: Literal["run_tool", "compute_verdict"]
    tool: ToolName | None = None
    args: dict[str, Any] = {}


class Verdict(Strict):
    hypothesis_id: str
    label: Literal[
        "untestable", "inconclusive", "inconclusive_budget", "luck", "premise_false",
        "premise_passed", "real_priced_in", "real_beats_market_provisional",
        "chance_supported", "chance_refuted", "confounder_supported", "confounder_refuted",
    ]
    rule: str  # the row of the verdict table that fired, for example "9.2#3"
    flags: list[
        Literal["holdout_failed", "human_overridden", "g4_flipped", "no_replication_split"]
    ] = []
    explanation_status: Literal[
        "mechanism_supported", "mechanism_failed", "mechanism_untested"
    ] = "mechanism_untested"
    checks: list[CheckResult]
    ends_run: bool = False  # True when a premise fails


class FinalReview(Strict):
    flips: list[CheckResult]  # G4 results that changed, with the final status
    verdicts: list[Verdict]  # these replace all earlier verdicts
    tests_spent: int


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def canonical_spec(spec: StatSpec) -> dict[str, Any]:
    """The fields that define a test. Wording and ids don't count, and order doesn't matter."""
    data = spec.model_dump(
        mode="json", exclude={"statement", "kind", "mechanism_prediction"}
    )
    for group in ("group_a", "group_b"):
        if isinstance(data[group], dict):
            for item in data[group]["filters"]:
                if isinstance(item["value"], list):
                    item["value"] = sorted(item["value"])
            data[group]["filters"].sort(key=lambda item: json.dumps(item, sort_keys=True))
    data["controls"] = sorted(data["controls"])
    return data


def spec_hash(tool: ToolName, purpose: TestPurpose, spec: StatSpec) -> str:
    """The ledger key. Also ToolResult.args_hash, the test id and the seed input."""
    return stable_hash({"tool": tool, "purpose": purpose, "spec": canonical_spec(spec)})
```

## 4. `ResearchSession` (A owns it, B calls it)

```python
from typing import Any, Protocol

from contracts.tools import (
    CheckResult, FinalReview, GateId, HypothesisSpec, NextStep, Registration, StatSpec,
    ToolCallOutcome, Verdict,
)


class ResearchSession(Protocol):
    run_id: str

    def register_hypothesis(
        self,
        hypothesis_id: str,  # B's Hypothesis node id
        spec: HypothesisSpec | StatSpec,
        revises: str | None = None,  # the hypothesis this one replaces
        mechanism_of: str | None = None,  # the hypothesis this secondary test belongs to
    ) -> Registration:
        """Validate and freeze a spec. It can't be edited after this."""

    def call(self, tool: str, args: dict[str, Any], node_id: str | None = None) -> ToolCallOutcome:
        """The only way to run a tool. Blocking and CPU-bound, and never raises."""

    def next_step(self, hypothesis_id: str) -> NextStep:
        """What the gates require next, with exact args. Works out the G1-G7 order."""

    def mark_untestable(self, hypothesis_id: str, missing_data: list[str]) -> None:
        """Record D1 or D3 finding that no whitelisted tool can express this."""

    def compute_verdict(self, hypothesis_id: str, budget_exhausted: bool = False) -> Verdict:
        """Apply the verdict table. Pure over the stored gate results."""

    def record_override(
        self, hypothesis_id: str, gate_id: GateId, note: str, actor: str
    ) -> CheckResult:
        """Apply a human override. A G7 override, or a note under 10 chars, returns a fail."""

    def tests_spent(self) -> int:
        """Ledger size, for the budget check and the top-bar counter."""

    def finalize(self) -> FinalReview:
        """Re-run G4 over the whole ledger and recompute all verdicts. Call once, at report time."""


def open_session(
    run_id: str,
    backend: str = "real",  # "stub" for section 7
    stub_scenarios: dict[str, str] | None = None,
) -> ResearchSession:
    """Open or resume a run's session. Ledger and holdout use survive a restart."""
    raise NotImplementedError
```

## 5. Tool registry

`tool_declarations(exposure="llm")` returns `{name, description, json_schema}` for Gemini, with
each schema inlined so it has no `$ref`. `exposure="all"` adds the orchestrator-only tools, and
`validate_args(name, args)` runs the Pydantic check.

The LLM must not be able to reach the sealed holdout or skip a gate, so `exposure` is part of the
registry rather than a prompt instruction.

| Tool | Exposed to | Args | Ledger entry? |
|---|---|---|---|
| `describe_dataset` | LLM | none | no |
| `define_slice` | LLM | `SliceSpec` | no |
| `add_feature` | LLM | `name`, `params` | no |
| `power` | LLM | `n_a`, `n_b`, `metric` | no |
| `compare_groups` | LLM | `hypothesis_id` | yes |
| `regress` | LLM | `hypothesis_id` | yes |
| `persistence` | LLM | `hypothesis_id` | yes |
| `market_test` | LLM | `hypothesis_id` | yes |
| `placebo` | LLM | `hypothesis_id`, `placebo_slice: SliceSpec` | no |
| `out_of_sample` | orchestrator | `hypothesis_id` | no |
| `holdout_confirm` | orchestrator | `hypothesis_id` | no |

Test tools take a `hypothesis_id`, not a spec. The spec was frozen at registration, so D3 can't
change the test after seeing other results. The session returns `not_eligible` if a test tool
doesn't match the hypothesis's declared `primary_test`. The two G6 runs come from `next_step`,
which adds `purpose`.

## 6. How a run uses the interface

| Stage | B does | B calls | B gets back |
|---|---|---|---|
| Start | Opens the run | `open_session(run_id)`, `tool_declarations("llm")` | Session, tool list for prompts |
| Context | Reads what data exists | `call("describe_dataset", {})` | Columns, metrics, splits, no holdout counts |
| D1 claim | Parses the Claim, or `Untestable` | `register_hypothesis(id, claim)` | `Registration`, or `mark_untestable` then `compute_verdict` |
| Premise test | Follows `next_step` | `next_step(id)`, `call(...)` | `ToolCallOutcome` |
| Premise result | Ends the run if `ends_run` | `compute_verdict(id)` | `premise_passed`, `premise_false` or `inconclusive` |
| D2 hypotheses | Parses 3-5 specs | `register_hypothesis` for each | `Registration` for each; an `invalid_args` one goes back to the LLM |
| D3 and tests | Repeats until `compute_verdict` comes up | `next_step(id)`, `call(tool, args, node_id)` | Results and gate checks |
| D4 interpret | Prompts with `result.for_llm()` and checks | none | D4 cites `gate_id`s |
| D5 mechanism | Runs the secondary test | `register_hypothesis(.., mechanism_of=id)`, `call` | Sets `explanation_status`, never the label |
| Verdict | Writes the D6 narrative around the label | `compute_verdict(id)` | `Verdict` |
| Budget out | Finishes open hypotheses | `compute_verdict(id, budget_exhausted=True)` | `inconclusive_budget` |
| Human override | Passes on C's intervention | `record_override(...)` | `CheckResult` |
| Report | Drains checks, then builds from the final verdicts | `finalize()` | `FinalReview` |

The loop for one hypothesis, in B's code:

```text
while (step := session.next_step(hypothesis_id)).action == "run_tool":
    # B creates a Test Result node, and may interrupt() here in Approval mode
    outcome = await asyncio.to_thread(session.call, step.tool, step.args, node_id)
    emit(outcome.result, outcome.ledger_entry, outcome.checks)
verdict = session.compute_verdict(hypothesis_id)
```

B keeps its LangGraph edges and decides **when** to step. A decides **what** the next required step
is, so the gate order lives in one place. Gate results come back inside `call`, so a conditional
edge never races a gate. The critic is still async.

Where results go: `result` onto the Test Result node, `checks` to `check_result` events,
`ledger_entry` and `Verdict` to their events, and `supports`/`refutes` edges from the G2 and G3
statuses. A never writes events itself.

## 7. What each side can rely on

**A guarantees**

- The same `(tool, args, data_version)` gives an identical `ToolResult`.
- An identical re-run costs nothing on the ledger, however it is reworded.
- `ToolResult.args`, `.extra` and `.charts` are summaries, never rows.
- Holdout rows are readable only inside `holdout_confirm`, and a test proves it.
- `holdout_confirm` marks itself used before it reads any data, so a crash can't grant a second look.
- A revised hypothesis inherits its parent's holdout use, so tweaking a spec can't buy another look.
- Session state survives a restart, in A's own SQLite file, so B has nothing to persist. The event
  log stays the record of what happened. This file only enforces the rules.

**B guarantees**

- Tools run off the event loop, through `asyncio.to_thread`. A test takes 10,000 permutations, so
  calling it directly would freeze the WebSocket.
- Every hypothesis is registered before it is tested, using B's node id as `hypothesis_id`.
- The LLM only sees tools from `tool_declarations("llm")`.
- `compute_verdict` is the only source of labels, and no label is rewritten after it returns.
- `finalize()` runs once, after outstanding checks are drained, and the report uses its verdicts
  rather than earlier ones.
- `chance` hypotheses are registered with the premise's fields. They get no test, and
  `next_step` goes straight to `compute_verdict`, which applies the chance rule.

**The critic** reads `result.for_llm()` and gate `CheckResult`s, and `metrics` gives it numbers to
cite. It writes `flag`/`fail` onto nodes through B. It has no write path into the session, so it
can't change a gate or a label.

**Stubs (what A ships first).** `open_session(run_id, backend="stub", stub_scenarios=...)` has the
same API and returns canned results, so B builds against the real shapes and can force each path:

```text
open_session("demo", backend="stub", stub_scenarios={
    "premise": "pass",            # or premise_false, inconclusive
    "explanation": "priced_in",   # or luck, beats_market, holdout_failed, inconclusive
    "confounder": "refuted",      # or supported
})
```

## 8. Changes from the docs

| Doc says | Proposal | Why |
|---|---|---|
| `Filter.value: Any` (v2) | A typed union | `Any` produces no schema type, so Gemini gets nothing to constrain on |
| `MechanismSpec.secondary_test: HypothesisSpec` | `StatSpec` | Otherwise the schema is recursive and can't be inlined |
| `HypothesisSpec.id` (v2) | Passed to `register_hypothesis` | The LLM shouldn't invent ids, and ids mustn't change the hash |
| `compare_groups(spec)` etc. | Test tools take `hypothesis_id` | Keeps the primary test frozen, and shortens D3 |
| `TestPlan` chosen by the LLM (D3) | Follows `next_step` for gated steps | v2 8.1 already says gates pick these edges, not LLM opinion |
| `out_of_sample(test_id)` | Takes `hypothesis_id` | Each hypothesis has one primary test |
| `ToolResult.n` (TASKS) | `n_a`, `n_b`, plus `status`, `reason`, `mde`, `seed`, `args_hash` | G1 needs both groups, and v2 needs the rest |
| `CheckResult.status`: pass, flag, fail | Adds `pending`, `not_applicable`, `provisional` | G5 isn't run yet, G6 can be N/A, G4 is provisional |
| Binary effect "in percentage points" | Fractions, with `effect_unit` | SESOI and BREAKEVEN are fractions, so mixing units would break G1 |
| `run_gate` as a background task (4.4) | Gates run inside `call` | They are cheap once the tool has run, and this removes a race |
| `duration_ms` in the tool record | Measured by B | It varies, so it can't be in a deterministic result |

## 9. Gaps in the spec to settle

**Need B's agreement**

1. **Session state lives in A's own SQLite file, not rebuilt from events.** This keeps B out of it
   and lets the holdout seal survive a crash. It is a small exception to [A2]. It enforces rules
   and doesn't describe the graph, and replay never touches it.
2. **D3 gets much smaller.** With frozen specs and `next_step`, the LLM chooses little except
   diagnostics like `placebo` and `power`. Does B want more freedom there?
3. **G6 needs a second test that 8.1 doesn't show.** It requires `market_test` and also a `regress`
   with `spread` as a control. `next_step` will return both, so B's graph needs to allow two steps
   at that stage.
4. **Gemini schema support.** The `HypothesisSpec` schema needs testing against real D1 and D2
   structured output. I checked it can be inlined (no `$ref`), but not against Gemini. B can
   confirm with one real call.

**Need B and C**

5. **Two event kinds are missing.** `contracts/events.py` has no `ledger_entry` or
   `verdict_computed`, which v2 lists. Add them, or tell me where these go. The rest of the
   envelope debate (`kind/data` against `actor/type`) doesn't affect this interface.
6. **New verdict labels.** Section 9.2 has no outcome for `chance` and `confounder` hypotheses, so
   I added four labels and `premise_passed`. C needs display wording.

**My call as gate owner, shown so B isn't surprised**

7. **The bubble question can't pass the plan as written.** 8.3 runs the premise on explore seasons,
   but the bubble is in `special`, so the slice is empty and it always ends Inconclusive. Its OOS and
   holdout slices are empty too. Default: the premise uses the spec's own split, G5 and G7 become
   `not_applicable`, and the verdict is capped at `inconclusive` with `no_replication_split`.
8. **`season` filters break the split swap.** A filter on an explore season is empty on validate, so
   `register_hypothesis` rejects `season` filters with `invalid_args`, except on `special`.
9. **Late G4 flips.** BH can flip a result either way as the ledger grows. A pass that becomes a
   fail downgrades normally. A fail that becomes a pass stays `luck` with `g4_flipped`, because that
   hypothesis stopped early and has no replication evidence.
10. **Confounder gating.** These can't use G2-G7 as written, since the claim is "the effect shrinks
    by at least 50% once controls are added". I'll gate them on G1 plus that shrink rule.
