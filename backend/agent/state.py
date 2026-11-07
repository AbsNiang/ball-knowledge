"""Router state: pointers, stages, gate outcomes and counters, plus the two things the
specialists act on that the contract's `Node` has no field for: each track's current spec and
each step's tool call. Everything else about a node lives in the event log (`EventLog.graph`).
"""

from typing import Any, Literal, TypedDict

Mode = Literal["live", "approval"]
GateStatus = Literal["pass", "fail"]

# A track's `stage` is the next thing to do for it.
#   premise:    spec -> primary -> gated -> done
#   hypothesis: spec -> plan -> primary -> choose -> (follow_up -> choose)* -> interpret
#               -> verdict -> done
# At `choose` the LLM picks the next follow-up test from the menu the gates have unlocked.
Stage = Literal[
    "spec", "plan", "primary", "gated", "choose", "follow_up", "interpret", "verdict", "done"
]
DECISION_STAGES = frozenset({"spec", "plan", "choose", "interpret", "verdict"})
RESULT_STAGES = frozenset({"primary", "follow_up"})

FOLLOW_UPS = ("out_of_sample", "market_test", "placebo", "holdout_confirm")
LEDGER_TOOLS = frozenset({"compare_groups", "regress", "persistence", "market_test"})
HOLDOUT_REQUIRES = ("G1", "G2", "G3", "G4", "G5", "G6")
# A failed gate stops testing a hypothesis, except G6: failing the market check means "no
# betting edge", not "not real", so it only locks the holdout.
NON_STOPPING_GATES = frozenset({"G6"})

PREMISE = "premise"
MAX_REVISIONS = 2  # plan.md [A10]
REPORT_RESERVE = 2  # LLM calls kept back for the report (plan.md §8.5)
DEFAULT_BUDGET = {"max_llm_calls": 60, "max_tests": 40}

INCONCLUSIVE = "Inconclusive"
PREMISE_FALSE = "Premise false"
UNTESTABLE = "Untestable"
BUDGET_LABEL = "Inconclusive (budget)"
REVIEW_LABEL = "Inconclusive (review)"


class Budget(TypedDict):
    max_llm_calls: int
    max_tests: int
    llm_calls: int
    tests: int


class Step(TypedDict):
    """A node a track produced, in creation order. A revision truncates the list."""

    node: str
    stage: Stage  # the stage that produced it
    tool: str | None  # results: the tool that ran; choices: the action chosen
    gates: dict[str, GateStatus] | None  # results only: None until the gate lands
    call: dict[str, Any] | None  # plans, choices and results: {"tool": ..., "args": ...}


class Track(TypedDict):
    """One line of inquiry: the premise, or one hypothesis."""

    id: str
    kind: str
    stage: Stage
    spec_node: str | None
    spec: dict[str, Any] | None  # the claim or hypothesis spec in `spec_node`
    steps: list[Step]
    revisions: int
    feedback: str | None  # critic or human objection the next attempt must answer
    revising: str | None  # node the next attempt replaces (for the `revises` edge)
    forced_label: str | None  # label set by rule outside compute_verdict
    holdout_used: bool
    verdict: str | None
    verdict_node: str | None


class AgentState(TypedDict):
    run_id: str
    question: str
    mode: Mode
    question_node: str | None
    tracks: dict[str, Track]
    order: list[str]  # hypothesis ids in test order, chance last
    current: str | None
    outcome: str | None  # set when the run ends early: premise_false, untestable, ...
    budget: Budget
    exhausted: bool
    awaiting_gate: list[str]
    awaiting_critic: list[str]
    waiting_on: list[str]
    approval: list[str]  # decisions queued for a human (approval mode)
    report: dict[str, Any] | None
    status: str


def new_track(track_id: str, kind: str, stage: Stage) -> Track:
    return Track(
        id=track_id,
        kind=kind,
        stage=stage,
        spec_node=None,
        spec=None,
        steps=[],
        revisions=0,
        feedback=None,
        revising=None,
        forced_label=None,
        holdout_used=False,
        verdict=None,
        verdict_node=None,
    )


def initial_state(
    run_id: str, question: str, mode: Mode = "live", budget: dict[str, int] | None = None
) -> AgentState:
    limits = {**DEFAULT_BUDGET, **(budget or {})}
    return AgentState(
        run_id=run_id,
        question=question,
        mode=mode,
        question_node=None,
        tracks={PREMISE: new_track(PREMISE, "premise", "spec")},
        order=[],
        current=None,
        outcome=None,
        budget=Budget(**limits, llm_calls=0, tests=0),
        exhausted=False,
        awaiting_gate=[],
        awaiting_critic=[],
        waiting_on=[],
        approval=[],
        report=None,
        status="running",
    )


def result_steps(track: Track) -> list[Step]:
    return [step for step in track["steps"] if step["stage"] in RESULT_STAGES]


def track_gates(track: Track) -> dict[str, GateStatus]:
    """Every gate outcome the track's live results have earned so far."""
    gates: dict[str, GateStatus] = {}
    for step in result_steps(track):
        gates.update(step["gates"] or {})
    return gates


def latest(track: Track, stage: Stage) -> Step | None:
    return next((step for step in reversed(track["steps"]) if step["stage"] == stage), None)
