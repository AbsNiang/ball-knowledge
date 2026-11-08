"""Run state: the premise, the seeds and their research steps, and the phase that runs next.

Graph node details (titles, tool outputs, checks) live in the event log (`EventLog.graph`); this
state holds pointers, frozen specs, gate outcomes and counters.
"""

from typing import Any, Literal, TypedDict

Mode = Literal["live", "approval"]
GateStatus = Literal["pass", "fail"]

# What the router dispatches next. Each phase is also the name of the LangGraph node that runs it.
#   premise -> premise_test -> judge -> (premise again after a flawed review) -> seed
#   -> per seed: research -> execute -> judge -> supervise -> (research ... | conclude)
#   -> summarize
Phase = Literal[
    "premise",
    "premise_test",
    "seed",
    "research",
    "execute",
    "judge",
    "supervise",
    "conclude",
    "summarize",
    "done",
]
# Where a research step's tool sits in the verdict: the confirmatory roles run the seed's frozen
# primary test and decide its label; the rest only inform the judge.
Role = Literal[
    "declare", "primary", "replication", "market", "holdout", "explore", "search", "data"
]
CONFIRMATORY = ("primary", "replication", "market", "holdout")

PREMISE = "premise"
MAX_STEPS_PER_SEED = 8
MAX_PREMISE_REVISIONS = 1
REPORT_RESERVE = 2  # LLM calls kept back for the final summary (plan.md §8.5)
STEP_COST = 4  # worst-case LLM calls of one step: research, a search, the judge, the supervisor
DEFAULT_BUDGET = {"max_llm_calls": 60, "max_tests": 40}

INCONCLUSIVE = "Inconclusive"
NOT_TESTED = "Inconclusive (not tested)"
PREMISE_FALSE = "Premise false"
UNTESTABLE = "Untestable"
BUDGET_LABEL = "Inconclusive (budget)"
REVIEW_LABEL = "Inconclusive (review)"
STOPPING_GATES = ("G1", "G2", "G3", "G4", "G5")  # a failure here fixes the label for good
HOLDOUT_REQUIRES = ("G1", "G2", "G3", "G4", "G5", "G6")


class Budget(TypedDict):
    max_llm_calls: int
    max_tests: int
    llm_calls: int
    tests: int


class Step(TypedDict):
    """One research step: the researcher's call, what it produced, and the judge's review."""

    plan: str  # test_plan node: the researcher's tool call
    result: str | None  # test_result or source node
    judgment: str | None  # judgment node
    tool: str  # toolkit tool name
    role: Role
    args: dict[str, Any]  # the arguments the researcher gave
    gates: dict[str, GateStatus] | None  # gate outcomes on the result, if it was gated
    error: str | None  # why the step produced no evidence (tool error, human rejection)


class Seed(TypedDict):
    id: str
    node: str  # hypothesis node
    statement: str
    kind: str
    rationale: str
    testable_with_data: bool
    sources: list[str]  # source nodes it was drawn from
    status: Literal["queued", "active", "done", "pruned"]
    primary: dict[str, Any] | None  # the frozen HypothesisSpec, once declared
    primary_result: str | None  # test_result node of the primary test
    primary_test_id: str | None
    gates: dict[str, GateStatus]  # verdict gates, from confirmatory steps only
    used: list[str]  # one-shot tools already run (declare and the confirmatory ones)
    explored: int  # exploratory specs so far, for their ids
    steps: list[Step]
    judgment: dict[str, Any] | None  # the latest audited judgment
    guidance: str | None  # the supervisor's guidance for the next step
    feedback: str | None  # a tool error or human note the next step must answer
    supervision: dict[str, Any] | None  # the supervisor's last decision and its LLM record
    forced_label: str | None
    verdict: str | None
    verdict_node: str | None


class Premise(TypedDict):
    node: str | None  # premise_check node
    spec: dict[str, Any] | None
    missing_data: list[str]
    result: str | None  # test_result node
    gates: dict[str, GateStatus] | None
    judgment: str | None  # judgment node
    revisions: int
    feedback: str | None
    revising: str | None  # node the next premise decision replaces
    label: str | None  # set when the run ends at the premise
    verdict_node: str | None


class AgentState(TypedDict):
    run_id: str
    question: str
    mode: Mode
    question_node: str | None
    phase: Phase
    premise: Premise
    search_node: str | None
    seeds: dict[str, Seed]
    order: list[str]  # seed ids in research order
    current: str | None  # the active seed
    outcome: str | None  # set when the run ends early: premise_false, untestable, ...
    budget: Budget
    exhausted: bool
    approval: list[str]  # decisions queued for a human (approval mode)
    report: dict[str, Any] | None
    status: str


def new_premise() -> Premise:
    return Premise(
        node=None,
        spec=None,
        missing_data=[],
        result=None,
        gates=None,
        judgment=None,
        revisions=0,
        feedback=None,
        revising=None,
        label=None,
        verdict_node=None,
    )


def new_seed(
    seed_id: str,
    node: str,
    *,
    statement: str,
    kind: str,
    rationale: str = "",
    testable_with_data: bool = True,
    sources: list[str] | None = None,
) -> Seed:
    return Seed(
        id=seed_id,
        node=node,
        statement=statement,
        kind=kind,
        rationale=rationale,
        testable_with_data=testable_with_data,
        sources=list(sources or []),
        status="queued",
        primary=None,
        primary_result=None,
        primary_test_id=None,
        gates={},
        used=[],
        explored=0,
        steps=[],
        judgment=None,
        guidance=None,
        feedback=None,
        supervision=None,
        forced_label=None,
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
        phase="premise",
        premise=new_premise(),
        search_node=None,
        seeds={},
        order=[],
        current=None,
        outcome=None,
        budget=Budget(**limits, llm_calls=0, tests=0),
        exhausted=False,
        approval=[],
        report=None,
        status="running",
    )


def stopping_failures(gates: dict[str, GateStatus]) -> list[str]:
    """Failed gates that fix a seed's label, so no further test can change it. A gate that
    errored (the pseudo-gate "error") fails closed the same way."""
    return sorted(g for g in (*STOPPING_GATES, "error") if gates.get(g) == "fail")


def primary_held(gates: dict[str, GateStatus]) -> bool:
    return all(gates.get(g) == "pass" for g in ("G1", "G2", "G3", "G4"))
