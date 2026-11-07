"""Stateful router (https://docs.langchain.com/oss/python/langchain/multi-agent/router).

Every step runs through the router. It folds finished checks into state, then dispatches exactly
one specialist with `Command(goto=...)`; every specialist returns to the router.

The router enforces the rules and the LLM makes the choices:
- It waits for a test's gates before anything builds on the result.
- If a gate fails, testing that hypothesis stops and the failure decides its verdict (G6 is the
  exception: it only locks the holdout). Re-testing after a failure would be p-hacking.
- If the gates pass, the LLM picks the next follow-up test (`choose_test`), but only from the
  menu the gates have unlocked, or concludes. Skipping a test can only weaken a verdict.

State persists across invocations through the checkpointer, which lets approval mode pause on
`interrupt()` and resume later.
"""

import copy
from collections.abc import Iterable
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Command

from backend.agent.checks import CheckBatch
from backend.agent.context import AgentContext
from backend.agent.ports import CheckResult
from backend.agent.record import Recorder
from backend.agent.state import (
    BUDGET_LABEL,
    DECISION_STAGES,
    HOLDOUT_REQUIRES,
    INCONCLUSIVE,
    MAX_REVISIONS,
    NON_STOPPING_GATES,
    PREMISE,
    PREMISE_FALSE,
    REPORT_RESERVE,
    RESULT_STAGES,
    REVIEW_LABEL,
    AgentState,
    Budget,
    Step,
    Track,
    result_steps,
    track_gates,
)
from backend.events.projection import ResearchGraph
from contracts.api import ModePayload, SetMode

ROUTES = (
    "premise",
    "hypothesize",
    "plan_test",
    "execute_test",
    "choose_test",
    "interpret",
    "verdict",
    "report",
    "wait_checks",
    "human_review",
)
_NODE_STATE = {"pass": "passed", "flag": "flagged", "fail": "failed"}
# Worst-case LLM calls per step: the decision plus a critic review of each node it creates.
_LLM_COST = {
    "premise": 2,
    "hypothesize": 6,
    "plan_test": 2,
    "choose_test": 2,
    "interpret": 2,
    "verdict": 2,
}


def follow_up_menu(track: Track) -> tuple[str, ...]:
    """What the LLM may choose next. Each follow-up runs at most once per hypothesis, and the
    holdout unlocks only after G1-G6 pass."""
    ran = {step["tool"] for step in result_steps(track)}
    menu = [tool for tool in ("out_of_sample", "market_test", "placebo") if tool not in ran]
    gates = track_gates(track)
    if not track["holdout_used"] and all(gates.get(g) == "pass" for g in HOLDOUT_REQUIRES):
        menu.append("holdout_confirm")
    return (*menu, "conclude")


def stopping_failures(track: Track) -> list[str]:
    """Failed gates that end testing for this hypothesis."""
    gates = track_gates(track)
    return sorted(g for g, v in gates.items() if v == "fail" and g not in NON_STOPPING_GATES)


class Draft:
    """A mutable copy of the routing state. `changes()` returns it as a LangGraph update."""

    def __init__(self, state: AgentState) -> None:
        self.tracks: dict[str, Track] = copy.deepcopy(state["tracks"])
        self.order: list[str] = list(state["order"])
        self.outcome: str | None = state["outcome"]
        self.exhausted: bool = state["exhausted"]
        self.awaiting_gate: list[str] = list(state["awaiting_gate"])
        self.awaiting_critic: list[str] = list(state["awaiting_critic"])
        self.approval: list[str] = list(state["approval"])
        self.node_states: list[tuple[str, str]] = []  # (node_id, state) events to emit

    def changes(self) -> dict[str, Any]:
        return {
            "tracks": self.tracks,
            "order": self.order,
            "outcome": self.outcome,
            "exhausted": self.exhausted,
            "awaiting_gate": self.awaiting_gate,
            "awaiting_critic": self.awaiting_critic,
            "approval": self.approval,
        }

    # -- folding checks ---------------------------------------------------------------------

    def fold_gate(self, node_id: str, results: list[CheckResult]) -> None:
        _discard(self.awaiting_gate, node_id)
        found = self._find(node_id)
        if found is None or found[1]["stage"] not in RESULT_STAGES:
            return  # superseded by a revision; the check stays on the node but steers nothing
        found[1]["gates"] = {
            r.gate_id: "pass" if r.status == "pass" else "fail" for r in results if r.gate_id
        }
        failed = any(result.status != "pass" for result in results)
        self.node_states.append((node_id, "failed" if failed else "passed"))

    def fold_critic(self, node_id: str, results: list[CheckResult], graph: ResearchGraph) -> None:
        _discard(self.awaiting_critic, node_id)
        for result in results:
            self.review(node_id, result.status, result.rationale, graph)

    def review(self, node_id: str, status: str, feedback: str, graph: ResearchGraph) -> None:
        """Apply a critic or human review to a decision node (plan.md §8.4)."""
        if status != "pass":
            _discard(self.approval, node_id)
        found = self._find(node_id)
        if found is None or found[1]["stage"] not in DECISION_STAGES:
            return
        track, step = found
        self.node_states.append((node_id, _NODE_STATE[status]))
        if status != "pass" and not self.exhausted:
            self._rewind(track, step, feedback, graph)

    def _rewind(self, track: Track, step: Step, feedback: str, graph: ResearchGraph) -> None:
        """Send `track` back to the decision in `step`; taint everything built on it."""
        stage = step["stage"]
        if stage == "verdict" and track["revisions"] >= MAX_REVISIONS:
            return  # the label is the gate's; only its narrative was disputed, so it stands
        tainted = graph.descendants(step["node"])
        self.node_states += [(nid, "tainted") for nid in tainted]
        self.approval = [nid for nid in self.approval if nid not in tainted]

        cut = track["steps"].index(step)
        dropped = track["steps"][cut:]
        track["steps"] = track["steps"][:cut]
        track.update(verdict=None, verdict_node=None)
        loses_holdout = any(
            s["stage"] == "follow_up" and s["tool"] == "holdout_confirm" for s in dropped
        )
        if track["revisions"] >= MAX_REVISIONS or loses_holdout:
            track.update(forced_label=REVIEW_LABEL, stage="verdict", feedback=None, revising=None)
            if track["id"] == PREMISE:
                self.outcome = "review"
            return

        track["revisions"] += 1
        track.update(feedback=feedback, revising=step["node"], stage=stage)
        if stage != "verdict":
            track["forced_label"] = None
            if track["id"] == PREMISE:
                self.outcome = None
        if track["id"] == PREMISE and stage == "spec":
            self.tracks = {PREMISE: track}
            self.order = []

    def _find(self, node_id: str) -> tuple[Track, Step] | None:
        for track in self.tracks.values():
            for step in track["steps"]:
                if step["node"] == node_id:
                    return track, step
        return None

    # -- routing ----------------------------------------------------------------------------

    def exhaust(self) -> None:
        """Budget hit: unfinished tracks get `Inconclusive (budget)` (plan.md [A11])."""
        self.exhausted = True
        for track in self.tracks.values():
            unfinished = track["stage"] not in ("verdict", "done") or (
                track["kind"] == "chance" and track["stage"] != "done"
            )
            if unfinished:
                track.update(forced_label=BUDGET_LABEL, stage="verdict")
        if self.tracks[PREMISE]["stage"] != "done" or not self.order:
            self.outcome = self.outcome or "budget"

    def advance(self, track: Track) -> tuple[str, list[str]] | None:
        """The specialist to run next for `track` (with nodes to wait on), or None when done."""
        while True:
            stage = track["stage"]
            if stage == "done":
                return None
            if stage == "spec":
                return ("premise" if track["id"] == PREMISE else "hypothesize"), []
            if stage == "plan":
                return "plan_test", []
            if stage in RESULT_STAGES:
                return "execute_test", []
            pending = [s["node"] for s in result_steps(track) if s["gates"] is None]
            if pending:
                return "wait_checks", pending  # nothing builds on a result before its gates
            if stage == "gated":
                self._settle_premise(track)
                continue
            if stage == "choose":
                if stopping_failures(track) or follow_up_menu(track) == ("conclude",):
                    track["stage"] = "interpret"  # a gate failed, or nothing is left to run
                    continue
                return "choose_test", []
            return stage, []  # "interpret" or "verdict": the specialist has the stage's name

    def _settle_premise(self, track: Track) -> None:
        """plan.md §8.3: G1 fails -> Inconclusive; G2 or G3 fails -> Premise false; else go on."""
        gates = track_gates(track)
        if gates.get("G1") != "pass":
            track.update(forced_label=INCONCLUSIVE, stage="verdict")
            self.outcome = "inconclusive"
        elif gates.get("G2") != "pass" or gates.get("G3") != "pass":
            track.update(forced_label=PREMISE_FALSE, stage="verdict")
            self.outcome = "premise_false"
        else:
            track["stage"] = "done"


def fold_checks(state: AgentState, batches: Iterable[CheckBatch], graph: ResearchGraph) -> Draft:
    draft = Draft(state)
    for batch in batches:
        if batch.checker == "gate":
            draft.fold_gate(batch.node_id, batch.results)
        else:
            draft.fold_critic(batch.node_id, batch.results, graph)
    return draft


def over_budget(budget: Budget) -> bool:
    llm_left = budget["max_llm_calls"] - budget["llm_calls"]
    return llm_left <= REPORT_RESERVE or budget["tests"] >= budget["max_tests"]


def affordable(state: AgentState, goto: str) -> bool:
    """Whether `goto` fits the LLM budget while keeping the report's reserve. Critics still
    pending count as spent, so the cap holds even though they run in the background."""
    budget = state["budget"]
    committed = budget["llm_calls"] + len(state["awaiting_critic"])
    cost = _LLM_COST.get(goto, 0)
    return cost == 0 or committed + cost + REPORT_RESERVE <= budget["max_llm_calls"]


def route(state: AgentState) -> tuple[dict[str, Any], str]:
    """Pick the next specialist. Returns (state update, node name)."""
    draft = Draft(state)
    if draft.approval:
        head = draft.approval[0]
        waiting_on = [head] if head in draft.awaiting_critic else []
        goto = "wait_checks" if waiting_on else "human_review"
        return _go(draft, goto, state["current"], waiting_on)

    if not draft.exhausted and over_budget(state["budget"]):
        draft.exhaust()
    goto, current, waiting_on = _next_step(draft)
    if not draft.exhausted and not affordable(state, goto):
        draft.exhaust()
        goto, current, waiting_on = _next_step(draft)
    return _go(draft, goto, current, waiting_on)


def _next_step(draft: Draft) -> tuple[str, str | None, list[str]]:
    for track_id in [PREMISE, *draft.order]:
        track = draft.tracks[track_id]
        pending = sorted({*draft.awaiting_gate, *draft.awaiting_critic})
        if track["kind"] == "chance" and track["stage"] == "verdict" and pending:
            return "wait_checks", None, pending  # chance is judged on settled competitors
        step = draft.advance(track)
        if step is not None:
            return step[0], track_id, step[1]
        if track_id == PREMISE:
            if draft.outcome:
                break
            if not draft.order:
                return "hypothesize", None, []

    pending = sorted({*draft.awaiting_gate, *draft.awaiting_critic})
    if pending:
        return "wait_checks", None, pending
    return "report", None, []


def _go(
    draft: Draft, goto: str, current: str | None, waiting_on: list[str]
) -> tuple[dict[str, Any], str]:
    return {**draft.changes(), "current": current, "waiting_on": waiting_on}, goto


async def router(state: AgentState, runtime: Runtime[AgentContext]) -> Command:
    ctx = runtime.context
    # Pause between steps; let an in-flight tool/check finish without losing evidence.
    rec = Recorder(ctx.log)
    if not ctx.running.is_set():
        rec.status("paused")
        await ctx.running.wait()

    mode = state["mode"]
    for message in ctx.take_inbox():
        if message.get("type") == "set_mode" and message.get("mode") in ("live", "approval"):
            rec.intervention(SetMode(type="set_mode", payload=ModePayload(mode=message["mode"])))
            if message["mode"] != mode:
                mode = message["mode"]
                rec.mode(mode)
    budget = Budget(**{**state["budget"], "llm_calls": ctx.llm.calls})
    state = {**state, "mode": mode, "budget": budget}
    if mode == "live":
        state["approval"] = []

    draft = fold_checks(state, ctx.checks.drain(), ctx.log.graph)
    lost = {*draft.awaiting_gate, *draft.awaiting_critic} - ctx.checks.pending()
    if lost:
        raise RuntimeError(f"checks were submitted but never finished for {sorted(lost)}")
    for node_id, node_state in draft.node_states:
        rec.mark(node_id, node_state)

    update, goto = route({**state, **draft.changes()})
    rec.budget(budget["llm_calls"], budget["tests"])
    return Command(update={"mode": mode, "budget": budget, **update}, goto=goto)


def _discard(items: list[str], item: str) -> None:
    if item in items:
        items.remove(item)
