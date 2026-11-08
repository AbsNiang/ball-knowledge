"""Specialists the router dispatches to. Each builds only the context its decision needs.

LLM specialists make one structured decision, write it as a research-graph node, and queue the
critic (and the human, in approval mode): D1 `premise`, D2 `hypothesize`, D3 `plan_test`,
D3b `choose_test` (the next follow-up after a passing gate), D4 `interpret`, D6 `verdict`
prose and `report`. `execute_test` runs tools; verdict labels come from the gate.
"""

import asyncio
import copy
import re
import time
from typing import Any, Literal, Self, get_args

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError, model_validator

from backend.agent.context import AgentContext
from backend.agent.ports import CheckResult, ToolResult, from_gate
from backend.agent.prompts import PROMPTS
from backend.agent.record import Recorder, contract_verdict, llm_record, tool_record
from backend.agent.router import Draft, follow_up_menu, stopping_failures
from backend.agent.schemas import (
    ClaimDecision,
    HypothesisRevision,
    HypothesisSet,
    Interpretation,
    Narrative,
    NextTest,
    PlanDecision,
    PrimaryTest,
    ReportDraft,
    next_test_schema,
)
from backend.agent.state import (
    BUDGET_LABEL,
    LEDGER_TOOLS,
    PREMISE,
    REVIEW_LABEL,
    UNTESTABLE,
    AgentState,
    Stage,
    Step,
    Track,
    latest,
    new_track,
    result_steps,
    track_gates,
)
from backend.events.projection import ResearchGraph
from backend.tools.registry import TOOL_REGISTRY
from contracts.api import Approve, OptionalNote, Reject, RequiredNote
from contracts.models import LLMRecord

_SILENT_LABELS = frozenset({BUDGET_LABEL, REVIEW_LABEL})
MENU_HELP = {
    "out_of_sample": "Re-run the frozen test on the validation seasons (G5: same sign, at least "
    "half the effect).",
    "market_test": "Check whether betting on it beats the closing line after vig (G6: cover-rate "
    "CI above 52.38%, and the effect survives the spread as a control).",
    "placebo": "Run the same test where the effect should be absent; set placebo_slice.",
    "holdout_confirm": "Confirm once on the sealed holdout seasons (G7). Cannot be repeated.",
    "conclude": "Stop testing this hypothesis. Skipped tests can only weaken its verdict.",
}
# What each primary test does (Person A's registry text, kept in sync) and when to pick it.
_WHEN_TO_USE = {
    "compare_groups": "Use it for a difference between two groups of team-games, which fits "
    "most explanations.",
    "regress": "Use it when other factors must be held fixed: every confounder hypothesis, with "
    "the confounders as controls.",
    "persistence": "Use it for team-level claims that may be regression to the mean: does a "
    "team's metric this season carry into the next?",
    "market_test": "Use it when the claim is about beating the betting line itself.",
}
_REGISTRY_TEXT = {tool["name"]: tool["description"] for tool in TOOL_REGISTRY}
PRIMARY_TOOL_HELP = {
    name: f"{_REGISTRY_TEXT[name]} {_WHEN_TO_USE[name]}" for name in get_args(PrimaryTest)
}


# -- helpers ---------------------------------------------------------------------------------


def _deps(runtime: Runtime[AgentContext]) -> tuple[AgentContext, Recorder, ResearchGraph]:
    ctx = runtime.context
    return ctx, Recorder(ctx.log), ctx.log.graph


def _current(state: AgentState) -> Track:
    return copy.deepcopy(state["tracks"][state["current"]])


def _put(state: AgentState, *tracks: Track, **extra: Any) -> dict[str, Any]:
    merged = dict(state["tracks"])
    for track in tracks:
        merged[track["id"]] = track
    return {"tracks": merged, **extra}


def _add(
    track: Track,
    node: str,
    stage: Stage,
    tool: str | None = None,
    call: dict[str, Any] | None = None,
) -> None:
    track["steps"].append(Step(node=node, stage=stage, tool=tool, gates=None, call=call))


async def _decide(
    ctx: AgentContext,
    state: AgentState,
    decision: str,
    schema: type[BaseModel],
    context: dict[str, Any],
) -> Any:
    """One LLM decision. Output that fails validation is retried once (plan.md §8.2)."""
    meta = {"run_id": state["run_id"], "decision": decision}
    try:
        return await ctx.llm.structured(
            decision, schema, system=PROMPTS[decision], context=context, meta=meta
        )
    except ValidationError as exc:
        retry = {**context, "previous_attempt_error": str(exc)}
        return await ctx.llm.structured(
            decision, schema, system=PROMPTS[decision], context=retry, meta=meta
        )


def _llm(ctx: AgentContext, decision: str, context: dict[str, Any], out: BaseModel) -> LLMRecord:
    return llm_record(ctx.llm.model_id, decision, context, out)


def _queue_review(
    ctx: AgentContext,
    state: AgentState,
    node_ids: list[str],
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The critic reviews every decision; in approval mode a human does too."""
    for node_id in node_ids:
        ctx.checks.submit_critic(node_id, facts)
    update: dict[str, Any] = {"awaiting_critic": [*state["awaiting_critic"], *node_ids]}
    if state["mode"] == "approval":
        update["approval"] = [*state["approval"], *node_ids]
    return update


def _replaces(track: Track) -> str | None:
    """The node the next decision replaces (for its `revises` edge); clears the feedback."""
    old, track["revising"], track["feedback"] = track["revising"], None, None
    return old


def _summary(graph: ResearchGraph, step: Step) -> dict[str, Any]:
    out = graph.nodes[step["node"]].tool.output
    extra = {k: v for k, v in out["extra"].items() if not isinstance(v, list | dict)}
    return {
        "node_id": step["node"],
        "tool": out["tool"],
        "status": out["status"],
        "reason": out["reason"],
        "n": out["n"],
        "effect": out["effect"],
        "ci": [out["ci_low"], out["ci_high"]],
        "p_value": out["p_value"],
        "gates": step["gates"],
        **extra,
    }


def _frozen_spec(track: Track) -> dict[str, Any]:
    """The spec the primary test ran with; every follow-up reuses it unchanged."""
    return latest(track, "primary")["call"]["args"]["spec"]


def _slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "hypothesis"
    slug, n = base, 2
    while slug in taken:
        slug, n = f"{base}_{n}", n + 1
    return slug


def _result_title(result: ToolResult) -> str:
    if result.status == "error":
        return f"{result.tool}: error ({result.reason})"
    if result.effect is None:
        return f"{result.tool}: n={result.n}"
    ci = f" [{result.ci_low:+.2f}, {result.ci_high:+.2f}]" if result.ci_low is not None else ""
    return f"{result.tool}: effect {result.effect:+.2f}{ci}, n={result.n}"


# -- nodes -----------------------------------------------------------------------------------


async def intake(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    _, rec, _ = _deps(runtime)
    rec.status("running")
    return {"question_node": rec.node("question", state["question"], done=True)}


async def premise(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D1: restate the question as a measurable claim (plan.md §8.3)."""
    ctx, rec, _ = _deps(runtime)
    track = _current(state)
    context = {"question": state["question"], "feedback": track["feedback"]}
    out: ClaimDecision = await _decide(ctx, state, "premise", ClaimDecision, context)

    title = out.claim.statement if out.claim else "Untestable: " + "; ".join(out.missing_data)
    node = rec.node(
        "premise_check",
        title,
        parents=[state["question_node"]],
        revises=_replaces(track),
        llm=_llm(ctx, "premise", context, out),
    )
    track["spec_node"] = node
    track["spec"] = out.claim.model_dump(mode="json") if out.claim else None
    _add(track, node, "spec")
    extra: dict[str, Any] = {}
    if out.testable:
        track["stage"] = "primary"
    else:
        track.update(stage="verdict", forced_label=UNTESTABLE)
        extra["outcome"] = "untestable"
    return _put(state, track, **extra, **_queue_review(ctx, state, [node]))


async def hypothesize(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D2: competing explanations for a premise that passed, or one revised after critique."""
    ctx, rec, graph = _deps(runtime)
    premise_track = state["tracks"][PREMISE]
    claim = premise_track["spec"]

    if state["current"] is not None:
        track = _current(state)
        context = {
            "question": state["question"],
            "claim": claim,
            "previous": track["spec"],
            "tools": PRIMARY_TOOL_HELP,
            "feedback": track["feedback"],
        }
        out: HypothesisRevision = await _decide(
            ctx, state, "revise_hypothesis", HypothesisRevision, context
        )
        spec = out.hypothesis.model_copy(update={"id": track["id"], "kind": track["kind"]})
        node = rec.node(
            "hypothesis",
            spec.statement,
            parents=[premise_track["spec_node"]],
            revises=_replaces(track),
            llm=_llm(ctx, "revise_hypothesis", context, out),
        )
        track["spec_node"] = node
        track["spec"] = spec.model_dump(mode="json")
        _add(track, node, "spec")
        track["stage"] = "verdict" if track["kind"] == "chance" else "plan"
        return _put(state, track, **_queue_review(ctx, state, [node]))

    context = {
        "question": state["question"],
        "claim": claim,
        "premise_result": _summary(graph, latest(premise_track, "primary")),
        "tools": PRIMARY_TOOL_HELP,
    }
    out_set: HypothesisSet = await _decide(ctx, state, "hypothesize", HypothesisSet, context)
    taken, tracks, nodes = set(state["tracks"]), [], []
    for proposed in out_set.hypotheses:
        spec = proposed.model_copy(update={"id": _slug(proposed.id, taken)})
        taken.add(spec.id)
        node = rec.node(
            "hypothesis",
            spec.statement,
            parents=[premise_track["spec_node"]],
            llm=_llm(ctx, "hypothesize", context, out_set),
        )
        track = new_track(spec.id, spec.kind, "verdict" if spec.kind == "chance" else "plan")
        track["spec_node"] = node
        track["spec"] = spec.model_dump(mode="json")
        _add(track, node, "spec")
        tracks.append(track)
        nodes.append(node)
    order = [t["id"] for t in tracks if t["kind"] != "chance"]
    order += [t["id"] for t in tracks if t["kind"] == "chance"]
    return _put(state, *tracks, order=order, **_queue_review(ctx, state, nodes))


async def plan_test(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D3: pick the primary test. Its args freeze the spec used for every later test."""
    ctx, rec, _ = _deps(runtime)
    track = _current(state)
    spec = track["spec"]
    context = {"hypothesis": spec, "tools": PRIMARY_TOOL_HELP, "feedback": track["feedback"]}
    out: PlanDecision = await _decide(ctx, state, "plan_test", PlanDecision, context)

    frozen = {**spec, "primary_test": out.tool, "controls": list(out.controls)}
    node = rec.node(
        "test_plan",
        f"{out.tool} on {spec['metric']}: {spec['statement']}",
        parents=[track["spec_node"]],
        revises=_replaces(track),
        llm=_llm(ctx, "plan_test", context, out),
    )
    _add(track, node, "plan", out.tool, {"tool": out.tool, "args": {"spec": frozen}})
    track["stage"] = "primary"
    return _put(state, track, **_queue_review(ctx, state, [node]))


async def choose_test(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D3b: the latest result passed its gates; pick the next follow-up or conclude."""
    ctx, rec, graph = _deps(runtime)
    track = _current(state)
    frozen = _frozen_spec(track)
    menu = follow_up_menu(track)
    context = {
        "hypothesis": frozen,
        "tests": [_summary(graph, step) for step in result_steps(track)],
        "gates": track_gates(track),
        "menu": {action: MENU_HELP[action] for action in menu},
        "feedback": track["feedback"],
    }
    out: NextTest = await _decide(ctx, state, "choose_test", next_test_schema(menu), context)

    call = None
    if out.action == "conclude":
        track["stage"] = "interpret"
    else:
        primary = graph.nodes[latest(track, "primary")["node"]]
        test_id = primary.tool.output["extra"].get("test_id")
        call = {"tool": out.action, "args": _follow_up_args(out, frozen, test_id, track["id"])}
        track["stage"] = "follow_up"
    node = rec.node(
        "test_plan",
        f"{out.action}: {out.rationale}",
        parents=[result_steps(track)[-1]["node"]],
        revises=_replaces(track),
        llm=_llm(ctx, "choose_test", context, out),
    )
    _add(track, node, "choose", out.action, call)
    return _put(state, track, **_queue_review(ctx, state, [node]))


def _follow_up_args(
    out: NextTest, frozen: dict[str, Any], test_id: str | None, track_id: str
) -> dict[str, Any]:
    """Arguments in the shapes of Person A's tool schemas (`backend/tools/schemas.py`)."""
    if out.action == "out_of_sample":
        return {"test_id": test_id}
    if out.action == "market_test":
        side = "opp" if frozen["direction"] == "a_lt_b" else "team"
        return {"slice": frozen["group_a"], "side": side}
    if out.action == "placebo":
        return {"test_id": test_id, "placebo_slice": out.placebo_slice.model_dump(mode="json")}
    return {"hypothesis_id": track_id}  # holdout_confirm


async def execute_test(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Run the tool the track's latest plan or choice asked for. No LLM; the gate runs async."""
    ctx, rec, _ = _deps(runtime)
    track = _current(state)
    stage = track["stage"]

    if track["id"] == PREMISE:
        spec = track["spec"]
        call, parent, frozen = (
            {"tool": "compare_groups", "args": {"spec": spec}},
            track["spec_node"],
            spec,
        )
    else:
        decision = latest(track, "plan" if stage == "primary" else "choose")
        call, parent = decision["call"], decision["node"]
        frozen = call["args"]["spec"] if stage == "primary" else _frozen_spec(track)
    tool = call["tool"]

    print(f"CALLING TOOL {tool.upper()} FOR {track['id'].upper()}")  # DEBUG
    started = time.perf_counter()
    result = await asyncio.to_thread(ctx.tools.run, tool, call["args"])
    elapsed_ms = (time.perf_counter() - started) * 1000
    node = rec.node(
        "test_result",
        _result_title(result),
        parents=[parent],
        actor="tool",
        tool=tool_record(result, elapsed_ms),
    )
    ctx.checks.submit_gate(node, result, frozen)

    _add(track, node, stage, tool, call)
    track["stage"] = "gated" if track["id"] == PREMISE else "choose"
    if tool == "holdout_confirm":
        track["holdout_used"] = True
    budget = state["budget"]
    if tool in LEDGER_TOOLS:
        budget = {**budget, "tests": budget["tests"] + 1}
    return _put(state, track, budget=budget, awaiting_gate=[*state["awaiting_gate"], node])


async def interpret(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D4: read the gated results for one hypothesis, citing gate ids."""
    ctx, rec, graph = _deps(runtime)
    track = _current(state)
    spec = _frozen_spec(track)
    results = result_steps(track)
    failed = stopping_failures(track)
    choice = latest(track, "choose")
    if failed:
        ended = f"a gate failed: {', '.join(failed)}"
    elif choice and choice["tool"] == "conclude":
        ended = "the orchestrator chose to conclude"
    else:
        ended = "no follow-up tests were left to run"
    context = {
        "hypothesis": {k: spec[k] for k in ("statement", "kind", "metric", "direction")},
        "tests": [_summary(graph, step) for step in results],
        "gates": track_gates(track),
        "testing_ended_because": ended,
        "feedback": track["feedback"],
    }
    out: Interpretation = await _decide(ctx, state, "interpret", Interpretation, context)
    node = rec.node(
        "interpretation",
        out.summary,
        parents=[step["node"] for step in results],
        revises=_replaces(track),
        llm=_llm(ctx, "interpret", context, out),
    )
    _add(track, node, "interpret")
    track["stage"] = "verdict"
    return _put(state, track, **_queue_review(ctx, state, [node]))


def _gate_label(ctx: AgentContext, track: Track, tracks: dict[str, Track]) -> str:
    """The gate's label for a track; a chance track is judged on all the others."""
    if track["kind"] == "chance":
        competing = [(t["kind"], track_gates(t)) for t in tracks.values() if t["kind"] != "chance"]
        return ctx.gate.verdict("chance", {}, competing)
    return ctx.gate.verdict(track["kind"], track_gates(track), [])


def _verdict_parent(track: Track) -> str:
    step = latest(track, "interpret") or latest(track, "primary")
    return step["node"] if step else track["spec_node"]


async def verdict(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The label comes from the gate (or a rule); D6 only narrates it."""
    ctx, rec, graph = _deps(runtime)
    track = _current(state)
    spec_node = graph.nodes[track["spec_node"]]
    gates = track_gates(track)

    label = track["forced_label"] or _gate_label(ctx, track, state["tracks"])
    contract = contract_verdict(label, gates)

    interpretation = latest(track, "interpret")
    primary = latest(track, "primary")
    parent = _verdict_parent(track)
    title = f"{label}: {spec_node.title}"

    review: dict[str, Any] = {}
    if state["exhausted"] or label in _SILENT_LABELS:
        # Label set by rule; no narrative (budget or revision limit reached).
        node = rec.node(
            "verdict",
            title,
            parents=[parent],
            revises=_replaces(track),
            actor="gate",
            done=True,
            verdict=contract,
        )
    else:
        context = {
            "question": state["question"],
            "hypothesis": spec_node.title,
            "kind": track["kind"],
            "verdict": label,
            "gates": gates,
            "missing_data": spec_node.llm.parsed_output.get("missing_data")
            if spec_node.llm
            else None,
            "interpretation": graph.nodes[interpretation["node"]].llm.parsed_output
            if interpretation
            else None,
            "feedback": track["feedback"],
        }
        out: Narrative = await _decide(ctx, state, "narrate_verdict", Narrative, context)
        node = rec.node(
            "verdict",
            title,
            parents=[parent],
            revises=_replaces(track),
            llm=_llm(ctx, "narrate_verdict", context, out),
            verdict=contract,
        )
        _add(track, node, "verdict")
        review = _queue_review(ctx, state, [node], facts={"gates": gates})
    if contract is not None:
        rec.verdict(track["spec_node"], contract)

    if primary and track["id"] != PREMISE:
        held = all(gates.get(g) == "pass" for g in ("G2", "G3"))
        rec.edge(primary["node"], track["spec_node"], "supports" if held else "refutes")
    track.update(stage="done", verdict=label, verdict_node=node)
    return _put(state, track, **review)


def _final_g4(ctx: AgentContext, rec: Recorder, state: AgentState) -> dict[str, Track]:
    """plan.md §9.1 G4: BH over the final ledger, before the report. Each flip is logged as a
    check; a verdict it changes is revised by the gate (no narrative, like a budget label)."""
    graph = rec.graph
    tracks = copy.deepcopy(state["tracks"])
    changed: set[str] = set()
    for check in ctx.gate.reevaluate():
        rec.check(from_gate(check))
        for track in tracks.values():
            step = next((s for s in result_steps(track) if s["node"] == check.node_id), None)
            if step is None:
                continue  # superseded by a revision: logged, but it steers nothing
            step["gates"] = {**(step["gates"] or {}), "G4": check.status}
            held = all(status == "pass" for status in step["gates"].values())
            rec.mark(step["node"], "passed" if held else "failed", actor="gate")
            changed.add(track["id"])
    if changed:
        changed |= {tid for tid, t in tracks.items() if t["kind"] == "chance"}
    for track_id in sorted(changed):
        track = tracks[track_id]
        if track["verdict"] is None or track["forced_label"]:
            continue  # no verdict yet (a premise that held), or a label set by rule
        label = _gate_label(ctx, track, tracks)
        if label == track["verdict"]:
            continue
        contract = contract_verdict(label, track_gates(track))
        node = rec.node(
            "verdict",
            f"{label}: {graph.nodes[track['spec_node']].title}",
            parents=[_verdict_parent(track)],
            revises=track["verdict_node"],
            actor="gate",
            done=True,
            verdict=contract,
        )
        if contract is not None:
            rec.verdict(track["spec_node"], contract)
        track.update(verdict=label, verdict_node=node)
    return tracks


async def report(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D6: answer the question from the verdict table. Runs once every check has landed."""
    ctx, rec, graph = _deps(runtime)
    tracks = _final_g4(ctx, rec, state)
    rows = []
    for track_id in [PREMISE, *state["order"]]:
        track = tracks[track_id]
        if track["verdict"] is None:
            continue  # a premise that held has no verdict of its own
        rows.append(
            {
                "track": track_id,
                "kind": track["kind"],
                "statement": graph.nodes[track["spec_node"]].title,
                "verdict": track["verdict"],
                "verdict_node": track["verdict_node"],
                "path": [track["verdict_node"], *graph.ancestors(track["verdict_node"])],
            }
        )
    context = {
        "question": state["question"],
        "outcome": state["outcome"],
        "verdicts": [{k: row[k] for k in ("statement", "kind", "verdict")} for row in rows],
    }
    out: ReportDraft = await _decide(ctx, state, "report", ReportDraft, context)
    budget = {**state["budget"], "llm_calls": ctx.llm.calls}
    result = {
        "question": state["question"],
        "outcome": state["outcome"],
        "answer": out.answer,
        "verdicts": rows,
        "budget": budget,
    }
    rec.budget(budget["llm_calls"], budget["tests"])
    rec.status("completed")
    return {"report": result, "status": "completed", "budget": budget, "tracks": tracks}


async def wait_checks(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    await runtime.context.checks.wait(state["waiting_on"])
    return {"waiting_on": []}


class ReviewAnswer(BaseModel):
    action: Literal["approve", "reject"]
    note: str | None = None

    @model_validator(mode="after")
    def _note_on_reject(self) -> Self:
        if self.action == "reject" and len((self.note or "").strip()) < 10:
            raise ValueError("a rejection needs a note of at least 10 characters")
        return self


async def human_review(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Approval mode: pause until a human approves or rejects the queued decision."""
    _, rec, graph = _deps(runtime)
    node_id = state["approval"][0]
    if "awaiting_approval" not in graph.nodes[node_id].flags:  # re-runs from the top on resume
        rec.mark(node_id, "awaiting_approval")
        rec.status("awaiting_approval")
    node = graph.nodes[node_id]
    request = {
        "node_id": node_id,
        "type": node.type,
        "title": node.title,
        "checks": [check.model_dump(mode="json") for check in node.checks],
    }
    while True:
        try:
            answer = ReviewAnswer.model_validate(interrupt(request))
            break
        except ValidationError as exc:
            request = {**request, "error": str(exc)}

    approved = answer.action == "approve"
    if approved:
        rec.intervention(
            Approve(type="approve", node_id=node_id, payload=OptionalNote(note=answer.note))
        )
    else:
        rec.intervention(
            Reject(type="reject", node_id=node_id, payload=RequiredNote(note=answer.note))
        )
    rec.check(
        CheckResult(
            node_id=node_id,
            checker="human",
            status="pass" if approved else "fail",
            rationale=answer.note or "Approved.",
        )
    )
    if answer.note:
        rec.annotate(node_id, answer.note)
    rec.status("running")
    remaining = {**state, "approval": state["approval"][1:]}
    if approved:
        rec.mark(node_id, "passed", actor="human")
        return {"approval": remaining["approval"]}
    draft = Draft(remaining)
    draft.review(node_id, "fail", f"Human reviewer: {answer.note}", graph)
    for nid, node_state in draft.node_states:
        rec.mark(nid, node_state)
    return draft.changes()


SPECIALISTS = {
    "premise": premise,
    "hypothesize": hypothesize,
    "plan_test": plan_test,
    "execute_test": execute_test,
    "choose_test": choose_test,
    "interpret": interpret,
    "verdict": verdict,
    "report": report,
    "wait_checks": wait_checks,
    "human_review": human_review,
}
