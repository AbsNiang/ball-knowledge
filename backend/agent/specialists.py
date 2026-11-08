"""The research loop's steps. The router dispatches one per turn and each returns to the router.

LLM roles:
- `premise` (D1): restate the question as a measurable claim.
- `seed`: one grounded web search, then up to three seeds (ideas that would answer the question).
- `research`: for the active seed, the researcher calls one toolkit tool (`toolkit.py`).
- `judge`: reviews the step that just ran and updates the seed's running synthesis.
- `supervise`: continue the seed with guidance for the researcher, or conclude it.
- `summarize`: aggregate every seed into the answer.
`premise_test`, `execute` and `conclude` run tools and rules only. Labels come from the gate.
"""

import asyncio
import copy
import re
import time
from typing import Any, Literal, Self

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError, model_validator

from backend.agent.context import AgentContext
from backend.agent.judge import audit, known_ids, premise_context, render, seed_context, to_check
from backend.agent.ports import CheckResult, ToolResult, from_gate
from backend.agent.prompts import DATASET, PROMPTS
from backend.agent.record import Recorder, contract_verdict, llm_record, tool_record
from backend.agent.schemas import Aggregate, ClaimDecision, Judgment, SeedSet, Supervision
from backend.agent.state import (
    CONFIRMATORY,
    INCONCLUSIVE,
    MAX_PREMISE_REVISIONS,
    MAX_STEPS_PER_SEED,
    NOT_TESTED,
    PREMISE,
    PREMISE_FALSE,
    REPORT_RESERVE,
    REVIEW_LABEL,
    UNTESTABLE,
    AgentState,
    GateStatus,
    Premise,
    Seed,
    Step,
    new_seed,
    stopping_failures,
)
from backend.agent.toolkit import (
    KIT,
    LEDGER_TOOLS,
    Prepared,
    offered,
    prepare,
    step_summary,
    tool_specs,
)
from backend.events.projection import ResearchGraph
from backend.gate.config import SEED as TOOL_SEED
from backend.llm.gateway import NoToolCall, SearchResult, ToolCall
from backend.tools.hashing import hash_args
from contracts.api import Approve, OptionalNote, Reject, RequiredNote
from contracts.models import LLMRecord, ToolRecord

_NODE_STATE = {"pass": "passed", "flag": "flagged", "fail": "failed"}


# -- helpers ---------------------------------------------------------------------------------


def _deps(runtime: Runtime[AgentContext]) -> tuple[AgentContext, Recorder, ResearchGraph]:
    ctx = runtime.context
    return ctx, Recorder(ctx.log), ctx.log.graph


def _meta(state: AgentState, decision: str) -> dict[str, Any]:
    return {"run_id": state["run_id"], "decision": decision, "seed": state["current"]}


async def _decide(
    ctx: AgentContext,
    state: AgentState,
    decision: str,
    schema: type[BaseModel],
    context: dict[str, Any],
) -> Any:
    """One structured decision. Output that fails validation is retried once (plan.md §8.2)."""
    prompt, meta = PROMPTS[decision], _meta(state, decision)
    try:
        return await ctx.llm.structured(decision, schema, system=prompt, context=context, meta=meta)
    except ValidationError as exc:
        retry = {**context, "previous_attempt_error": str(exc)}
        return await ctx.llm.structured(decision, schema, system=prompt, context=retry, meta=meta)


async def _choose(
    ctx: AgentContext, state: AgentState, names: list[str], context: dict[str, Any]
) -> ToolCall:
    """The researcher's tool call. A reply with no valid call is retried once."""
    tools, meta = tool_specs(names), _meta(state, "research")
    prompt = PROMPTS["research"]
    try:
        return await ctx.llm.call_tool("research", tools, system=prompt, context=context, meta=meta)
    except (ValidationError, NoToolCall) as exc:
        retry = {**context, "previous_attempt_error": str(exc)}
        return await ctx.llm.call_tool("research", tools, system=prompt, context=retry, meta=meta)


def _llm(ctx: AgentContext, decision: str, context: dict[str, Any], out: BaseModel, **kw) -> Any:
    return llm_record(ctx.llm.model_id, decision, context, out, **kw)


def _active(state: AgentState) -> Seed:
    return copy.deepcopy(state["seeds"][state["current"]])


def _put(state: AgentState, seed: Seed, **extra: Any) -> dict[str, Any]:
    return {"seeds": {**state["seeds"], seed["id"]: seed}, **extra}


def _approve(state: AgentState, node_ids: list[str]) -> dict[str, Any]:
    """In approval mode a human reviews each decision before anything runs after it."""
    if state["mode"] != "approval":
        return {}
    return {"approval": [*state["approval"], *node_ids]}


def _slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "seed"
    slug, n = base, 2
    while slug in taken or slug == PREMISE:
        slug, n = f"{base}_{n}", n + 1
    return slug


def _result_title(result: ToolResult) -> str:
    if result.status == "error":
        return f"{result.tool}: error ({result.reason})"
    if result.effect is None:
        return f"{result.tool}: n={result.n}"
    ci = f" [{result.ci_low:+.2f}, {result.ci_high:+.2f}]" if result.ci_low is not None else ""
    return f"{result.tool}: effect {result.effect:+.2f}{ci}, n={result.n}"


def _last_node(seed: Seed) -> str:
    """The seed's latest judgment: each step builds on the synthesis before it. Steps that
    produced no evidence (errors, rejections) hang off it as side branches."""
    judged = [step["judgment"] for step in seed["steps"] if step["judgment"]]
    return judged[-1] if judged else seed["node"]


async def _run(ctx: AgentContext, tool: str, args: dict[str, Any]) -> tuple[ToolResult, float]:
    started = time.perf_counter()
    result = await asyncio.to_thread(ctx.tools.run, tool, args)
    return result, (time.perf_counter() - started) * 1000


async def _gate(
    ctx: AgentContext, rec: Recorder, node_id: str, result: ToolResult, spec: dict[str, Any]
) -> dict[str, GateStatus] | None:
    """Run the gate on a result and record its checks. A gate error fails closed."""
    try:
        checks = await asyncio.to_thread(ctx.gate.evaluate, node_id, result, spec)
        results = [from_gate(check) for check in checks]
    except Exception as exc:  # noqa: BLE001 - fail closed and keep the run alive
        results = [
            CheckResult(
                node_id=node_id,
                checker="gate",
                gate_id="error",
                status="fail",
                rationale=f"Gate error: {exc!r}",
            )
        ]
    for check in results:
        rec.check(check)
    failed = result.status != "ok" or any(check.status != "pass" for check in results)
    rec.mark(node_id, "failed" if failed else "passed", actor="gate")
    if not results:
        return None
    return {c.gate_id: "pass" if c.status == "pass" else "fail" for c in results if c.gate_id}


def _tests_spent(budget: dict[str, int], prepared: Prepared, result: ToolResult) -> dict:
    """Ledger entries this call added: the test itself, plus G6's spread-controlled regression."""
    spent = int(result.tool in LEDGER_TOOLS and result.status == "ok")
    spent += int(prepared.role == "market" and result.status == "ok")
    return {**budget, "tests": budget["tests"] + spent}


def _error_result(tool: str, args: dict[str, Any], reason: str, **extra: Any) -> ToolResult:
    return ToolResult(
        tool=tool, args=args, status="error", reason=reason, args_hash=hash_args(args), extra=extra
    )


def _source_node(
    ctx: AgentContext,
    rec: Recorder,
    found: SearchResult,
    parents: list[str],
    elapsed_ms: float,
    error: str | None = None,
) -> str:
    """A web search as evidence: what the sources say, with links. Never statistical proof."""
    args = {"query": found.query}
    output = {
        "tool": "web_search",
        "status": "error" if error else "ok",
        "reason": error,
        "model_id": ctx.llm.model_id,
        **found.model_dump(mode="json"),
    }
    links = "\n".join(f"- [{hit.title}]({hit.url})" for hit in found.sources)
    text = found.summary or f"The search failed: {error}"
    node = rec.node(
        "source",
        f"Search: {found.query}",
        parents=parents,
        actor="tool",
        done=error is None,
        llm=LLMRecord(
            model_id=ctx.llm.model_id,
            prompt=found.query,
            raw_response=found.summary,
            parsed_output=found.model_dump(mode="json"),
            rationale=f"{text}\n\n**Sources**\n{links}" if links else text,
        ),
        tool=ToolRecord(
            tool_name="web_search",
            args=args,
            args_hash=hash_args(args),
            output=output,
            data_slice_hash="",
            seed=TOOL_SEED,
            duration_ms=elapsed_ms,
        ),
    )
    if error:
        rec.mark(node, "failed", actor="tool")
    return node


async def _search(
    ctx: AgentContext, rec: Recorder, state: AgentState, query: str, parents: list[str]
) -> tuple[SearchResult, str, str | None]:
    """A failed search becomes a failed source node, not a failed run."""
    started = time.perf_counter()
    try:
        found = await ctx.llm.search(query, system=PROMPTS["search"], meta=_meta(state, "search"))
        error = None
    except Exception as exc:  # noqa: BLE001 - the network is not allowed to end the run
        found, error = SearchResult(query=query, summary=""), repr(exc)
    elapsed = (time.perf_counter() - started) * 1000
    return found, _source_node(ctx, rec, found, parents, elapsed, error), error


def seed_label(ctx: AgentContext, seed: Seed) -> str:
    """The gate's label for a seed, from its confirmatory gates only."""
    if seed["primary"] is None:
        return UNTESTABLE if not seed["testable_with_data"] else NOT_TESTED
    if not seed["gates"]:
        return NOT_TESTED
    return ctx.gate.verdict(seed["kind"], seed["gates"], [])


def _judgment_view(seed: Seed) -> dict[str, Any] | None:
    judged = seed["judgment"]
    if judged is None:
        return None
    keys = ("synthesis", "stance", "confidence", "open_questions", "suggested_next")
    return {k: judged[k] for k in keys}


def _steps(graph: ResearchGraph, seed: Seed) -> list[dict[str, Any]]:
    return [step_summary(graph, step, i) for i, step in enumerate(seed["steps"], 1)]


# -- the premise -----------------------------------------------------------------------------


async def intake(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    _, rec, _ = _deps(runtime)
    rec.status("running")
    return {"question_node": rec.node("question", state["question"], done=True)}


async def premise(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """D1: restate the question as a measurable claim (plan.md §8.3)."""
    ctx, rec, _ = _deps(runtime)
    track = copy.deepcopy(state["premise"])
    context = {"question": state["question"], "dataset": DATASET, "feedback": track["feedback"]}
    out: ClaimDecision = await _decide(ctx, state, "premise", ClaimDecision, context)

    title = out.claim.statement if out.claim else "Untestable: " + "; ".join(out.missing_data)
    node = rec.node(
        "premise_check",
        title,
        parents=[state["question_node"]],
        revises=track["revising"],
        done=not out.testable,  # a testable claim is checked by the judge after its test
        llm=_llm(ctx, "premise", context, out),
    )
    track.update(
        node=node,
        spec=out.claim.model_dump(mode="json") if out.claim else None,
        missing_data=list(out.missing_data),
        result=None,
        gates=None,
        judgment=None,
        feedback=None,
        revising=None,
        label=None if out.testable else UNTESTABLE,
    )
    phase = "premise_test" if out.testable else "summarize"
    outcome = None if out.testable else "untestable"
    return {"premise": track, "phase": phase, "outcome": outcome, **_approve(state, [node])}


async def premise_test(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The claim's test on the explore seasons. No LLM; the gate decides whether it holds."""
    ctx, rec, _ = _deps(runtime)
    track = copy.deepcopy(state["premise"])
    spec = track["spec"]
    result, elapsed = await _run(ctx, "compare_groups", {"spec": spec})
    node = rec.node(
        "test_result",
        _result_title(result),
        parents=[track["node"]],
        actor="tool",
        tool=tool_record(result, elapsed),
    )
    track.update(result=node, gates=await _gate(ctx, rec, node, result, spec) or {})
    spent = int(result.status == "ok")
    budget = {**state["budget"], "tests": state["budget"]["tests"] + spent}
    return {"premise": track, "budget": budget, "phase": "judge"}


def _settle_premise(track: Premise) -> dict[str, Any]:
    """plan.md §8.3: G1 fails -> Inconclusive; G2 or G3 fails -> Premise false; else seed."""
    gates = track["gates"] or {}
    if gates.get("G1") != "pass":
        track["label"] = INCONCLUSIVE
        return {"premise": track, "outcome": "inconclusive", "phase": "summarize"}
    if gates.get("G2") != "pass" or gates.get("G3") != "pass":
        track["label"] = PREMISE_FALSE
        return {"premise": track, "outcome": "premise_false", "phase": "summarize"}
    return {"premise": track, "phase": "seed"}


async def _judge_premise(
    ctx: AgentContext, rec: Recorder, graph: ResearchGraph, state: AgentState
) -> dict[str, Any]:
    track = copy.deepcopy(state["premise"])
    context = premise_context(graph, state["question"], track)
    out: Judgment = await _decide(ctx, state, "judge", Judgment, context)
    judged, _ = audit(out, known_ids(context), track["gates"] or {})
    node = rec.node(
        "judgment",
        f"Judge on the premise: {judged['step_review']}",
        parents=[track["result"]],
        actor="judge",
        done=True,
        llm=_llm(ctx, "judge", context, out, parsed=judged, rationale=render(judged)),
    )
    check = rec.check(to_check(track["node"], judged))
    rec.mark(track["node"], _NODE_STATE[check.status], actor="judge")
    track["judgment"] = node
    if judged["step_review"] != "flawed":
        return _settle_premise(track)
    if track["revisions"] >= MAX_PREMISE_REVISIONS:
        track["label"] = REVIEW_LABEL  # still flawed after a revision: no seed builds on it
        return {"premise": track, "outcome": "review", "phase": "summarize"}
    track.update(
        revisions=track["revisions"] + 1,
        feedback=judged["step_rationale"],
        revising=track["node"],
    )
    return {"premise": track, "phase": "premise"}


# -- seeding ---------------------------------------------------------------------------------


async def seed(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """A grounded web search, then up to three seeds from it and the model's own reasoning."""
    ctx, rec, graph = _deps(runtime)
    track = state["premise"]
    found, source, _ = await _search(ctx, rec, state, state["question"], [track["node"]])
    context = {
        "question": state["question"],
        "claim": track["spec"],
        "premise_result": graph.nodes[track["result"]].title,
        "dataset": DATASET,
        "search": {
            "summary": found.summary,
            "sources": [hit.model_dump() for hit in found.sources],
        },
    }
    out: SeedSet = await _decide(ctx, state, "seed", SeedSet, context)

    urls = {hit.url for hit in found.sources}
    seeds: dict[str, Seed] = {}
    taken: set[str] = set()
    for idea in out.seeds:
        seed_id = _slug(idea.id, taken)
        taken.add(seed_id)
        cited = [url for url in idea.sources if url in urls]
        links = "".join(f"\n- {url}" for url in cited)
        node = rec.node(
            "hypothesis",
            idea.statement,
            parents=[track["node"], source],
            done=True,
            llm=_llm(
                ctx,
                "seed",
                context,
                out,
                rationale=idea.rationale + (f"\n\n**Sources**{links}" if links else ""),
            ),
        )
        seeds[seed_id] = new_seed(
            seed_id,
            node,
            statement=idea.statement,
            kind=idea.kind,
            rationale=idea.rationale,
            testable_with_data=idea.testable_with_data,
            sources=[source] if cited else [],
        )
    order = list(seeds)
    seeds[order[0]]["status"] = "active"
    nodes = [s["node"] for s in seeds.values()]
    return {
        "seeds": seeds,
        "order": order,
        "current": order[0],
        "search_node": source,
        "phase": "research",
        **_approve(state, nodes),
    }


# -- one research step: research -> execute -> judge -> supervise ----------------------------


async def research(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The researcher calls one tool for the active seed."""
    ctx, rec, graph = _deps(runtime)
    track = _active(state)
    names = offered(track)
    supervision = track["supervision"] or {}
    context = {
        "question": state["question"],
        "seed": {
            k: track[k] for k in ("id", "statement", "kind", "rationale", "testable_with_data")
        },
        "primary_test": track["primary"],
        "primary_test_id": track["primary_test_id"],
        "verdict_gates": track["gates"],
        "label_if_concluded_now": seed_label(ctx, track),
        "steps": _steps(graph, track),
        "latest_judgment": _judgment_view(track),
        "supervisor_guidance": track["guidance"],
        "supervisor_rationale": supervision.get("rationale"),
        "feedback": track["feedback"],
        "dataset": DATASET,
        "steps_left": MAX_STEPS_PER_SEED - len(track["steps"]),
    }
    call = await _choose(ctx, state, names, context)
    node = rec.node(
        "test_plan",
        f"{call.tool}: {call.reason or 'no reason given'}",
        parents=[_last_node(track)],
        llm=_llm(ctx, "research", context, call, rationale=call.reason),
    )
    role = KIT[call.tool].role if call.tool in KIT else "data"
    track["steps"].append(
        Step(
            plan=node,
            result=None,
            judgment=None,
            tool=call.tool,
            role=role,
            args=dict(call.args),
            gates=None,
            error=None,
        )
    )
    track.update(guidance=None, feedback=None)
    return _put(state, track, phase="execute", **_approve(state, [node]))


async def execute(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Run the researcher's call through the toolkit's rules. No LLM except a web search."""
    ctx, rec, _ = _deps(runtime)
    track = _active(state)
    step = track["steps"][-1]
    prepared = prepare(track, ToolCall(tool=step["tool"], args=step["args"]))
    if prepared.error:
        problems = {"problems": prepared.problems} if prepared.problems else {}
        failed = _error_result(step["tool"], step["args"], prepared.error, **problems)
        return _put(state, _failed_step(rec, track, step, failed, 0.0), phase="supervise")
    if prepared.role == "search":
        _, node, error = await _search(ctx, rec, state, prepared.args["query"], [step["plan"]])
        step["result"] = node
        if error:
            rec.mark(step["plan"], "failed", actor="tool")
            step["error"] = f"search failed: {error}"
            track["feedback"] = f"Your web_search failed: {error}"
            return _put(state, track, phase="supervise")
        return _put(state, track, phase="judge")
    if prepared.role == "declare":
        return await _declare(ctx, rec, state, track, step, prepared)

    result, elapsed = await _run(ctx, prepared.registry, prepared.args)
    node = rec.node(
        "test_result",
        _result_title(result),
        parents=[step["plan"]],
        actor="tool",
        tool=tool_record(result, elapsed),
    )
    if prepared.gate_spec is None:  # diagnostics and data tools: no gate applies
        rec.mark(node, "passed" if result.status == "ok" else "failed", actor="tool")
        gates = None
    else:
        gates = await _gate(ctx, rec, node, result, prepared.gate_spec)
    step.update(result=node, gates=gates)
    if prepared.explored:
        track["explored"] += 1
    if prepared.role in CONFIRMATORY:
        track["used"].append(prepared.tool)
        track["gates"].update(gates or {})
        if prepared.role == "primary":
            track.update(primary_result=node, primary_test_id=result.extra.get("test_id"))
    budget = _tests_spent(state["budget"], prepared, result)
    if prepared.role == "data":  # group sizes only: nothing to judge, so straight back
        rec.mark(step["plan"], "passed" if result.status == "ok" else "failed", actor="tool")
        if result.status != "ok":
            step["error"] = f"tool error: {result.reason}"
            track["feedback"] = f"Your last call ({step['tool']}) failed: {result.reason}"
        return _put(state, track, budget=budget, phase="research")
    if result.status != "ok":
        rec.mark(step["plan"], "failed", actor="tool")
        step["error"] = f"tool error: {result.reason}"
        track["feedback"] = f"Your last call ({step['tool']}) failed: {result.reason}"
        return _put(state, track, budget=budget, phase="supervise")
    return _put(state, track, budget=budget, phase="judge")


def _failed_step(
    rec: Recorder, track: Seed, step: Step, failed: ToolResult, elapsed: float
) -> Seed:
    """A call that could not run: recorded, and fed back to the researcher's next turn."""
    node = rec.node(
        "test_result",
        _result_title(failed),
        parents=[step["plan"]],
        actor="tool",
        tool=tool_record(failed, elapsed),
    )
    rec.mark(node, "failed", actor="tool")
    rec.mark(step["plan"], "failed", actor="tool")
    detail = "; ".join(f"{p['field']}: {p['problem']}" for p in failed.extra.get("problems", []))
    step.update(result=node, error=failed.reason + (f" ({detail})" if detail else ""))
    track["feedback"] = f"Your last call ({step['tool']}) did not run: {step['error']}"
    return track


async def _declare(
    ctx: AgentContext,
    rec: Recorder,
    state: AgentState,
    track: Seed,
    step: Step,
    prepared: Prepared,
) -> dict[str, Any]:
    """Freeze the primary test, once both groups are known to exist. Sizes only: no outcomes."""
    spec = prepared.spec
    sizes: dict[str, int] = {}
    problems = []
    started = time.perf_counter()
    for name, group in (("n_a", spec["group_a"]), ("n_b", spec["group_b"])):
        if group == "complement":
            continue
        result, _ = await _run(ctx, "define_slice", {"spec": group})
        if result.status != "ok" or not result.n:
            problems.append(f"{name}: {result.reason or 'the slice is empty'}")
        else:
            sizes[name] = result.n
    elapsed = (time.perf_counter() - started) * 1000
    if problems:
        failed = _error_result("declare_primary_test", {"spec": spec}, "; ".join(problems))
        return _put(state, _failed_step(rec, track, step, failed, elapsed), phase="supervise")
    declared = ToolResult(
        tool="declare_primary_test",
        args={"spec": spec},
        args_hash=hash_args({"spec": spec}),
        n=sizes.get("n_a"),
        extra={"spec": spec, **sizes},
    )
    node = rec.node(
        "test_result",
        f"Declared: {spec['statement']}",
        parents=[step["plan"]],
        actor="tool",
        done=True,
        tool=tool_record(declared, elapsed),
    )
    step["result"] = node
    track.update(primary=spec)
    track["used"].append("declare_primary_test")
    return _put(state, track, phase="judge")


async def judge(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Review the step that just ran and update the running synthesis (the premise first)."""
    ctx, rec, graph = _deps(runtime)
    if state["current"] is None:
        return await _judge_premise(ctx, rec, graph, state)
    track = _active(state)
    step = track["steps"][-1]
    context = seed_context(graph, state["question"], track, step, track["sources"])
    out: Judgment = await _decide(ctx, state, "judge", Judgment, context)
    judged, _ = audit(out, known_ids(context), track["gates"])
    node = rec.node(
        "judgment",
        f"Judge: {judged['stance']}, {judged['confidence']} confidence",
        parents=[step["result"]],
        actor="judge",
        done=True,
        llm=_llm(ctx, "judge", context, out, parsed=judged, rationale=render(judged)),
    )
    check = rec.check(to_check(step["plan"], judged))
    rec.mark(step["plan"], _NODE_STATE[check.status], actor="judge")
    step["judgment"] = node
    track["judgment"] = judged
    return _put(state, track, phase="supervise")


async def supervise(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Continue the seed with guidance, or conclude it. The router enforces the step limit."""
    ctx, _, graph = _deps(runtime)
    track = _active(state)
    left = state["budget"]["max_llm_calls"] - ctx.llm.calls - REPORT_RESERVE
    context = {
        "question": state["question"],
        "seed": {k: track[k] for k in ("id", "statement", "kind")},
        "primary_test": track["primary"],
        "verdict_gates": track["gates"],
        "label_if_concluded_now": seed_label(ctx, track),
        "label_fixed": bool(stopping_failures(track["gates"])),
        "steps": _steps(graph, track),
        "latest_judgment": _judgment_view(track),
        "feedback": track["feedback"],
        "tools_offered_next": offered(track),
        "steps_left": MAX_STEPS_PER_SEED - len(track["steps"]),
        "llm_calls_left": left,
        "seeds_after_this": sum(state["seeds"][s]["status"] == "queued" for s in state["order"]),
    }
    out: Supervision = await _decide(ctx, state, "supervise", Supervision, context)
    track["supervision"] = {
        **out.model_dump(mode="json"),
        "llm": _llm(ctx, "supervise", context, out).model_dump(mode="json"),
    }
    if out.action == "continue":
        track["guidance"] = out.guidance
        return _put(state, track, phase="research")
    return _put(state, track, phase="conclude")


# -- concluding ------------------------------------------------------------------------------


def _conclude_seed(ctx: AgentContext, rec: Recorder, track: Seed) -> None:
    label = track["forced_label"] or seed_label(ctx, track)
    contract = contract_verdict(label, track["gates"])
    supervision = track["supervision"]
    concluded_by = supervision if supervision and supervision["action"] == "conclude" else None
    node = rec.node(
        "verdict",
        f"{label}: {track['statement']}",
        parents=[_last_node(track)],
        actor="gate",
        done=True,
        llm=LLMRecord.model_validate(concluded_by["llm"]) if concluded_by else None,
        verdict=contract,
    )
    if contract is not None:
        rec.verdict(track["node"], contract)
    if track["primary_result"]:
        held = all(track["gates"].get(g) == "pass" for g in ("G2", "G3"))
        rec.edge(track["primary_result"], track["node"], "supports" if held else "refutes")
    track.update(status="done", verdict=label, verdict_node=node)


async def conclude(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The active seed's verdict, by rule; then the next seed, or the summary."""
    ctx, rec, _ = _deps(runtime)
    seeds = copy.deepcopy(state["seeds"])
    ending = [state["current"]] if state["current"] else []
    if state["exhausted"]:
        ending += [sid for sid in state["order"] if seeds[sid]["status"] == "queued"]
    for seed_id in ending:
        if seeds[seed_id]["status"] in ("active", "queued"):
            _conclude_seed(ctx, rec, seeds[seed_id])
    upcoming = next((sid for sid in state["order"] if seeds[sid]["status"] == "queued"), None)
    if upcoming is None:
        return {"seeds": seeds, "current": None, "phase": "summarize"}
    seeds[upcoming]["status"] = "active"
    return {"seeds": seeds, "current": upcoming, "phase": "research"}


def _final_g4(
    ctx: AgentContext, rec: Recorder, state: AgentState
) -> tuple[dict[str, Seed], Premise]:
    """plan.md §9.1 G4: BH over the final ledger, before the summary. Each flip is logged as a
    check; a seed label it changes gets a revised verdict node (by rule, like a budget label)."""
    graph = rec.graph
    seeds = copy.deepcopy(state["seeds"])
    track = copy.deepcopy(state["premise"])
    changed: set[str] = set()
    for check in ctx.gate.reevaluate():
        rec.check(from_gate(check))
        if check.node_id == track["result"] and track["gates"] is not None:
            track["gates"]["G4"] = check.status  # logged; the premise was settled on G1-G3
            held = all(status == "pass" for status in track["gates"].values())
            rec.mark(check.node_id, "passed" if held else "failed", actor="gate")
        for seed_track in seeds.values():
            step = next((s for s in seed_track["steps"] if s["result"] == check.node_id), None)
            if step is None:
                continue
            step["gates"] = {**(step["gates"] or {}), "G4": check.status}
            held = all(status == "pass" for status in step["gates"].values())
            rec.mark(check.node_id, "passed" if held else "failed", actor="gate")
            if seed_track["primary_result"] == check.node_id:
                seed_track["gates"]["G4"] = check.status
                changed.add(seed_track["id"])
    for seed_id in sorted(changed):
        seed_track = seeds[seed_id]
        if seed_track["verdict"] is None or seed_track["forced_label"]:
            continue
        label = seed_label(ctx, seed_track)
        if label == seed_track["verdict"]:
            continue
        contract = contract_verdict(label, seed_track["gates"])
        node = rec.node(
            "verdict",
            f"{label}: {seed_track['statement']}",
            parents=graph.parents(seed_track["verdict_node"]),
            revises=seed_track["verdict_node"],
            actor="gate",
            done=True,
            verdict=contract,
        )
        if contract is not None:
            rec.verdict(seed_track["node"], contract)
        seed_track.update(verdict=label, verdict_node=node)
    return seeds, track


def _premise_verdict(rec: Recorder, state: AgentState, track: Premise) -> dict[str, Any]:
    """The single row of a run that ended at the premise."""
    label = track["label"]
    anchor = track["node"] or state["question_node"]
    parent = track["judgment"] or track["result"] or anchor
    contract = contract_verdict(label, track["gates"] or {})
    title = track["spec"]["statement"] if track["spec"] else rec.graph.nodes[anchor].title
    if track["missing_data"]:
        title = f"{title} (missing: {', '.join(track['missing_data'])})"
    node = rec.node(
        "verdict", f"{label}: {title}", parents=[parent], actor="gate", done=True, verdict=contract
    )
    if contract is not None:
        rec.verdict(anchor, contract)
    track["verdict_node"] = node
    return {
        "track": PREMISE,
        "kind": "premise",
        "statement": title,
        "verdict": label,
        "verdict_node": node,
        "hypothesis_node": anchor,
        "stance": None,
        "confidence": None,
    }


async def summarize(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Aggregate every seed's outcome into the answer to the question."""
    ctx, rec, graph = _deps(runtime)
    seeds, track = _final_g4(ctx, rec, state)
    if state["outcome"]:
        rows = [_premise_verdict(rec, state, track)]
    else:
        rows = [
            {
                "track": seed_id,
                "kind": seeds[seed_id]["kind"],
                "statement": seeds[seed_id]["statement"],
                "verdict": seeds[seed_id]["verdict"],
                "verdict_node": seeds[seed_id]["verdict_node"],
                "hypothesis_node": seeds[seed_id]["node"],
                "stance": (seeds[seed_id]["judgment"] or {}).get("stance"),
                "confidence": (seeds[seed_id]["judgment"] or {}).get("confidence"),
            }
            for seed_id in state["order"]
            if seeds[seed_id]["status"] == "done"
        ]
    for row in rows:
        row["path"] = [row["verdict_node"], *graph.ancestors(row["verdict_node"])]

    context = {
        "question": state["question"],
        "outcome": state["outcome"],
        "premise": {
            "claim": track["spec"]["statement"] if track["spec"] else None,
            "result": graph.nodes[track["result"]].title if track["result"] else None,
            "missing_data": track["missing_data"],
        },
        "seeds": [
            {
                "seed_id": row["track"],
                "statement": row["statement"],
                "kind": row["kind"],
                "verdict": row["verdict"],
                "judge": _judgment_view(seeds[row["track"]]) if row["track"] in seeds else None,
            }
            for row in rows
        ],
        "pruned_by_reviewer": [
            seeds[s]["statement"] for s in state["order"] if seeds[s]["status"] == "pruned"
        ],
    }
    out: Aggregate = await _decide(ctx, state, "summarize", Aggregate, context)
    summaries = {s.seed_id: s.summary for s in out.seeds}
    for row in rows:
        row["summary"] = summaries.get(row["track"], "")
    answer = out.answer
    node = rec.node(
        "report",
        f"Answer: {state['question']}",
        parents=[row["verdict_node"] for row in rows] or [state["question_node"]],
        done=True,
        llm=_llm(ctx, "summarize", context, out, rationale=_render_report(answer, rows)),
    )
    budget = {**state["budget"], "llm_calls": ctx.llm.calls}
    report = {
        "question": state["question"],
        "outcome": state["outcome"],
        "answer": answer,
        "verdicts": rows,
        "report_node": node,
        "budget": budget,
    }
    rec.budget(budget["llm_calls"], budget["tests"])
    rec.status("completed")
    return {
        "report": report,
        "status": "completed",
        "phase": "done",
        "seeds": seeds,
        "premise": track,
        "budget": budget,
    }


def _render_report(answer: str, rows: list[dict[str, Any]]) -> str:
    lines = [answer, "", "| Seed | Verdict | Judge |", "|---|---|---|"]
    for row in rows:
        judge_view = f"{row['stance']} ({row['confidence']})" if row["stance"] else "n/a"
        lines.append(f"| {row['statement']} | {row['verdict']} | {judge_view} |")
    for row in rows:
        if row.get("summary"):
            lines += ["", f"**{row['statement']}**: {row['summary']}"]
    return "\n".join(lines)


# -- approval mode ---------------------------------------------------------------------------


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
    rest = state["approval"][1:]
    rec.mark(node_id, "passed" if approved else "failed", actor="human")
    if approved:
        return {"approval": rest}
    return {"approval": rest, **_rejected(rec, state, node.type, node_id, answer.note)}


def _rejected(
    rec: Recorder, state: AgentState, node_type: str, node_id: str, note: str
) -> dict[str, Any]:
    """What a human rejection does to each kind of decision."""
    if node_type == "premise_check":
        track = copy.deepcopy(state["premise"])
        if track["revisions"] >= MAX_PREMISE_REVISIONS:
            track["label"] = REVIEW_LABEL
            return {"premise": track, "outcome": "review", "phase": "summarize"}
        track.update(
            revisions=track["revisions"] + 1, feedback=f"Human reviewer: {note}", revising=node_id
        )
        return {"premise": track, "outcome": None, "phase": "premise"}

    seeds = copy.deepcopy(state["seeds"])
    owner = next((s for s in seeds.values() if s["node"] == node_id or node_id in _plans(s)), None)
    if owner is None:
        return {}
    if node_type == "hypothesis":  # a rejected seed is pruned: never researched or reported
        rec.mark(node_id, "pruned", actor="human")
        owner["status"] = "pruned"
        if state["current"] != owner["id"]:
            return {"seeds": seeds}
        upcoming = next((sid for sid in state["order"] if seeds[sid]["status"] == "queued"), None)
        if upcoming is None:
            return {"seeds": seeds, "current": None, "phase": "summarize"}
        seeds[upcoming]["status"] = "active"
        return {"seeds": seeds, "current": upcoming, "phase": "research"}

    step = next(s for s in owner["steps"] if s["plan"] == node_id)  # a rejected tool call
    step["error"] = f"rejected by a human reviewer: {note}"
    owner["feedback"] = f"A human reviewer rejected your last call: {note}"
    return {"seeds": seeds, "phase": "research"}


def _plans(track: Seed) -> set[str]:
    return {step["plan"] for step in track["steps"]}


SPECIALISTS = {
    "premise": premise,
    "premise_test": premise_test,
    "seed": seed,
    "research": research,
    "execute": execute,
    "judge": judge,
    "supervise": supervise,
    "conclude": conclude,
    "summarize": summarize,
    "human_review": human_review,
}
