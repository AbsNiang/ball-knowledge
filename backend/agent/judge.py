"""The judge: reviews each research step and keeps a running synthesis of each seed.

It reads the evidence (tool outputs, gate checks, web summaries), never the researcher's or the
supervisor's reasons, and uses its own system prompt (plan.md risk 6). The schema makes it
self-check in words; `audit` then enforces the checks that must not depend on its honesty:
- every claim must cite node ids the judge was shown, or it is dropped;
- once a stopping gate (G1-G5) fails, the seed cannot be called supported;
- until the primary test holds (G1-G4), confidence cannot be high.
Gate results stay final: the judge can flag or fail a step, never upgrade a gate or a label.
"""

from typing import Any

from backend.agent.ports import CheckResult
from backend.agent.schemas import Judgment
from backend.agent.state import (
    GateStatus,
    Premise,
    Seed,
    Step,
    primary_held,
    stopping_failures,
)
from backend.agent.toolkit import step_summary
from backend.events.projection import ResearchGraph

_STATUS = {"sound": "pass", "concerns": "flag", "flawed": "fail"}
_SEVERITY = {"sound": "low", "concerns": "medium", "flawed": "high"}


def gate_notes(graph: ResearchGraph, node_id: str | None) -> list[str]:
    if node_id is None or node_id not in graph.nodes:
        return []
    return [
        f"{c.gate_id or 'gate'} {c.status}: {c.rationale}"
        for c in graph.nodes[node_id].checks
        if c.checker == "gate"
    ]


def _evidence(graph: ResearchGraph, seed: Seed, extra: list[str]) -> list[dict[str, Any]]:
    items = [
        step_summary(graph, step, index)
        for index, step in enumerate(seed["steps"], 1)
        if step["result"] and not step["error"]
    ]
    for node_id in extra:
        node = graph.nodes[node_id]
        summary = node.tool.output.get("summary", "")[:800] if node.tool else node.title
        items.append({"result_id": node_id, "type": node.type, "summary": summary})
    for item in items:
        item["id"] = item.pop("result_id", None)
    return [item for item in items if item["id"]]


def seed_context(
    graph: ResearchGraph, question: str, seed: Seed, step: Step, extra: list[str]
) -> dict[str, Any]:
    index = seed["steps"].index(step) + 1
    previous = seed["judgment"]
    return {
        "question": question,
        "seed": {k: seed[k] for k in ("id", "statement", "kind")},
        "primary_test": seed["primary"],
        "verdict_gates": seed["gates"],
        "step": {
            **step_summary(graph, step, index),
            "plan_id": step["plan"],
            "gate_checks": gate_notes(graph, step["result"]),
        },
        "evidence": _evidence(graph, seed, extra),
        "previous_synthesis": {k: previous[k] for k in ("synthesis", "stance", "confidence")}
        if previous
        else None,
    }


def premise_context(graph: ResearchGraph, question: str, premise: Premise) -> dict[str, Any]:
    result = graph.nodes[premise["result"]]
    output = result.tool.output
    summary = {
        "id": result.id,
        "tool": output["tool"],
        "status": output["status"],
        "reason": output["reason"],
        "n": output["n"],
        "effect": output["effect"],
        "ci": [output["ci_low"], output["ci_high"]],
        "p_value": output["p_value"],
        "gates": premise["gates"],
    }
    return {
        "question": question,
        "seed": {
            "id": "premise",
            "statement": "The claim restates the question faithfully and the test measures it",
        },
        "primary_test": premise["spec"],
        "verdict_gates": premise["gates"],
        "step": {
            **summary,
            "plan_id": premise["node"],
            "gate_checks": gate_notes(graph, premise["result"]),
        },
        "evidence": [summary],
        "previous_synthesis": None,
    }


def known_ids(context: dict[str, Any]) -> set[str]:
    ids = {item["id"] for item in context["evidence"]}
    step = context["step"]
    return ids | {step.get("plan_id"), step.get("result_id"), step.get("id")} - {None}


def audit(
    judgment: Judgment, known: set[str], gates: dict[str, GateStatus]
) -> tuple[dict[str, Any], list[str]]:
    """The judgment as it counts, and what the rules changed."""
    notes: list[str] = []
    claims = []
    for claim in judgment.claims:
        cited = [node_id for node_id in claim.evidence if node_id in known]
        if not cited:
            notes.append(f"Dropped a claim that cites no evidence it was shown: {claim.text}")
            continue
        claims.append({"text": claim.text, "evidence": cited})
    stance, confidence = judgment.stance, judgment.confidence
    failed = stopping_failures(gates)
    if stance == "supported" and failed:
        stance = "mixed"
        notes.append(f"Stance capped at mixed: the primary test failed {', '.join(failed)}.")
    if confidence == "high" and not primary_held(gates):
        confidence = "medium"
        notes.append("Confidence capped at medium: the primary test has not held (G1-G4).")
    audited = {
        **judgment.model_dump(mode="json"),
        "claims": claims,
        "stance": stance,
        "confidence": confidence,
        "audit": notes,
    }
    return audited, notes


def to_check(node_id: str, judgment: dict[str, Any]) -> CheckResult:
    review = judgment["step_review"]
    return CheckResult(
        node_id=node_id,
        checker="judge",
        status=_STATUS[review],
        severity=_SEVERITY[review],
        rationale=judgment["step_rationale"],
        evidence=sorted({e for claim in judgment["claims"] for e in claim["evidence"]}),
    )


def render(judgment: dict[str, Any]) -> str:
    """Markdown for the inspector's Research tab."""
    lines = [
        f"**Step review: {judgment['step_review']}.** {judgment['step_rationale']}",
        "",
        judgment["synthesis"],
        "",
        f"**Stance:** {judgment['stance']} ({judgment['confidence']} confidence)",
    ]
    if judgment["claims"]:
        lines += ["", "**Claims**"]
        lines += [f"- {c['text']} ({', '.join(c['evidence'])})" for c in judgment["claims"]]
    lines += ["", f"**Self-check:** {judgment['self_check']}"]
    if judgment["open_questions"]:
        lines += ["", "**Open questions**", *(f"- {q}" for q in judgment["open_questions"])]
    if judgment["audit"]:
        lines += ["", "**Audit**", *(f"- {note}" for note in judgment["audit"])]
    return "\n".join(lines)
