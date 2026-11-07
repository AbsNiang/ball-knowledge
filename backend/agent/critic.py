"""LLM critic (plan.md §9.3): adversarial, fixed checklist, can only downgrade a node.

It sees the decision, its rationale and the parent evidence, never the orchestrator's prompts.
"""

from collections.abc import Mapping
from typing import Any

from backend.agent.ports import CheckResult
from backend.agent.prompts import PROMPTS
from backend.agent.schemas import CriticReview
from backend.events.projection import ResearchGraph
from backend.llm.gateway import LLMGateway

_STATUS = {"approve": "pass", "concerns": "flag", "reject": "fail"}


def evidence_for(
    graph: ResearchGraph, node_id: str, facts: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    parents = [graph.nodes[parent] for parent in graph.parents(node_id)]
    return {
        "gates": (facts or {}).get("gates"),
        "parents": [
            {
                "id": parent.id,
                "type": parent.type,
                "title": parent.title,
                "tool_output": parent.tool.output if parent.tool else None,
                "gate_checks": [
                    c.model_dump(mode="json") for c in parent.checks if c.checker == "gate"
                ],
            }
            for parent in parents
        ],
    }


async def review(
    llm: LLMGateway, graph: ResearchGraph, node_id: str, facts: Mapping[str, Any] | None = None
) -> CheckResult:
    node = graph.nodes[node_id]
    context = {
        "node_id": node_id,
        "decision_type": node.type,
        "title": node.title,
        "output": node.llm.parsed_output if node.llm else None,
        "rationale": node.llm.rationale if node.llm else None,
        "evidence": evidence_for(graph, node_id, facts),
    }
    result = await llm.structured(
        "critic", CriticReview, system=PROMPTS["critic"], context=context, meta={"node_id": node_id}
    )
    return to_check(node_id, result)


def to_check(node_id: str, result: CriticReview) -> CheckResult:
    status = _STATUS[result.verdict]
    if status != "pass" and not result.evidence:
        return CheckResult(
            node_id=node_id,
            checker="critic",
            status="pass",
            severity="low",
            rationale=f"Objection discarded: it cited no evidence. {result.rationale}",
        )
    fix = f" Suggested fix: {result.suggested_fix}" if result.suggested_fix else ""
    return CheckResult(
        node_id=node_id,
        checker="critic",
        status=status,
        severity=result.severity,
        rationale=result.rationale + fix,
        evidence=list(result.evidence),
    )
