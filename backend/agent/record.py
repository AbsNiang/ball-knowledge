"""Writes research-graph events in the contract's types (`contracts/events.py`, plan.md §5).

Edges follow plan.md §5.3: `derives_from` points child -> parent, `revises` new -> old, and
`supports`/`refutes` test result -> hypothesis. Every `node_updated` carries a full snapshot.
"""

import json
from typing import Any, get_args

from pydantic import BaseModel

from backend.agent.ports import CheckResult, ToolResult
from backend.events.projection import ResearchGraph
from backend.events.store import EventLog
from backend.gate.config import SEED
from contracts.api import Intervention
from contracts.events import (
    BudgetUpdated,
    EdgeCreated,
    ErrorOccurred,
    InterventionRecorded,
    ModeChanged,
    NodeCreated,
    NodeUpdated,
    StatusChanged,
    VerdictComputed,
)
from contracts.events import (
    CheckResult as CheckEvent,
)
from contracts.models import (
    Actor,
    CheckRecord,
    Edge,
    HumanAnnotation,
    LLMRecord,
    Mode,
    Node,
    RunStatus,
    ToolRecord,
    Verdict,
)

GATE_IDS = frozenset({"G1", "G2", "G3", "G4", "G5", "G6", "G7"})
CONTRACT_LABELS = frozenset(get_args(Verdict.model_fields["label"].annotation))
_OUTCOMES = frozenset({"passed", "flagged", "failed"})
TITLE_LIMIT = 80


def title(text: str) -> str:
    text = " ".join(text.split()) or "(untitled)"
    return text if len(text) <= TITLE_LIMIT else text[: TITLE_LIMIT - 1] + "…"


def llm_record(model_id: str, decision: str, context: dict[str, Any], out: BaseModel) -> LLMRecord:
    return LLMRecord(
        model_id=model_id,
        prompt=json.dumps({"decision": decision, "context": context}, default=str),
        raw_response=out.model_dump_json(),
        parsed_output=out.model_dump(mode="json"),
        rationale=getattr(out, "rationale", ""),
    )


def tool_record(result: ToolResult, duration_ms: float) -> ToolRecord:
    return ToolRecord(
        tool_name=result.tool,
        args=result.args,
        args_hash=result.args_hash,
        output=result.model_dump(mode="json"),
        data_slice_hash=result.slice_hash or "",
        seed=SEED,
        duration_ms=duration_ms,
    )


def contract_verdict(label: str, gates: dict[str, str]) -> Verdict | None:
    """The contract's `Verdict` for an agent label (plan.md §9.2).

    Agent labels outside the contract's taxonomy keep their detail in the node title:
    `Inconclusive (review)` and `Inconclusive (not replicated)` become `Inconclusive`, and a chance
    hypothesis gets no `Verdict` because it is resolved by rule as supported/refuted (plan.md [A6]).
    """
    if label in CONTRACT_LABELS:
        contract_label = label
    elif label.startswith("Inconclusive ("):
        contract_label = "Inconclusive"
    elif label.startswith("Chance "):
        return None
    else:
        raise ValueError(f"no contract verdict for label {label!r}")
    return Verdict(label=contract_label, holdout_failed=gates.get("G7") == "fail")


class Recorder:
    def __init__(self, log: EventLog) -> None:
        self.log = log

    @property
    def graph(self) -> ResearchGraph:
        return self.log.graph

    # -- nodes and edges ------------------------------------------------------------------

    def node(
        self,
        node_type: str,
        text: str,
        *,
        parents: list[str] | tuple[str, ...] = (),
        revises: str | None = None,
        actor: Actor = "orchestrator",
        done: bool = False,
        llm: LLMRecord | None = None,
        tool: ToolRecord | None = None,
        verdict: Verdict | None = None,
    ) -> str:
        """Create a node. It is `checking` until its checks land, unless `done` (no checks)."""
        node_id = f"{node_type}-{len(self.graph.nodes) + 1}"
        node = Node(
            id=node_id,
            run_id=self.log.run_id,
            type=node_type,
            version=self.graph.nodes[revises].version + 1 if revises else 1,
            created_at=self.log.now(),
            lifecycle="complete" if done else "checking",
            check_outcome="passed" if done else "pending",
            title=title(text),
            llm=llm,
            tool=tool,
            verdict=verdict,
        )
        self.log.append(actor, NodeCreated(node=node), node_id)
        for parent in parents:
            self.edge(node_id, parent, "derives_from")
        if revises:
            self.edge(node_id, revises, "revises")
        return node_id

    def edge(self, source: str, target: str, edge_type: str) -> None:
        edge_id = f"{source}-{target}-{edge_type}"
        if edge_id not in self.graph.edges:
            edge = Edge(id=edge_id, source=source, target=target, type=edge_type)
            self.log.append("orchestrator", EdgeCreated(edge=edge))

    def mark(self, node_id: str, state: str, actor: Actor = "orchestrator") -> None:
        """Apply a router node state: an outcome (`passed`/`flagged`/`failed`), or a flag
        (`tainted`, `awaiting_approval`)."""
        node = self.graph.nodes[node_id]
        flags = set(node.flags)
        changes: dict[str, Any] = {}
        if state in _OUTCOMES:
            changes = {"lifecycle": "complete", "check_outcome": state}
            flags.discard("awaiting_approval")
        elif state == "awaiting_approval":
            changes = {"lifecycle": "checking"}
            flags.add(state)
        elif state == "tainted":
            flags.add(state)
        else:
            raise ValueError(f"unknown node state {state!r}")
        self._snapshot(node, actor, flags=flags, **changes)

    def annotate(self, node_id: str, note: str, actor_id: str = "human") -> None:
        node = self.graph.nodes[node_id]
        annotation = HumanAnnotation(actor_id=actor_id, ts=self.log.now(), note=note)
        self._snapshot(node, "human", human=[*node.human, annotation])

    def _snapshot(self, node: Node, actor: Actor, **changes: Any) -> None:
        updated = Node.model_validate({**node.model_dump(), **changes})
        if updated != node:
            self.log.append(actor, NodeUpdated(node=updated), node.id)

    # -- checks, verdicts, run state ------------------------------------------------------

    def check(self, result: CheckResult) -> CheckRecord:
        failed = result.status == "fail"
        record = CheckRecord(
            id=f"check-{len(self.log.events()) + 1}",
            node_id=result.node_id,
            checker=result.checker,
            gate_id=result.gate_id if result.gate_id in GATE_IDS else None,
            status=result.status,
            severity=result.severity or ("high" if failed else "low"),
            rationale=result.rationale,
            evidence=list(result.evidence),
        )
        self.log.append(result.checker, CheckEvent(check=record), result.node_id)
        return record

    def verdict(self, hypothesis_node: str, verdict: Verdict) -> None:
        self.log.append("gate", VerdictComputed(hypothesis_id=hypothesis_node, verdict=verdict))

    def status(self, status: RunStatus) -> None:
        self.log.append("system", StatusChanged(status=status))

    def budget(self, llm_calls: int, tests: int) -> None:
        payload = BudgetUpdated(
            llm_calls_used=llm_calls, tests_spent=tests, depth=self.graph.depth()
        )
        last = self.graph.budget
        if last is None or last != payload:
            self.log.append("system", payload)

    def mode(self, mode: Mode) -> None:
        self.log.append("system", ModeChanged(mode=mode))

    def intervention(self, request: Intervention) -> None:
        intervention_id = f"intervention-{len(self.graph.interventions) + 1}"
        payload = InterventionRecorded(intervention_id=intervention_id, request=request)
        self.log.append("human", payload, request.node_id)

    def error(self, code: str, message: str) -> None:
        self.log.append("system", ErrorOccurred(code=code, message=message))
