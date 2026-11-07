"""Canonical plan.md section 10 event envelope and typed payloads."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from contracts.api import Intervention
from contracts.models import (
    Actor,
    Budget,
    CheckRecord,
    Contract,
    Edge,
    Mode,
    Node,
    RunStatus,
    Verdict,
)


class RunCreated(Contract):
    type: Literal["run_created"] = "run_created"
    question: str
    mode: Mode
    budget: Budget


class StatusChanged(Contract):
    type: Literal["run_status"] = "run_status"
    status: RunStatus


class NodeCreated(Contract):
    type: Literal["node_created"] = "node_created"
    node: Node


class NodeUpdated(Contract):
    type: Literal["node_updated"] = "node_updated"
    node: Node


class EdgeCreated(Contract):
    type: Literal["edge_created"] = "edge_created"
    edge: Edge


class CheckResult(Contract):
    type: Literal["check_result"] = "check_result"
    check: CheckRecord


class LedgerEntry(Contract):
    type: Literal["ledger_entry"] = "ledger_entry"
    test_id: str
    hypothesis_id: str
    tests_spent: int = Field(ge=0)


class VerdictComputed(Contract):
    type: Literal["verdict_computed"] = "verdict_computed"
    hypothesis_id: str
    verdict: Verdict


class BudgetUpdated(Contract):
    type: Literal["budget_update"] = "budget_update"
    llm_calls_used: int = Field(ge=0)
    tests_spent: int = Field(ge=0)
    depth: int = Field(ge=0)


class ModeChanged(Contract):
    type: Literal["mode_changed"] = "mode_changed"
    mode: Mode


class ErrorOccurred(Contract):
    type: Literal["error"] = "error"
    code: str
    message: str


class InterventionRecorded(Contract):
    type: Literal["intervention"] = "intervention"
    intervention_id: str
    request: Intervention


Payload = Annotated[
    RunCreated
    | StatusChanged
    | NodeCreated
    | NodeUpdated
    | EdgeCreated
    | CheckResult
    | LedgerEntry
    | VerdictComputed
    | BudgetUpdated
    | ModeChanged
    | ErrorOccurred
    | InterventionRecorded,
    Field(discriminator="type"),
]
EventType = Literal[
    "run_created",
    "run_status",
    "node_created",
    "node_updated",
    "edge_created",
    "check_result",
    "ledger_entry",
    "verdict_computed",
    "intervention",
    "budget_update",
    "mode_changed",
    "error",
]


class Event(Contract):
    seq: int = Field(ge=1)
    run_id: str
    ts: AwareDatetime
    actor: Actor
    type: EventType
    node_id: str | None = None
    payload: Payload

    @model_validator(mode="after")
    def matching_type(self):
        if self.type != self.payload.type:
            raise ValueError("type must match payload.type")
        if isinstance(self.payload, (NodeCreated, NodeUpdated)) and (
            self.payload.node.run_id != self.run_id or self.payload.node.id != self.node_id
        ):
            raise ValueError("Node event must match envelope run_id and node_id")
        if isinstance(self.payload, CheckResult) and self.payload.check.node_id != self.node_id:
            raise ValueError("Check result must match envelope node_id")
        return self
