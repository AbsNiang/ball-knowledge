"""Product records. A owns tool-specific result and dataset schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


Mode = Literal["live", "approval"]
Actor = Literal["orchestrator", "tool", "gate", "critic", "judge", "human", "system"]
RunStatus = Literal["queued", "running", "paused", "awaiting_approval", "completed", "failed", "cancelled"]


class Budget(Contract):
    max_llm_calls: int = Field(default=60, ge=1)
    max_tests: int = Field(default=40, ge=1)
    max_depth: int = Field(default=8, ge=1)


class Verdict(Contract):
    label: Literal[
        "Premise false",
        "Untestable",
        "Inconclusive",
        "Inconclusive (budget)",
        "Luck",
        "Real but priced in",
        "Real and beats market (provisional)",
    ]
    explanation_status: Literal[
        "Mechanism supported", "Mechanism tested and failed", "Mechanism plausible but untested"
    ] = "Mechanism plausible but untested"
    holdout_failed: bool = False
    human_overridden: bool = False


class LLMRecord(Contract):
    model_id: str
    prompt: str
    raw_response: str
    parsed_output: dict[str, Any]
    rationale: str
    from_cache: bool = False


class ToolRecord(Contract):
    tool_name: str
    args: dict[str, Any]
    args_hash: str
    output: dict[str, Any]
    data_slice_hash: str
    seed: int
    duration_ms: float = Field(ge=0)


class CheckRecord(Contract):
    id: str
    node_id: str
    checker: Literal["gate", "critic", "judge", "human"]
    gate_id: Literal["G1", "G2", "G3", "G4", "G5", "G6", "G7"] | None = None
    status: Literal["pass", "flag", "fail"]
    severity: Literal["low", "medium", "high"]
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class HumanAnnotation(Contract):
    actor_id: str
    ts: datetime
    note: str


class Node(Contract):
    id: str
    run_id: str
    type: Literal[
        "question",
        "premise_check",
        "hypothesis",
        "test_plan",
        "test_result",
        "interpretation",
        "mechanism_prediction",
        "verdict",
        "human_note",
        "source",
        "judgment",
        "report",
    ]
    version: int = Field(default=1, ge=1)
    created_at: datetime
    lifecycle: Literal["proposed", "running", "checking", "complete"] = "proposed"
    check_outcome: Literal["pending", "passed", "flagged", "failed"] = "pending"
    flags: set[Literal["tainted", "pruned", "awaiting_approval", "overridden", "stale"]] = Field(
        default_factory=set
    )
    title: str = Field(min_length=1, max_length=80)
    llm: LLMRecord | None = None
    tool: ToolRecord | None = None
    checks: list[CheckRecord] = Field(default_factory=list)
    human: list[HumanAnnotation] = Field(default_factory=list)
    verdict: Verdict | None = None


class Edge(Contract):
    id: str
    source: str
    target: str
    label: str | None = None
    type: Literal["derives_from", "tests", "supports", "refutes", "revises"]
