"""What the agent needs from Person A's tools and gate.

`ToolResult` and `CheckResult` mirror the H0-H2 contracts in TASKS.md. Move them to `contracts/`
(with TypeScript copies) once all owners agree.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

GateStatus = Literal["pass", "fail"]


class ToolResult(BaseModel):
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    n: int | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    slice_hash: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    node_id: str
    checker: Literal["gate", "critic", "human"]
    gate_id: str | None = None
    status: Literal["pass", "flag", "fail"]
    severity: Literal["low", "medium", "high"] | None = None
    rationale: str


class Tools(Protocol):
    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Deterministic and seeded. Never raises to the agent: errors come back as status='error'."""
        ...


class Gate(Protocol):
    def evaluate(
        self, node_id: str, result: ToolResult, spec: Mapping[str, Any]
    ) -> list[CheckResult]:
        """The gates (G1-G7) that apply to this tool's result. `spec` is the frozen hypothesis."""
        ...

    def verdict(
        self,
        kind: str,
        gates: Mapping[str, GateStatus],
        competing: Sequence[Mapping[str, GateStatus]],
    ) -> str:
        """`compute_verdict` (plan.md §9.2). `competing` is only read for chance hypotheses."""
        ...
