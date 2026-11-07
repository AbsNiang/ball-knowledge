"""What the agent needs from Person A's tools and gate.

`ToolResult` is A's (`backend/tools/schemas.py`), so tools plug in without an adapter layer.
`CheckResult` is the gate's and critic's answer inside the agent; `Recorder.check` turns it into
the contract's `CheckRecord`. It allows the pseudo-gate id "error", which fails closed.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from backend.tools.registry import dispatch
from backend.tools.schemas import ToolResult

__all__ = ["CheckResult", "Gate", "GateStatus", "RegistryTools", "ToolResult", "Tools"]

GateStatus = Literal["pass", "fail"]


class CheckResult(BaseModel):
    node_id: str
    checker: Literal["gate", "critic", "human"]
    gate_id: str | None = None
    status: Literal["pass", "flag", "fail"]
    severity: Literal["low", "medium", "high"] | None = None
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class Tools(Protocol):
    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Deterministic and seeded. Never raises to the agent: errors come back as status='error'."""
        ...


class RegistryTools:
    """Person A's tool registry (mock tools until the real ones land) behind the `Tools` port."""

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        return dispatch(tool, args)


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
