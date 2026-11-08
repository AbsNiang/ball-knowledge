"""What the agent needs from Person A's tools and gate.

`ToolResult` and `GateCheck` are A's (`backend/tools/schemas.py`, `backend/gate/gates.py`), so
tools and the gate plug in without an adapter layer. `CheckResult` is any check's answer inside
the agent (gate, judge or human); `Recorder.check` turns it into the contract's `CheckRecord`.
It allows the pseudo-gate id "error", which fails closed.
"""

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from backend.gate.gates import GateCheck
from backend.tools.hashing import hash_args
from backend.tools.registry import dispatch
from backend.tools.schemas import ToolResult
from backend.tools.state import ToolState, _active

__all__ = [
    "CheckResult",
    "Gate",
    "GateCheck",
    "GateStatus",
    "RegistryTools",
    "ToolResult",
    "Tools",
    "from_gate",
]

GateStatus = Literal["pass", "fail"]


class CheckResult(BaseModel):
    node_id: str
    checker: Literal["gate", "critic", "judge", "human"]
    gate_id: str | None = None
    status: Literal["pass", "flag", "fail"]
    severity: Literal["low", "medium", "high"] | None = None
    rationale: str
    evidence: list[str] = Field(default_factory=list)


def from_gate(check: GateCheck) -> CheckResult:
    return CheckResult(checker="gate", **check.model_dump())


class Tools(Protocol):
    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Deterministic and seeded. Never raises to the agent: errors come back as status='error'."""
        ...


class RegistryTools:
    """Run-scoped statistical history and a shared agent/gate test budget."""

    def __init__(self, max_tests: int = 12):
        self.state = ToolState()
        self.max_tests = min(max_tests, 12)
        self.tests = 0
        self._lock = RLock()

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        # Gate follow-up calls use this same instance and consume the same budget.
        cost = 3 if tool == "holdout_confirm" else int(tool in {
            "compare_groups", "regress", "persistence", "market_test",
            "out_of_sample", "placebo",
        })
        with self._lock:
            if self.tests + cost > self.max_tests:
                return ToolResult(tool=tool, args=args, status="error", reason="test_budget_exhausted", args_hash=hash_args(args))
            self.tests += cost
            token = _active.set(self.state)
            try:
                return dispatch(tool, args)
            finally:
                _active.reset(token)


class Gate(Protocol):
    """Person A's `backend.gate.gates.StatsGate`. One per run: it owns the run's test ledger."""

    def evaluate(
        self, node_id: str, result: ToolResult, spec: Mapping[str, Any]
    ) -> list[GateCheck]:
        """The gates (G1-G7) that apply to this tool's result. `spec` is the frozen hypothesis."""
        ...

    def verdict(
        self,
        kind: str,
        gates: Mapping[str, GateStatus],
        competing: Sequence[tuple[str, Mapping[str, GateStatus]]],
    ) -> str:
        """`compute_verdict` (plan.md §9.2). `competing` is (kind, gates) for the premise and
        every non-chance hypothesis, and is only read for chance hypotheses."""
        ...

    def reevaluate(self) -> list[GateCheck]:
        """G4 over the final ledger (plan.md §9.1): the checks whose result flipped."""
        ...
