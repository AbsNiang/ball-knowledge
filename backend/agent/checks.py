"""Background gate and critic checks (plan.md §4.4).

Specialists submit checks and move on; the router drains finished batches at its next step.
A check that errors fails closed (gate: fail, critic: flag) instead of passing silently.
"""

import asyncio
from collections.abc import Collection, Coroutine, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from backend.agent import critic
from backend.agent.ports import CheckResult, Gate, ToolResult
from backend.agent.record import Recorder
from backend.events.store import EventLog
from backend.llm.gateway import LLMGateway


@dataclass(frozen=True)
class CheckBatch:
    node_id: str
    checker: Literal["gate", "critic"]
    results: list[CheckResult]


class CheckRunner:
    def __init__(self, *, gate: Gate, llm: LLMGateway, log: EventLog) -> None:
        self._gate = gate
        self._llm = llm
        self._log = log
        self._rec = Recorder(log)
        self._tasks: dict[str, list[asyncio.Task[None]]] = {}
        self._done: list[CheckBatch] = []

    def submit_gate(self, node_id: str, result: ToolResult, spec: Mapping[str, Any]) -> None:
        self._spawn(node_id, "gate", self._run_gate(node_id, result, spec))

    def submit_critic(self, node_id: str, facts: Mapping[str, Any] | None = None) -> None:
        """`facts` are extra evidence the node itself does not carry (a verdict's gates)."""
        self._spawn(node_id, "critic", self._run_critic(node_id, facts))

    def pending(self) -> set[str]:
        return {nid for nid, tasks in self._tasks.items() if any(not t.done() for t in tasks)}

    def drain(self) -> list[CheckBatch]:
        done, self._done = self._done, []
        return done

    async def wait(self, node_ids: Collection[str] | None = None) -> None:
        tasks = [
            task
            for nid, node_tasks in self._tasks.items()
            if node_ids is None or nid in node_ids
            for task in node_tasks
        ]
        await asyncio.gather(*tasks)

    def _spawn(self, node_id: str, checker: str, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro, name=f"{checker}:{node_id}")
        self._tasks.setdefault(node_id, []).append(task)

    async def _run_gate(self, node_id: str, result: ToolResult, spec: Mapping[str, Any]) -> None:
        try:
            results = self._gate.evaluate(node_id, result, spec)
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
        self._finish(CheckBatch(node_id, "gate", results))

    async def _run_critic(self, node_id: str, facts: Mapping[str, Any] | None) -> None:
        try:
            result = await critic.review(self._llm, self._log.graph, node_id, facts)
        except Exception as exc:  # noqa: BLE001 - fail closed and keep the run alive
            result = CheckResult(
                node_id=node_id,
                checker="critic",
                status="flag",
                severity="low",
                rationale=f"Critic error: {exc!r}",
            )
        self._finish(CheckBatch(node_id, "critic", [result]))

    def _finish(self, batch: CheckBatch) -> None:
        for result in batch.results:
            self._rec.check(result)
        self._done.append(batch)
