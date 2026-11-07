"""Per-run dependencies passed to every LangGraph node as `runtime.context` (never checkpointed)."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from backend.agent.checks import CheckRunner
from backend.agent.ports import Gate, Tools
from backend.events.store import EventLog
from backend.llm.gateway import LLMGateway


def _running() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


@dataclass
class AgentContext:
    llm: LLMGateway
    tools: Tools
    gate: Gate
    log: EventLog
    checks: CheckRunner
    running: asyncio.Event = field(default_factory=_running)
    inbox: list[dict[str, Any]] = field(default_factory=list)
    """Interventions waiting for the router's next step, such as {'type': 'set_mode', ...}."""

    @classmethod
    def create(cls, *, llm: LLMGateway, tools: Tools, gate: Gate, log: EventLog) -> "AgentContext":
        return cls(
            llm=llm,
            tools=tools,
            gate=gate,
            log=log,
            checks=CheckRunner(gate=gate, llm=llm, log=log),
        )

    def take_inbox(self) -> list[dict[str, Any]]:
        messages, self.inbox = self.inbox, []
        return messages
