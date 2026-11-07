"""B implements AgentPort; C supplies EventSink. Neither requires FastAPI."""

from typing import Protocol

from contracts.api import CreateRun, Intervention, Report
from contracts.events import Event, Payload
from contracts.models import Actor


class EventSink(Protocol):
    async def emit(self, actor: Actor, payload: Payload, node_id: str | None = None) -> Event:
        """C assigns sequence/timestamp, appends, then publishes a validated Event."""
        ...


class AgentPort(Protocol):
    async def start(self, run_id: str, request: CreateRun, events: EventSink) -> None:
        """Enqueue and return promptly; B owns workers and emits failures/status changes."""
        ...

    async def intervene(
        self, run_id: str, intervention_id: str, request: Intervention, events: EventSink
    ) -> None:
        """Validate state then enqueue; raise ValueError for a 409 conflict."""
        ...

    async def report(self, run_id: str) -> Report | None:
        """None until ready; B supplies narrative and retains A's verdict labels."""
        ...
