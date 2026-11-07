"""Append-only event log for one run, in the contract's wire format (`contracts/events.py`).

The agent keeps its own synchronous log and projection; C's `EventSink` receives the same events
once the agent is connected through `AgentPort`. In memory for now.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from backend.events.projection import ResearchGraph
from contracts.events import Event, Payload
from contracts.models import Actor

Listener = Callable[[Event], None]


class EventLog:
    def __init__(self, run_id: str, *, clock: Callable[[], datetime] | None = None) -> None:
        self.run_id = run_id
        self.graph = ResearchGraph()
        self._events: list[Event] = []
        self._listeners: list[Listener] = []
        self._clock = clock or (lambda: datetime.now(UTC))

    def now(self) -> datetime:
        return self._clock()

    def append(self, actor: Actor, payload: Payload, node_id: str | None = None) -> Event:
        event = Event(
            seq=len(self._events) + 1,
            run_id=self.run_id,
            ts=self.now(),
            actor=actor,
            type=payload.type,
            node_id=node_id,
            payload=payload,
        )
        self._events.append(event)
        self.graph.apply(event)
        for listener in list(self._listeners):
            listener(event)
        return event

    def events(self, from_seq: int = 1) -> list[Event]:
        return self._events[max(from_seq, 1) - 1 :]

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Call `listener` on every new event; returns an unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def to_jsonl(self) -> str:
        return "".join(event.model_dump_json() + "\n" for event in self._events)
