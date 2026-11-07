"""Single-process development event log. Replace with durable storage before production."""

import asyncio
from datetime import UTC, datetime

from contracts.events import Event, Payload
from contracts.models import Actor


class MemoryEventLog:
    def __init__(self):
        self.events: dict[str, list[Event]] = {}
        self.changed = asyncio.Condition()

    def create(self, run_id: str) -> None:
        self.events[run_id] = []

    def read(self, run_id: str, from_seq: int = 1) -> list[Event]:
        return [
            event.model_copy(deep=True) for event in self.events[run_id] if event.seq >= from_seq
        ]

    async def append(
        self, run_id: str, actor: Actor, payload: Payload, node_id: str | None = None
    ) -> Event:
        async with self.changed:
            event = Event(
                seq=len(self.events[run_id]) + 1,
                run_id=run_id,
                ts=datetime.now(UTC),
                actor=actor,
                type=payload.type,
                node_id=node_id,
                payload=payload,
            )
            self.events[run_id].append(event.model_copy(deep=True))
            self.changed.notify_all()
            return event


class RunEventSink:
    def __init__(self, log: MemoryEventLog, run_id: str):
        self.log = log
        self.run_id = run_id

    async def emit(self, actor: Actor, payload: Payload, node_id: str | None = None) -> Event:
        return await self.log.append(self.run_id, actor, payload, node_id)
