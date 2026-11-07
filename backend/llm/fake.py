"""LLM_MODE=fake: scripted, schema-valid answers so the pipeline runs with zero quota."""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
Script = Callable[[str, type[BaseModel], dict[str, Any]], BaseModel | dict[str, Any]]


class FakeLLM:
    model_id = "fake"

    def __init__(self, script: Script) -> None:
        self.calls = 0
        self.transcript: list[tuple[str, dict[str, Any]]] = []
        self._script = script

    async def structured(
        self,
        decision: str,
        schema: type[T],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> T:
        await asyncio.sleep(0)  # yield to the event loop like a network call would
        self.calls += 1
        self.transcript.append((decision, context))
        answer = self._script(decision, schema, context)
        if isinstance(answer, BaseModel):
            answer = answer.model_dump()
        return schema.model_validate(answer)
