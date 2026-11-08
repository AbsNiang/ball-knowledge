"""LLM_MODE=fake: scripted, schema-valid answers so the pipeline runs with zero quota.

One script answers everything. It is called as `script(decision, schema, context)`:
- structured decisions pass their output schema;
- tool choices pass `ToolCall`, with the offered tool names in `context["offered_tools"]`;
- searches pass `SearchResult`, with `decision="search"` and the query in `context["query"]`.
"""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.llm.gateway import NoToolCall, SearchResult, ToolCall, ToolSpec

T = TypeVar("T", bound=BaseModel)
Script = Callable[[str, type[BaseModel], dict[str, Any]], BaseModel | dict[str, Any]]


class FakeLLM:
    model_id = "fake"

    def __init__(self, script: Script) -> None:
        self.calls = 0
        self.transcript: list[tuple[str, dict[str, Any]]] = []
        self._script = script

    async def _answer(self, decision: str, schema: type[T], context: dict[str, Any]) -> T:
        await asyncio.sleep(0)  # yield to the event loop like a network call would
        self.calls += 1
        self.transcript.append((decision, context))
        answer = self._script(decision, schema, context)
        if isinstance(answer, BaseModel):
            answer = answer.model_dump()
        return schema.model_validate(answer)

    async def structured(
        self,
        decision: str,
        schema: type[T],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> T:
        return await self._answer(decision, schema, context)

    async def call_tool(
        self,
        decision: str,
        tools: Sequence[ToolSpec],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> ToolCall:
        offered = [tool.name for tool in tools]
        call = await self._answer(decision, ToolCall, {**context, "offered_tools": offered})
        if call.tool not in offered:  # Gemini's allowed_function_names enforces the same
            raise NoToolCall(f"{call.tool!r} is not one of the offered tools {offered}")
        return call

    async def search(
        self, query: str, *, system: str, meta: dict[str, Any] | None = None
    ) -> SearchResult:
        result = await self._answer("search", SearchResult, {"query": query})
        return result.model_copy(update={"query": query})
