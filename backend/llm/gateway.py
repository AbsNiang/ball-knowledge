"""The interface every LLM call goes through: structured decisions, tool choices and web searches.

The agent never talks to a model directly. Rate limiting, caching and 429 backoff live behind
these methods, and `calls` counts every request (searches included) for the run budget.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: `parameters` is a self-contained JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """The one tool the model chose, with its arguments and the reason it gave."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class SearchHit(BaseModel):
    title: str
    url: str


class SearchResult(BaseModel):
    """A grounded web search: a summary of what the sources say, and the sources."""

    query: str
    summary: str
    sources: list[SearchHit] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class BudgetExceeded(RuntimeError):
    """The server's hard per-run execution limit was reached."""


class NoToolCall(ValueError):
    """The model answered without calling one of the offered tools."""


class LLMGateway(Protocol):
    model_id: str
    calls: int
    """Calls made so far; the run budget is read from this counter."""

    async def structured(
        self,
        decision: str,
        schema: type[T],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> T:
        """Answer `system` + `context` (rendered as JSON) as an instance of `schema`.

        Raises `pydantic.ValidationError` when the output does not fit the schema.
        """
        ...

    async def call_tool(
        self,
        decision: str,
        tools: Sequence[ToolSpec],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> ToolCall:
        """Make the model call exactly one of `tools`. Raises `NoToolCall` if it does not.

        Arguments are returned as the model wrote them; the caller validates them.
        """
        ...

    async def search(
        self, query: str, *, system: str, meta: dict[str, Any] | None = None
    ) -> SearchResult:
        """Search the web for `query` and summarise what the sources say."""
        ...
