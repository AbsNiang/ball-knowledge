"""The interface every LLM call goes through (orchestrator decisions and the critic alike)."""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
        Rate limiting, caching and 429 backoff live behind this method.
        """
        ...
