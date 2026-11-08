"""LLM_MODE=gemini: real Gemini calls through the one gateway.

- `structured`: JSON output matching the decision's schema. A reply that does not fit raises
  `pydantic.ValidationError`, which the agent retries once with the error attached.
- `call_tool`: native function calling in `ANY` mode, restricted to the offered tools. Every
  declaration gets a required `reason` parameter, so each choice comes with its reason.
- `search`: a separate request grounded with Google Search. Its sources come from the response's
  grounding metadata, so the agent can cite and record them.

Requests share one sliding-window rate limiter (default 5 per minute, half the free-tier limit)
and are retried on 429/5xx. Configuration comes from the environment (`.env` is loaded by the
CLI): `GEMINI_API_KEY`, `GEMINI_MODEL` and optionally `GEMINI_RPM`. The key is never logged or
written to events.
"""

import asyncio
import json
import os
import time
from collections import deque
from collections.abc import Sequence
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from backend.llm.gateway import NoToolCall, SearchHit, SearchResult, ToolCall, ToolSpec

T = TypeVar("T", bound=BaseModel)

DEFAULT_RPM = 5
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
_REASON = {
    "type": "string",
    "description": "Why this call, and what you expect it to establish (1-2 sentences).",
}


class RateLimiter:
    """At most `rpm` requests in any 60-second window, shared by concurrent callers."""

    def __init__(self, rpm: int, window: float = 60.0) -> None:
        self.rpm = rpm
        self.window = window
        self._sent: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._sent and now - self._sent[0] >= self.window:
                    self._sent.popleft()
                if len(self._sent) < self.rpm:
                    self._sent.append(now)
                    return
                await asyncio.sleep(self.window - (now - self._sent[0]))


def _with_reason(parameters: dict[str, Any]) -> dict[str, Any]:
    required = [name for name in parameters.get("required", []) if name != "reason"]
    return {
        **parameters,
        "type": "object",
        "properties": {**parameters.get("properties", {}), "reason": _REASON},
        "required": [*required, "reason"],
    }


def _prompt(decision: str, context: dict[str, Any]) -> str:
    return f"Decision: {decision}\nContext (JSON):\n{json.dumps(context, default=str)}"


class GeminiGateway:
    def __init__(
        self, *, api_key: str, model_id: str, rpm: int = DEFAULT_RPM, client: Any = None
    ) -> None:
        self.model_id = model_id
        self.calls = 0
        self._client = client or genai.Client(api_key=api_key)
        self._limiter = RateLimiter(rpm)

    @classmethod
    def from_env(cls) -> "GeminiGateway":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        model_id = os.environ.get("GEMINI_MODEL", "").strip()
        if not api_key or not model_id:
            raise RuntimeError("Set GEMINI_API_KEY and GEMINI_MODEL in .env to use LLM_MODE=gemini")
        rpm = int(os.environ.get("GEMINI_RPM") or DEFAULT_RPM)
        return cls(api_key=api_key, model_id=model_id, rpm=rpm)

    async def _generate(self, contents: str, config: types.GenerateContentConfig) -> Any:
        self.calls += 1
        for attempt in range(1, MAX_ATTEMPTS + 1):
            await self._limiter.acquire()
            try:
                return await self._client.aio.models.generate_content(
                    model=self.model_id, contents=contents, config=config
                )
            except errors.APIError as exc:
                if exc.code not in RETRYABLE or attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(5 * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    async def structured(
        self,
        decision: str,
        schema: type[T],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=0,
        )
        response = await self._generate(_prompt(decision, context), config)
        return schema.model_validate_json(response.text or "")

    async def call_tool(
        self,
        decision: str,
        tools: Sequence[ToolSpec],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> ToolCall:
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=_with_reason(tool.parameters),
            )
            for tool in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool.name for tool in tools]
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0,
        )
        response = await self._generate(_prompt(decision, context), config)
        calls = response.function_calls or []
        if not calls:
            raise NoToolCall(f"no function call in the reply: {(response.text or '')[:200]!r}")
        args = dict(calls[0].args or {})
        reason = str(args.pop("reason", ""))
        return ToolCall(tool=calls[0].name or "", args=args, reason=reason)

    async def search(
        self, query: str, *, system: str, meta: dict[str, Any] | None = None
    ) -> SearchResult:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
        )
        response = await self._generate(query, config)
        candidates = response.candidates or []
        grounding = candidates[0].grounding_metadata if candidates else None
        sources, seen = [], set()
        for chunk in (grounding.grounding_chunks or []) if grounding else []:
            web = chunk.web
            if web is None or not web.uri or web.uri in seen:
                continue
            seen.add(web.uri)
            sources.append(SearchHit(title=web.title or web.domain or web.uri, url=web.uri))
        return SearchResult(
            query=query,
            summary=response.text or "",
            sources=sources,
            search_queries=list(grounding.web_search_queries or []) if grounding else [],
        )
