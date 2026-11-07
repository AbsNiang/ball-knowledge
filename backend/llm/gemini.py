"""LLM_MODE=gemini: real structured decisions from Gemini through the one gateway.

Every orchestrator decision and critic review goes through `structured`. Requests share one
sliding-window rate limiter (default 5 per minute, half the free-tier limit) and are retried on
429/5xx. The reply must be JSON matching the decision's schema; a reply that does not fit raises
`pydantic.ValidationError`, which the agent retries once with the error attached.

Configuration comes from the environment (`.env` is loaded by the CLI): `GEMINI_API_KEY`,
`GEMINI_MODEL` and optionally `GEMINI_RPM`. The key is never logged or written to events.
"""

import asyncio
import json
import os
import time
from collections import deque
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_RPM = 5
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4


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

    async def structured(
        self,
        decision: str,
        schema: type[T],
        *,
        system: str,
        context: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> T:
        self.calls += 1
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=0,
        )
        contents = f"Decision: {decision}\nContext (JSON):\n{json.dumps(context, default=str)}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model_id, contents=contents, config=config
                )
                break
            except errors.APIError as exc:
                if exc.code not in RETRYABLE or attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(5 * 2 ** (attempt - 1))
        return schema.model_validate_json(response.text or "")
