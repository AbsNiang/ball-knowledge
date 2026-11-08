"""Mutable statistical history scoped to a RegistryTools instance, including thread calls."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolState:
    tests: dict[str, dict] = field(default_factory=dict)
    hypotheses: dict[str, Any] = field(default_factory=dict)
    holdout_used: set[str] = field(default_factory=set)


_active: ContextVar[ToolState | None] = ContextVar("tool_state", default=None)
_standalone = ToolState()


def current() -> ToolState:
    return _active.get() or _standalone
