"""Initial event envelope from TASKS.md; coordinate changes with all owners."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=1)
    run_id: str
    ts: datetime
    kind: Literal[
        "node_created", "node_updated", "edge_created", "check_result", "intervention", "run_status"
    ]
    data: dict[str, Any]

