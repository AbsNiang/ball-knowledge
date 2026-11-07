"""PROVISIONAL tool input and output models, until `contracts/` has the agreed versions.

`SliceSpec`, `Filter`, `HypothesisSpec` and `MechanismSpec` follow plan.md v2 section 6.5.
`ToolResult` follows TASKS.md, plus `status`, `reason` and `args_hash`, which v2 section 7 needs
but the TASKS.md shape doesn't have. Once agreed, these should move to `contracts/` and this
module should re-export them, so the tools and B's agent code don't change.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SliceSplit = Literal["explore", "validate", "special"]  # holdout is deliberately not allowed
Metric = Literal["ats_margin", "ats_cover", "margin", "win", "total_over", "roi"]
FilterColumn = Literal[
    "season",
    "team",
    "opp",
    "is_home",
    "rest_days",
    "b2b",
    "opp_b2b",
    "travel_km",
    "tz_shift",
    "altitude_m",
    "month",
    "is_bubble",
    "is_playoffs",
    "games_into_season",
]
FeatureName = Literal[
    "rest_days",
    "b2b",
    "travel_km",
    "tz_shift",
    "altitude_m",
    "rolling_ats",
    "rolling_margin",
    "games_into_season",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filter(StrictModel):
    col: FilterColumn
    op: Literal["eq", "ne", "in", "gte", "lte", "between"]
    value: Any

    @model_validator(mode="after")
    def value_matches_op(self) -> "Filter":
        if self.op == "in" and not (isinstance(self.value, list) and self.value):
            raise ValueError("'in' needs a non-empty list")
        if self.op == "between" and not (isinstance(self.value, list) and len(self.value) == 2):
            raise ValueError("'between' needs a [low, high] list")
        return self


class SliceSpec(StrictModel):
    split: SliceSplit = "explore"
    filters: list[Filter]


class MechanismSpec(StrictModel):
    statement: str
    secondary_test: "HypothesisSpec"


class HypothesisSpec(StrictModel):
    id: str
    statement: str
    kind: Literal["premise", "explanation", "confounder", "chance", "market"]
    metric: Metric
    group_a: SliceSpec
    group_b: SliceSpec | Literal["complement"]
    direction: Literal["a_gt_b", "a_lt_b", "two_sided"]
    primary_test: Literal["compare_groups", "regress", "persistence", "market_test"]
    controls: list[str] = []
    mechanism_prediction: MechanismSpec | None = None


class ToolResult(StrictModel):
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "error"] = "ok"
    reason: str | None = None
    args_hash: str
    n: int | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    slice_hash: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DescribeDatasetArgs(StrictModel):
    pass


class DefineSliceArgs(StrictModel):
    spec: SliceSpec


class AddFeatureArgs(StrictModel):
    name: FeatureName
    params: dict[str, int] = {}

    @model_validator(mode="after")
    def rolling_needs_k(self) -> "AddFeatureArgs":
        is_rolling = self.name in ("rolling_ats", "rolling_margin")
        if is_rolling and self.params.get("k", 0) < 1:
            raise ValueError(f"{self.name} needs params.k >= 1")
        return self


class CompareGroupsArgs(StrictModel):
    spec: HypothesisSpec


class RegressArgs(StrictModel):
    spec: HypothesisSpec


class OutOfSampleArgs(StrictModel):
    test_id: str


class PersistenceArgs(StrictModel):
    metric: Metric
    entity: Literal["team"] = "team"


class MarketTestArgs(StrictModel):
    slice: SliceSpec
    side: Literal["team", "opp", "over", "under"]


class PowerArgs(StrictModel):
    n_a: int = Field(gt=0)
    n_b: int = Field(gt=0)
    metric: Metric


class PlaceboArgs(StrictModel):
    test_id: str
    placebo_slice: SliceSpec


class HoldoutConfirmArgs(StrictModel):
    hypothesis_id: str
