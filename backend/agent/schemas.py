"""Structured outputs for LLM decisions D1-D6 and the critic (plan.md §6.5, §8.2, §9.3).

Every decision carries a `rationale`. No schema has a verdict label: labels come from the gate.
`HypothesisSpec` doubles as the tool input; it mirrors plan.md until Person A's tool contracts land.
"""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, Field, create_model, model_validator

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
Control = Literal[
    "spread",
    "is_home",
    "rest_days",
    "opp_rest_days",
    "b2b",
    "opp_b2b",
    "travel_km",
    "tz_shift",
    "altitude_m",
    "games_into_season",
    "rolling_ats_10",
    "rolling_margin_10",
]
Metric = Literal["ats_margin", "ats_cover", "margin", "win", "total_over", "roi"]
PrimaryTest = Literal["compare_groups", "regress", "persistence", "market_test"]


class Filter(BaseModel):
    col: FilterColumn
    op: Literal["eq", "ne", "in", "gte", "lte", "between"]
    value: bool | int | float | str | list[int | float | str]


class SliceSpec(BaseModel):
    split: Literal["explore", "validate", "special"] = "explore"
    filters: list[Filter] = Field(default_factory=list)


class HypothesisSpec(BaseModel):
    id: str = Field(description="Short snake_case slug, unique within the run")
    statement: str
    kind: Literal["premise", "explanation", "confounder", "chance", "market"]
    metric: Metric
    group_a: SliceSpec
    group_b: SliceSpec | Literal["complement"] = "complement"
    direction: Literal["a_gt_b", "a_lt_b", "two_sided"]
    primary_test: PrimaryTest
    controls: list[Control] = Field(default_factory=list)


class ClaimDecision(BaseModel):
    """D1: the question restated as a measurable claim, or why it cannot be tested."""

    rationale: str
    testable: bool
    claim: HypothesisSpec | None = None
    missing_data: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.testable and (self.claim is None or self.claim.kind != "premise"):
            raise ValueError("a testable question needs a claim with kind='premise'")
        if not self.testable and not self.missing_data:
            raise ValueError("an untestable question must list the missing data")
        return self


class HypothesisSet(BaseModel):
    """D2: 3-5 competing explanations, including one chance and at least one confounder."""

    rationale: str
    hypotheses: list[HypothesisSpec]

    @model_validator(mode="after")
    def _competing(self) -> Self:
        kinds = [h.kind for h in self.hypotheses]
        if not 3 <= len(kinds) <= 5:
            raise ValueError("propose 3-5 hypotheses")
        if kinds.count("chance") != 1 or "confounder" not in kinds or "premise" in kinds:
            raise ValueError("include exactly one chance and at least one confounder hypothesis")
        return self


class HypothesisRevision(BaseModel):
    """D2 after critique: one hypothesis revised (or justified point by point)."""

    rationale: str
    hypothesis: HypothesisSpec


class PlanDecision(BaseModel):
    """D3: the primary test for one hypothesis."""

    rationale: str
    tool: PrimaryTest
    controls: list[Control] = Field(default_factory=list)


FollowUpAction = Literal["out_of_sample", "market_test", "placebo", "holdout_confirm", "conclude"]


class NextTest(BaseModel):
    """D3b: after a result passes its gates, the next follow-up test, or `conclude`."""

    rationale: str
    action: FollowUpAction
    placebo_slice: SliceSpec | None = Field(
        default=None, description="Required for placebo: a slice where the effect should be absent"
    )

    @model_validator(mode="after")
    def _placebo_needs_slice(self) -> Self:
        if self.action == "placebo" and self.placebo_slice is None:
            raise ValueError("a placebo needs placebo_slice")
        return self


def next_test_schema(menu: Sequence[str]) -> type[NextTest]:
    """`NextTest` narrowed to this call's menu, so a locked tool cannot even be chosen."""
    return create_model("NextTest", __base__=NextTest, action=(Literal[tuple(menu)], ...))


class Interpretation(BaseModel):
    """D4: what the gated results do and do not show."""

    rationale: str
    summary: str
    claims_made: list[str]
    limitations: list[str]
    cited_checks: list[str] = Field(description="Gate ids such as G2 that each claim rests on")


class Narrative(BaseModel):
    """D6: prose for a verdict whose label was already set by the gate."""

    rationale: str
    text: str


class ReportDraft(BaseModel):
    """D6: the answer to the user's question, written from the verdict table."""

    rationale: str
    answer: str


Mark = Literal["ok", "problem"]


class CriticChecklist(BaseModel):
    construct_validity: Mark = "ok"
    missing_confounder: Mark = "ok"
    selection_or_method_bias: Mark = "ok"
    overclaim_given_gates: Mark = "ok"
    testable_mechanism: Mark = "ok"


class CriticReview(BaseModel):
    verdict: Literal["approve", "concerns", "reject"]
    checklist: CriticChecklist
    severity: Literal["low", "medium", "high"]
    evidence: list[str] = Field(description="Specific numbers or node ids the review relies on")
    rationale: str
    suggested_fix: str | None = None
