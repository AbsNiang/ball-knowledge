"""Structured outputs of the research loop's LLM roles, and the specs the researcher declares.

Roles: premise (D1), seeder, researcher (tool calls, see `toolkit.py`), judge, supervisor and
aggregator. No schema has a verdict label: labels come from the gate. `HypothesisSpec` doubles as
the tool input; it mirrors plan.md until Person A's tool contracts land.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

MAX_SEEDS = 3

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
SeedKind = Literal["explanation", "confounder", "market"]
Direction = Literal["a_gt_b", "a_lt_b", "two_sided"]


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
    direction: Direction
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


# -- seeds -----------------------------------------------------------------------------------


class SeedIdea(BaseModel):
    id: str = Field(description="Short snake_case slug, unique within the run")
    statement: str = Field(description="One specific, falsifiable explanation, in one sentence")
    kind: SeedKind
    rationale: str = Field(description="Why it could answer the question: sources and reasoning")
    sources: list[str] = Field(
        default_factory=list, description="URLs from `search.sources` that suggest this idea"
    )
    testable_with_data: bool = Field(
        description="Whether the dataset's columns can express a statistical test of it"
    )


class SeedSet(BaseModel):
    """Up to three competing ideas that would answer the user's question."""

    rationale: str
    seeds: list[SeedIdea]

    @model_validator(mode="after")
    def _size(self) -> Self:
        if not 1 <= len(self.seeds) <= MAX_SEEDS:
            raise ValueError(f"propose 1-{MAX_SEEDS} seeds")
        return self


# -- tests the researcher declares -----------------------------------------------------------


class _TestFields(BaseModel):
    statement: str = Field(description="What this test checks, in one sentence")
    kind: SeedKind
    metric: Metric
    group_a: SliceSpec
    group_b: SliceSpec | Literal["complement"] = "complement"
    direction: Direction = Field(description="Declared before the test runs")
    controls: list[Control] = Field(default_factory=list)


class ExploreSpec(_TestFields):
    """An exploratory test. Its id is set by the system; its p-value enters the ledger."""


class DeclaredTest(_TestFields):
    """The seed's primary test. Frozen once declared; it decides the seed's verdict."""

    primary_test: PrimaryTest


# -- judge, supervisor, aggregator -----------------------------------------------------------


class JudgedClaim(BaseModel):
    text: str
    evidence: list[str] = Field(description="Node ids from `evidence` that back this claim")


class Judgment(BaseModel):
    """The judge's review of one research step and its running synthesis of the seed."""

    step_review: Literal["sound", "concerns", "flawed"]
    step_rationale: str = Field(
        description="Construct validity, confounders, selection or look-ahead bias, overclaiming"
    )
    synthesis: str = Field(description="What all the evidence so far says, in 2-5 sentences")
    stance: Literal["supported", "refuted", "mixed", "insufficient"]
    confidence: Literal["low", "medium", "high"]
    claims: list[JudgedClaim] = Field(default_factory=list)
    self_check: str = Field(
        description="Check your own synthesis: which evidence cuts against it, and did you claim "
        "more than the gate results allow?"
    )
    open_questions: list[str] = Field(default_factory=list)
    suggested_next: str | None = None


class Supervision(BaseModel):
    """The supervisor's call after each judged step on the active seed."""

    rationale: str
    action: Literal["continue", "conclude"]
    guidance: str | None = Field(
        default=None, description="For continue: what the researcher should establish next"
    )

    @model_validator(mode="after")
    def _guided(self) -> Self:
        if self.action == "continue" and not (self.guidance or "").strip():
            raise ValueError("continue needs guidance for the researcher")
        return self


class SeedSummary(BaseModel):
    seed_id: str
    summary: str


class Aggregate(BaseModel):
    """The final answer: every seed's outcome, aggregated into an answer to the question."""

    rationale: str
    answer: str
    seeds: list[SeedSummary] = Field(default_factory=list)
