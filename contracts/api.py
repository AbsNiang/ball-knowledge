"""FastAPI wire requests and responses; no dataset schema is defined here."""

from typing import Annotated, Literal

from pydantic import Field

from contracts.models import Budget, Contract, Mode, Verdict


class CreateRun(Contract):
    question: str = Field(min_length=1, max_length=4000)
    mode: Mode = "live"
    budget: Budget = Field(default_factory=Budget)


class RuntimeInfo(Contract):
    ready: bool
    execution_mode: Literal["gemini_real", "unavailable"]
    model_id: str | None = None
    dataset_version: str | None = None
    historical_coverage: str = "2007–08 through 2021–22 (2021–22 sealed holdout). Web research does not extend statistical coverage."
    unavailable_reason: str | None = None
    max_llm_calls: int = 60
    max_tests: int = 12
    max_outbound_attempts: int = 120


class RunAccepted(Contract):
    run_id: str


class OptionalNote(Contract):
    note: str | None = None


class RequiredNote(Contract):
    note: str = Field(min_length=10)


class OverridePayload(RequiredNote):
    check_id: str


class ModePayload(Contract):
    mode: Mode


class HypothesisPayload(Contract):
    statement: str = Field(min_length=1, max_length=4000)


class Prune(Contract):
    type: Literal["prune"]
    node_id: str
    payload: OptionalNote = Field(default_factory=OptionalNote)


class Rerun(Contract):
    type: Literal["rerun"]
    node_id: str
    payload: Contract = Field(default_factory=Contract)


class Override(Contract):
    type: Literal["override"]
    node_id: str
    payload: OverridePayload


class Approve(Contract):
    type: Literal["approve"]
    node_id: str
    payload: OptionalNote = Field(default_factory=OptionalNote)


class Reject(Contract):
    type: Literal["reject"]
    node_id: str
    payload: RequiredNote


class SetMode(Contract):
    type: Literal["set_mode"]
    node_id: None = None
    payload: ModePayload


class AddHypothesis(Contract):
    type: Literal["add_hypothesis"]
    node_id: None = None
    payload: HypothesisPayload


class Pause(Contract):
    type: Literal["pause"]
    node_id: None = None
    payload: Contract = Field(default_factory=Contract)


class Resume(Contract):
    type: Literal["resume"]
    node_id: None = None
    payload: Contract = Field(default_factory=Contract)


class Redirect(Contract):
    type: Literal["redirect"]
    node_id: str
    payload: HypothesisPayload


Intervention = Annotated[
    Prune
    | Rerun
    | Override
    | Approve
    | Reject
    | SetMode
    | AddHypothesis
    | Pause
    | Resume
    | Redirect,
    Field(discriminator="type"),
]


class InterventionAccepted(Contract):
    intervention_id: str
    status: Literal["accepted"] = "accepted"


class ReplayRequest(Contract):
    log_file: str = Field(min_length=1)
    speed: float = Field(default=1, gt=0, le=100, allow_inf_nan=False)


class ReportRow(Contract):
    hypothesis_id: str
    title: str
    verdict: Verdict
    path: list[str]
    summary: str


class Report(Contract):
    run_id: str
    question: str
    answer: str
    verdicts: list[ReportRow]


class ErrorDetail(Contract):
    code: str
    message: str


class ErrorResponse(Contract):
    detail: ErrorDetail
