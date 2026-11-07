"""Run from the repository root: python -m contracts.export_schema."""

import json
from pathlib import Path

from contracts.api import (
    CreateRun,
    ErrorResponse,
    Intervention,
    InterventionAccepted,
    ReplayRequest,
    Report,
    RunAccepted,
)
from contracts.events import Event
from contracts.models import Contract


class Contracts(Contract):
    event: Event
    create_run: CreateRun
    run_accepted: RunAccepted
    intervention: Intervention
    intervention_accepted: InterventionAccepted
    replay: ReplayRequest
    report: Report
    error: ErrorResponse


if __name__ == "__main__":
    output = Path(__file__).with_name("schema.json")
    output.write_text(json.dumps(Contracts.model_json_schema(), indent=2) + "\n", encoding="utf-8")
