"""Development API serving a validated mock event log."""

from pathlib import Path

from fastapi import FastAPI

from contracts.events import Event

app = FastAPI(title="Whack", version="0.1.0")
REPLAY_PATH = Path(__file__).resolve().parents[2] / "replays" / "mock.jsonl"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/fixtures/events", response_model=list[Event])
def fixture_events() -> list[Event]:
    return [
        Event.model_validate_json(line)
        for line in REPLAY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

