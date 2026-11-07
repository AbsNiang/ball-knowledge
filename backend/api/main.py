"""C's API skeleton. Inject B's AgentPort using create_app(agent=...)."""

import asyncio
import json
from contextlib import asynccontextmanager
from itertools import pairwise
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import TypeAdapter, ValidationError

from backend.api.demo import DemoAgent
from backend.events.memory import MemoryEventLog, RunEventSink
from contracts.api import (
    CreateRun,
    ErrorResponse,
    Intervention,
    InterventionAccepted,
    ReplayRequest,
    Report,
    RunAccepted,
)
from contracts.events import Event, InterventionRecorded, RunCreated, StatusChanged
from contracts.models import Node
from contracts.ports import AgentPort

REPLAYS = Path(__file__).resolve().parents[2] / "replays"


def fail(status: int, code: str, message: str):
    raise HTTPException(status, detail={"code": code, "message": message})


def load_replay(name: str) -> list[Event]:
    path = (REPLAYS / name).resolve()
    if not path.is_relative_to(REPLAYS.resolve()) or path.suffix != ".jsonl":
        fail(422, "invalid_replay", "Choose a .jsonl file within replays/.")
    if not path.is_file():
        fail(404, "replay_not_found", "Replay file not found.")
    try:
        events = [
            Event.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not events or [e.seq for e in events] != list(range(1, len(events) + 1)):
            raise ValueError("Replay sequence must start at 1 without gaps")
        if len({e.run_id for e in events}) != 1:
            raise ValueError("Replay must contain exactly one run")
        if any(b.ts < a.ts for a, b in pairwise(events)):
            raise ValueError("Replay timestamps must be ordered")
        return events
    except (ValidationError, ValueError, OSError) as exc:
        fail(422, "invalid_replay", str(exc))


def create_app(agent: AgentPort | None = None) -> FastAPI:
    log = MemoryEventLog()
    demo = DemoAgent()
    demo_ids: set[str] = set()
    creation_lock = asyncio.Lock()
    replay_tasks: set[asyncio.Task] = set()
    replay_ids: set[str] = set()

    @asynccontextmanager
    async def lifespan(app):
        yield
        for task in replay_tasks | demo.tasks:
            task.cancel()
        await asyncio.gather(*replay_tasks, *demo.tasks, return_exceptions=True)

    app = FastAPI(
        title="Whack",
        version="0.2.0",
        lifespan=lifespan,
        responses={status: {"model": ErrorResponse} for status in (404, 409, 503)},
    )
    app.state.event_log = log

    def require_run(run_id: str):
        if run_id not in log.events:
            fail(404, "run_not_found", "Run does not exist.")

    def require_agent() -> AgentPort:
        if agent is None:
            fail(503, "agent_unavailable", "Person B's AgentPort has not been connected.")
        return agent

    def require_idle(exclude: str | None = None):
        for other in log.events:
            if other == exclude or other in replay_ids:
                continue
            statuses = [
                e.payload.status for e in log.read(other) if isinstance(e.payload, StatusChanged)
            ]
            if not statuses or statuses[-1] in ("queued", "running", "awaiting_approval"):
                fail(409, "run_active", "Pause the active session first.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/fixtures/events", response_model=list[Event])
    def fixture_events():
        return load_replay("mock.jsonl")

    @app.post("/runs", response_model=RunAccepted, status_code=202)
    async def create_run(request: CreateRun):
        async with creation_lock:
            return await start_live(request)

    async def start_live(request: CreateRun):
        runtime = require_agent()
        require_idle()
        run_id = str(uuid4())
        log.create(run_id)
        sink = RunEventSink(log, run_id)
        await sink.emit("system", RunCreated(**request.model_dump()))
        try:
            await runtime.start(run_id, request, sink)
        except Exception:
            await sink.emit("system", StatusChanged(status="failed"))
            raise
        return RunAccepted(run_id=run_id)

    @app.post("/demo/runs", response_model=RunAccepted, status_code=202)
    async def create_demo(request: CreateRun):
        async with creation_lock:
            require_idle()
            run_id = str(uuid4())
            log.create(run_id)
            demo_ids.add(run_id)
            sink = RunEventSink(log, run_id)
            await sink.emit("system", RunCreated(**request.model_dump()))
            await demo.start(run_id, request, sink)
            return RunAccepted(run_id=run_id)

    @app.get("/runs/{run_id}/events", response_model=list[Event])
    def get_events(run_id: str, from_seq: int = Query(default=1, ge=1)):
        require_run(run_id)
        return log.read(run_id, from_seq)

    @app.websocket("/runs/{run_id}/events")
    async def stream_events(websocket: WebSocket, run_id: str, from_seq: int = 1):
        await websocket.accept()
        if run_id not in log.events or from_seq < 1:
            await websocket.send_json(
                {
                    "detail": {
                        "code": "run_not_found" if run_id not in log.events else "invalid_cursor",
                        "message": "Unknown run or invalid from_seq (must be >= 1).",
                    }
                }
            )
            await websocket.close(code=1008)
            return

        async def send_events():
            cursor = from_seq
            while True:
                async with log.changed:
                    batch = log.read(run_id, cursor)
                    if not batch:
                        await log.changed.wait()
                        continue
                for event in batch:
                    await websocket.send_json(event.model_dump(mode="json"))
                    cursor = event.seq + 1

        sender = asyncio.create_task(send_events())
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    await websocket.send_json(
                        {
                            "type": "command_error",
                            "request_id": None,
                            "detail": {"message": "Invalid JSON"},
                        }
                    )
                    continue
                request_id = message.get("request_id") if isinstance(message, dict) else None
                try:
                    request = TypeAdapter(Intervention).validate_python(
                        message.get("command") if isinstance(message, dict) else None
                    )
                    result = await intervene(run_id, request)
                    await websocket.send_json(
                        {"type": "command_ack", "request_id": request_id, **result.model_dump()}
                    )
                except (ValidationError, HTTPException) as exc:
                    detail = (
                        exc.detail
                        if isinstance(exc, HTTPException)
                        else {"message": "Invalid command"}
                    )
                    await websocket.send_json(
                        {"type": "command_error", "request_id": request_id, "detail": detail}
                    )
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    @app.post("/runs/{run_id}/interventions", response_model=InterventionAccepted, status_code=202)
    async def intervene(run_id: str, request: Intervention):
        async with creation_lock:
            return await apply_intervention(run_id, request)

    async def apply_intervention(run_id: str, request: Intervention):
        require_run(run_id)
        if run_id in replay_ids:
            fail(409, "replay_read_only", "Recorded runs do not accept interventions.")
        runtime = demo if run_id in demo_ids else require_agent()
        if request.type in ("resume", "redirect"):
            require_idle(run_id)
        sink = RunEventSink(log, run_id)
        intervention_id = str(uuid4())
        try:
            await runtime.intervene(run_id, intervention_id, request, sink)
        except ValueError as exc:
            fail(409, "intervention_conflict", str(exc))
        await sink.emit(
            "human",
            InterventionRecorded(intervention_id=intervention_id, request=request),
            request.node_id,
        )
        return InterventionAccepted(intervention_id=intervention_id)

    @app.get(
        "/runs/{run_id}/report",
        response_model=Report,
        responses={200: {"content": {"text/markdown": {"schema": {"type": "string"}}}}},
    )
    async def get_report(run_id: str, format: Literal["json", "md"] = "json"):
        require_run(run_id)
        if run_id in replay_ids:
            fail(409, "report_not_ready", "Replay report projection is not implemented yet.")
        report = await require_agent().report(run_id)
        if report is None:
            fail(409, "report_not_ready", "Report is not available yet.")
        if format == "md":
            rows = [f"# Research report\n\n{report.question}\n\n{report.answer}"]
            for row in report.verdicts:
                rows.append(f"## {row.title}\n\n{row.verdict.label}\n\n{row.summary}")
            return PlainTextResponse("\n\n".join(rows), media_type="text/markdown")
        return report

    async def play(run_id: str, events: list[Event], speed: float):
        sink = RunEventSink(log, run_id)
        previous = events[0].ts
        for event in events:
            await asyncio.sleep((event.ts - previous).total_seconds() / speed)
            previous = event.ts
            payload = event.payload.model_copy(deep=True)
            # Remap run provenance while preserving node/edge ids for graph paths.
            if hasattr(payload, "node") and isinstance(payload.node, Node):
                payload.node.run_id = run_id
            await sink.emit(event.actor, payload, event.node_id)

    @app.post("/replay", response_model=RunAccepted, status_code=202)
    async def replay(request: ReplayRequest):
        events = load_replay(request.log_file)
        run_id = str(uuid4())
        log.create(run_id)
        replay_ids.add(run_id)
        task = asyncio.create_task(play(run_id, events, request.speed))
        replay_tasks.add(task)
        task.add_done_callback(replay_tasks.discard)
        return RunAccepted(run_id=run_id)

    return app


app = create_app()
