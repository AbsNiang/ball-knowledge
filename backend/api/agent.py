"""Bridge the standalone agent to the API's event sink and background lifecycle."""

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.api import CreateRun, Intervention, Report, ReportRow
from contracts.events import Event, StatusChanged
from contracts.ports import EventSink

logger = logging.getLogger(__name__)


class AgentUnavailableError(RuntimeError):
    """Missing dependencies or invalid server-side agent configuration."""


@dataclass
class Session:
    run: Any
    events: EventSink
    queue: asyncio.Queue
    unsubscribe: Callable
    publisher: asyncio.Task
    worker: asyncio.Task | None = None
    paused: bool = False


def configured_run(run_id: str, request: CreateRun):
    """Keep the CLI's defaults: fake LLM and explicitly synthetic tools/gates."""
    try:
        from dotenv import load_dotenv

        from backend.agent.graph import AgentRun
        from backend.agent.stubs import StubGate, StubTools, b2b_script
        from backend.llm.fake import FakeLLM

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        mode = os.environ.get("LLM_MODE", "fake")
        if mode == "gemini":
            from backend.llm.gemini import GeminiGateway

            llm = GeminiGateway.from_env()
        elif mode == "fake":
            llm = FakeLLM(b2b_script)
        else:
            raise RuntimeError("LLM_MODE must be fake or gemini.")
        return AgentRun(
            request.question,
            run_id=run_id,
            mode=request.mode,
            budget=request.budget.model_dump(include={"max_llm_calls", "max_tests"}),
            llm=llm,
            tools=StubTools(),
            gate=StubGate(),
        )
    except ImportError as exc:
        raise AgentUnavailableError(
            'Install the backend dependencies with python -m pip install -e ".[dev]".'
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise AgentUnavailableError(str(exc)) from exc


class AgentService:
    def __init__(self, factory: Callable = configured_run):
        self.factory = factory
        self.sessions: dict[str, Session] = {}

    async def start(self, run_id: str, request: CreateRun, events: EventSink) -> None:
        run = self.factory(run_id, request)
        queue: asyncio.Queue[Event] = asyncio.Queue()

        def enqueue(event: Event):
            # The API owns creation and accepted intervention envelopes/IDs.
            if event.type not in {"run_created", "intervention"}:
                queue.put_nowait(event.model_copy(deep=True))

        async def publish():
            while True:
                event = await queue.get()
                try:
                    await events.emit(event.actor, event.payload, event.node_id)
                finally:
                    queue.task_done()

        session = Session(
            run, events, queue, run.log.subscribe(enqueue), asyncio.create_task(publish())
        )
        self.sessions[run_id] = session
        session.worker = asyncio.create_task(self._drive(session, run.start()))

    async def _drive(self, session: Session, invocation):
        try:
            await invocation
        except Exception as exc:  # noqa: BLE001 - AgentRun logs the failure as an event
            # AgentRun emits the error and failed status in the same ordered log.
            logger.error("Agent run %s failed (%s)", session.run.run_id, type(exc).__name__)
            await session.run.context.checks.aclose()
        finally:
            # Shutdown cancels the publisher too, so don't wait on its queue then.
            if not asyncio.current_task().cancelling():
                await session.queue.join()

    async def intervene(
        self, run_id: str, intervention_id: str, request: Intervention, events: EventSink
    ) -> None:
        session = self.sessions[run_id]
        run = session.run
        if run.log.graph.status in {"completed", "failed", "cancelled"}:
            raise ValueError("This session has finished.")
        if request.type == "pause":
            run.context.running.clear()
            session.paused = True
            # An approval interrupt is already at a boundary. Otherwise the router
            # publishes paused when the in-flight step has finished.
            if run.pending_review:
                run.log.append("system", StatusChanged(status="paused"))
        elif request.type == "resume":
            if not session.paused:
                raise ValueError("This session is not paused.")
            session.paused = False
            status = "awaiting_approval" if run.pending_review else "running"
            run.log.append("system", StatusChanged(status=status))
            run.context.running.set()
        elif request.type in {"approve", "reject"}:
            if session.paused:
                raise ValueError("Resume the session before reviewing a decision.")
            if not run.pending_review or run.pending_review["node_id"] != request.node_id:
                raise ValueError("This node is not awaiting approval.")
            if session.worker and not session.worker.done():
                raise ValueError("The previous decision is still being processed.")
            run.pending_review = None
            session.worker = asyncio.create_task(
                self._drive(session, run.resume(request.type, request.payload.note))
            )
        elif request.type == "set_mode":
            if run.pending_review:
                raise ValueError("Review the pending decision before changing mode.")
            run.set_mode(request.payload.mode)
        else:
            raise ValueError(
                f"The research agent does not support {request.type} yet. "
                "Redirect is available in Scripted demo."
            )
        if request.type in {"pause", "resume"}:
            await session.queue.join()

    async def report(self, run_id: str) -> Report | None:
        run = self.sessions[run_id].run
        result = run.state.get("report")
        if result is None:
            return None
        rows = []
        extra = []
        for row in result["verdicts"]:
            node = run.log.graph.nodes[row["verdict_node"]]
            if node.verdict is None:
                # Chance conclusions aren't part of the contract's verdict taxonomy.
                extra.append(f"{row['statement']}: {row['verdict']}")
                continue
            rows.append(
                ReportRow(
                    hypothesis_id=run.state["tracks"][row["track"]]["spec_node"],
                    title=row["statement"],
                    verdict=node.verdict,
                    path=row["path"],
                    summary=node.llm.rationale if node.llm else row["verdict"],
                )
            )
        return Report(
            run_id=run_id,
            question=result["question"],
            answer="\n\n".join([result["answer"], *extra]),
            verdicts=rows,
        )

    async def aclose(self):
        tasks = []
        for session in self.sessions.values():
            session.unsubscribe()
            tasks.extend(t for t in (session.worker, session.publisher) if t is not None)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for session in self.sessions.values():
            await session.run.context.checks.aclose()
