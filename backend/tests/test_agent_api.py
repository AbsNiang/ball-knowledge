"""The real orchestrator through HTTP/WS with a fake LLM; no external API calls."""

import asyncio
import time

from fastapi.testclient import TestClient

from backend.agent.graph import AgentRun
from backend.agent.stubs import StubGate, StubTools, b2b_script
from backend.api.agent import AgentService, AgentUnavailableError
from backend.api.main import create_app
from backend.llm.fake import FakeLLM
from contracts.api import Report
from contracts.events import Event


def factory(run_id, request):
    return AgentRun(
        request.question,
        run_id=run_id,
        mode=request.mode,
        llm=FakeLLM(b2b_script),
        tools=StubTools(),
        gate=StubGate(),
    )


def until(api, path, predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = api.get(path)
        if predicate(response):
            return response
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


def test_agent_stream_is_ordered_reconnectable_and_report_matches_verdicts():
    service = AgentService(factory)
    with TestClient(create_app(service)) as api:
        response = api.post("/runs", json={"question": "Why do back-to-back teams underperform?"})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        path = f"/runs/{run_id}"
        with api.websocket_connect(f"{path}/events") as ws:
            first = [ws.receive_json() for _ in range(3)]
        with api.websocket_connect(f"{path}/events?from_seq=4") as ws:
            rest = []
            while True:
                event = Event.model_validate(ws.receive_json())
                rest.append(event.model_dump(mode="json"))
                if event.type == "run_status" and event.payload.status in {"completed", "failed"}:
                    assert event.payload.status == "completed"
                    break
        events = first + rest
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert len([e for e in events if e["type"] == "run_created"]) == 1
        assert all(e["run_id"] == run_id for e in events)
        assert any(e["type"] == "node_created" for e in events)
        response = until(api, f"{path}/report", lambda r: r.status_code == 200)
        report = Report.model_validate(response.json())
        assert report.answer and report.verdicts
        labels = {
            e["payload"]["verdict"]["label"] for e in events if e["type"] == "verdict_computed"
        }
        assert {r.verdict.label for r in report.verdicts} == labels
        assert "Chance supported" in report.answer
        assert api.get(f"{path}/report?format=md").status_code == 200
    assert all(s.worker.done() and s.publisher.done() for s in service.sessions.values())


def test_agent_pause_resume_over_websocket_and_unsupported_redirect():
    class SlowFake(FakeLLM):
        async def structured(self, *args, **kwargs):
            await asyncio.sleep(0.05)
            return await super().structured(*args, **kwargs)

    def slow_factory(run_id, request):
        return AgentRun(
            request.question,
            run_id=run_id,
            llm=SlowFake(b2b_script),
            tools=StubTools(),
            gate=StubGate(),
        )

    service = AgentService(slow_factory)
    with TestClient(create_app(service)) as api:
        run_id = api.post("/runs", json={"question": "Why?"}).json()["run_id"]
        path = f"/runs/{run_id}"
        with api.websocket_connect(f"{path}/events") as ws:
            for command in ("pause", "resume"):
                ws.send_json({"request_id": command, "command": {"type": command}})
                while True:
                    message = ws.receive_json()
                    if message.get("request_id") == command:
                        assert message["type"] == "command_ack"
                        break
                expected = "paused" if command == "pause" else "running"
                events = until(
                    api,
                    f"{path}/events",
                    lambda r, expected=expected: (
                        [e["payload"]["status"] for e in r.json() if e["type"] == "run_status"][-1]
                        == expected
                    ),
                ).json()
                statuses = [e["payload"]["status"] for e in events if e["type"] == "run_status"]
                assert statuses[-1] == ("paused" if command == "pause" else "running")
                if command == "pause":
                    assert (
                        api.post("/demo/runs", json={"question": "Parallel demo"}).status_code
                        == 202
                    )
                    # Pause the demo so the single-active-run rule permits resuming.
                    demo_id = next(i for i in api.app.state.event_log.events if i != run_id)
                    api.post(f"/runs/{demo_id}/interventions", json={"type": "pause"})
            result = api.post(
                f"{path}/interventions",
                json={
                    "type": "redirect",
                    "node_id": "question-1",
                    "payload": {"statement": "Travel"},
                },
            )
            assert result.status_code == 409
            assert "does not support redirect" in result.json()["detail"]["message"]


def test_configuration_failure_is_actionable_and_does_not_block_future_runs():
    def unavailable(*args):
        raise AgentUnavailableError("Set GEMINI_API_KEY and GEMINI_MODEL.")

    with TestClient(create_app(AgentService(unavailable))) as api:
        response = api.post("/runs", json={"question": "Why?"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "agent_unavailable"
        assert api.post("/demo/runs", json={"question": "Fallback"}).status_code == 202


def test_unknown_saved_session_after_restart_is_explicit():
    with TestClient(create_app()) as before:
        run_id = before.post("/demo/runs", json={"question": "Why?"}).json()["run_id"]
    with (
        TestClient(create_app()) as after,
        after.websocket_connect(f"/runs/{run_id}/events?from_seq=20") as ws,
    ):
        assert ws.receive_json()["detail"]["code"] == "run_not_found"


def test_approval_is_forwarded_once_and_restarts_the_checkpointed_agent():
    service = AgentService(factory)
    with TestClient(create_app(service)) as api:
        run_id = api.post("/runs", json={"question": "Why?", "mode": "approval"}).json()["run_id"]
        path = f"/runs/{run_id}"
        session = service.sessions[run_id]
        until(api, f"{path}/events", lambda r: session.run.pending_review and session.worker.done())
        node_id = session.run.pending_review["node_id"]
        result = api.post(f"{path}/interventions", json={"type": "approve", "node_id": node_id})
        assert result.status_code == 202
        until(api, f"{path}/events", lambda r: session.run.pending_review and session.worker.done())
        assert session.run.pending_review["node_id"] != node_id
        events = api.get(f"{path}/events").json()
        approvals = [e for e in events if e["type"] == "intervention"]
        assert len(approvals) == 1
        assert approvals[0]["payload"]["intervention_id"] == result.json()["intervention_id"]


def test_agent_failure_reaches_the_stream_and_releases_active_run_lock():
    def broken_factory(run_id, request):
        run = factory(run_id, request)

        async def broken(*args, **kwargs):
            raise RuntimeError("Deliberate test failure")

        run.graph.ainvoke = broken
        return run

    with TestClient(create_app(AgentService(broken_factory))) as api:
        run_id = api.post("/runs", json={"question": "Why?"}).json()["run_id"]
        events = until(
            api,
            f"/runs/{run_id}/events",
            lambda r: any(
                e["type"] == "run_status" and e["payload"]["status"] == "failed" for e in r.json()
            ),
        ).json()
        assert any(e["type"] == "error" for e in events)
        assert api.post("/demo/runs", json={"question": "Fallback"}).status_code == 202
