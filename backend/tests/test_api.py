import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api.main import app, create_app
from contracts.api import Report
from contracts.events import Event
from contracts.export_schema import Contracts

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_fixture_is_ordered_and_contract_valid():
    response = client.get("/fixtures/events")
    assert response.status_code == 200
    events = [Event.model_validate(item) for item in response.json()]
    assert events
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert len({event.run_id for event in events}) == 1
    assert events[-1].payload.status == "completed"


class FakeAgent:
    """Contract-level stand-in for B; deliberately performs no statistical work."""

    async def start(self, run_id, request, events):
        from contracts.events import StatusChanged

        await events.emit("orchestrator", StatusChanged(status="running"))

    async def intervene(self, run_id, intervention_id, request, events):
        if request.type == "override":
            raise ValueError("G7 cannot be overridden")

    async def report(self, run_id):
        return Report(run_id=run_id, question="Why?", answer="Fixture answer", verdicts=[])


def test_unconnected_agent_and_missing_runs():
    assert client.post("/runs", json={"question": "Why?"}).status_code == 503
    for path in ("events", "report"):
        assert client.get(f"/runs/missing/{path}").status_code == 404


def test_run_stream_backfill_live_events_and_reports():
    with TestClient(create_app(FakeAgent())) as api:
        result = api.post("/runs", json={"question": "Why?", "mode": "approval"})
        assert result.status_code == 202
        run_id = result.json()["run_id"]
        path = f"/runs/{run_id}"
        with api.websocket_connect(f"{path}/events?from_seq=2") as ws:
            assert ws.receive_json()["seq"] == 2
            response = api.post(
                f"{path}/interventions",
                json={
                    "type": "set_mode",
                    "payload": {"mode": "live"},
                },
            )
            assert response.status_code == 202
            event = Event.model_validate(ws.receive_json())
            assert event.seq == 3
            assert event.type == "intervention"
            assert event.payload.intervention_id == response.json()["intervention_id"]
        assert len(api.get(f"{path}/events").json()) == 3
        assert api.get(f"{path}/events?from_seq=0").status_code == 422
        assert api.get(f"{path}/report").json()["answer"] == "Fixture answer"
        markdown = api.get(f"{path}/report?format=md")
        assert markdown.headers["content-type"].startswith("text/markdown")
        assert "Fixture answer" in markdown.text


@pytest.mark.parametrize(
    "command",
    [
        {"type": "reject", "node_id": "n", "payload": {}},
        {"type": "override", "node_id": "n", "payload": {"check_id": "c", "note": "short"}},
        {"type": "set_mode", "payload": {"mode": "invalid"}},
        {"type": "rerun", "payload": {}},
        {"type": "rerun", "node_id": "n", "payload": {"arbitrary": True}},
    ],
)
def test_invalid_interventions_rejected_before_dispatch(command):
    with TestClient(create_app(FakeAgent())) as api:
        run_id = api.post("/runs", json={"question": "Why?"}).json()["run_id"]
        assert api.post(f"/runs/{run_id}/interventions", json=command).status_code == 422


def test_agent_conflict_is_not_logged_as_accepted():
    with TestClient(create_app(FakeAgent())) as api:
        run_id = api.post("/runs", json={"question": "Why?"}).json()["run_id"]
        response = api.post(
            f"/runs/{run_id}/interventions",
            json={
                "type": "override",
                "node_id": "n",
                "payload": {"check_id": "g7", "note": "A sufficiently long note"},
            },
        )
        assert response.status_code == 409
        assert len(api.get(f"/runs/{run_id}/events").json()) == 2


def test_replay_uses_same_stream_without_agent():
    with TestClient(create_app()) as api:
        response = api.post("/replay", json={"log_file": "mock.jsonl", "speed": 100})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        expected = api.get("/fixtures/events").json()
        with api.websocket_connect(f"/runs/{run_id}/events") as ws:
            actual = [ws.receive_json() for _ in expected]
        assert [item["payload"] for item in actual if item["type"] != "node_created"] == [
            item["payload"] for item in expected if item["type"] != "node_created"
        ]
        assert all(item["run_id"] == run_id for item in actual)
        assert all(
            item["payload"]["node"]["run_id"] == run_id
            for item in actual
            if item["type"] == "node_created"
        )
        assert api.get(f"/runs/{run_id}/report").status_code == 409
        assert (
            api.post(
                f"/runs/{run_id}/interventions",
                json={
                    "type": "set_mode",
                    "payload": {"mode": "live"},
                },
            ).status_code
            == 409
        )


def test_replay_path_and_speed_validation():
    assert client.post("/replay", json={"log_file": "../plan.md"}).status_code == 422
    assert client.post("/replay", json={"log_file": "missing.jsonl"}).status_code == 404
    assert client.post("/replay", json={"log_file": "mock.jsonl", "speed": 0}).status_code == 422


def test_event_type_and_generated_schema_stay_consistent():
    event = client.get("/fixtures/events").json()[0]
    event["type"] = "node_created"
    with pytest.raises(ValidationError):
        Event.model_validate(event)
    node_event = client.get("/fixtures/events").json()[1]
    node_event["node_id"] = "wrong-node"
    with pytest.raises(ValidationError):
        Event.model_validate(node_event)
    schema = Path("contracts/schema.json").read_text(encoding="utf-8")
    assert json.loads(schema) == Contracts.model_json_schema()


def test_missing_report_and_unknown_websocket_run():
    class PendingAgent(FakeAgent):
        async def report(self, run_id):
            return None

    with TestClient(create_app(PendingAgent())) as api:
        run_id = api.post("/runs", json={"question": "Why?"}).json()["run_id"]
        assert api.get(f"/runs/{run_id}/report").status_code == 409
        with api.websocket_connect("/runs/unknown/events") as ws:
            assert ws.receive_json()["detail"]["code"] == "run_not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": "Why?", "mode": "unsupported"},
        {"question": "Why?", "budget": {"max_tests": 0}},
        {"question": "Why?", "unknown": True},
    ],
)
def test_invalid_run_requests(body):
    with TestClient(create_app(FakeAgent())) as api:
        assert api.post("/runs", json=body).status_code == 422


def test_openapi_exposes_all_planned_http_paths():
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/runs",
        "/runs/{run_id}/events",
        "/runs/{run_id}/interventions",
        "/runs/{run_id}/report",
        "/replay",
    }.issubset(paths)


def test_demo_websocket_pause_redirect_and_retained_history():
    with TestClient(create_app()) as api:
        result = api.post("/demo/runs", json={"question": "Investigate fatigue"})
        assert result.status_code == 202
        run_id = result.json()["run_id"]
        with api.websocket_connect(f"/runs/{run_id}/events") as ws:
            while True:
                event = ws.receive_json()
                if event.get("type") == "node_created":
                    original = event["node_id"]
                    break
            ws.send_json({"request_id": "pause-1", "command": {"type": "pause", "payload": {}}})
            messages = []
            while not any(m.get("type") == "command_ack" for m in messages):
                messages.append(ws.receive_json())
            assert messages[-1]["request_id"] == "pause-1"
            events = api.get(f"/runs/{run_id}/events").json()
            assert [e for e in events if e["type"] == "run_status"][-1]["payload"][
                "status"
            ] == "paused"
            ws.send_json(
                {
                    "request_id": "redirect-1",
                    "command": {
                        "type": "redirect",
                        "node_id": original,
                        "payload": {"statement": "Explore travel instead"},
                    },
                }
            )
            while True:
                message = ws.receive_json()
                if message.get("type") == "edge_created":
                    assert message["payload"]["edge"]["target"] == original
                    assert message["payload"]["edge"]["label"] == "Explore travel instead"
                    break
            events = api.get(f"/runs/{run_id}/events").json()
            assert any(
                e["type"] == "node_updated" and "pruned" in e["payload"]["node"]["flags"]
                for e in events
            )
            assert api.post("/demo/runs", json={"question": "Another run"}).status_code == 409
            ws.send_json({"request_id": "invalid-1", "command": {"type": "bogus"}})
            while True:
                message = ws.receive_json()
                if message.get("type") == "command_error":
                    assert message["request_id"] == "invalid-1"
                    break
