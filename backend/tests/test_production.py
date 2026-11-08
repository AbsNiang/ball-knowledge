from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.production import create_production_app


def test_production_routes_and_websocket(tmp_path):
    (tmp_path / "index.html").write_text("<html>Whack</html>")
    (tmp_path / "app.js").write_text("console.log('Whack')")
    with TestClient(create_production_app(create_app(), tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.head("/").content == b""
        assert client.get("/app.js").status_code == 200
        assert client.get("/workspace", headers={"accept": "text/html"}).status_code == 200
        for path in ("/assets/missing", "/missing.js", "/api/missing"):
            assert client.get(path).status_code == 404
        assert client.get("/api/health").json() == {"status": "ok"}
        run = client.post("/api/demo/runs", json={"question": "Why?"}).json()["run_id"]
        with client.websocket_connect(f"/api/runs/{run}/events") as socket:
            assert socket.receive_json()["seq"] == 1


def test_production_forwards_lifespan(tmp_path):
    (tmp_path / "index.html").write_text("Whack")
    transitions = []

    @asynccontextmanager
    async def lifespan(_app):
        transitions.append("started")
        yield
        transitions.append("stopped")

    with TestClient(create_production_app(FastAPI(lifespan=lifespan), tmp_path)):
        assert transitions == ["started"]
    assert transitions == ["started", "stopped"]


def test_production_requires_frontend(tmp_path):
    with (
        pytest.raises(RuntimeError, match="Frontend build missing"),
        TestClient(create_production_app(create_app(), tmp_path)),
    ):
        pass


def test_production_rejects_invalid_inputs_and_preserves_cursor(tmp_path):
    """Exercise malformed input boundaries through the actual production mount."""
    import random

    (tmp_path / "index.html").write_text("Whack")
    rng = random.Random(42)
    with TestClient(create_production_app(create_app(), tmp_path)) as client:
        for _ in range(30):
            token = "".join(rng.choices("abcdef0123456789", k=24))
            assert client.get(f"/api/{token}").status_code == 404
            assert client.get(f"/assets/{token}").status_code == 404
        for body in ({}, {"question": ""}, {"question": "a" * 4001}, {"mode": "invalid"}):
            assert client.post("/api/demo/runs", json=body).status_code == 422
        run = client.post("/api/demo/runs", json={"question": "Cursor check"}).json()["run_id"]
        client.post(f"/api/runs/{run}/interventions", json={"type": "pause"})
        events = client.get(f"/api/runs/{run}/events").json()
        cursor = events[-1]["seq"]
        with client.websocket_connect(f"/api/runs/{run}/events?from_seq={cursor}") as socket:
            assert socket.receive_json() == events[-1]
            socket.send_json({"type": "ping"})
            assert socket.receive_json() == {"type": "pong"}
            socket.send_text("not json")
            assert socket.receive_json()["type"] == "command_error"
        for cursor in (0, -1):
            with client.websocket_connect(f"/api/runs/{run}/events?from_seq={cursor}") as socket:
                assert socket.receive_json()["detail"]["code"] == "invalid_cursor"
