from fastapi.testclient import TestClient

from backend.api.main import app
from contracts.events import Event

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
    assert events[-1].data["status"] == "completed"

