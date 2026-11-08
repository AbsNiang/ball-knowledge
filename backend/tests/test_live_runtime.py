"""Fail-closed configuration, data packaging, run isolation and hard execution budgets."""

import asyncio
import shutil
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors
from pydantic import BaseModel, ValidationError

from backend.agent.ports import RegistryTools
from backend.api.agent import AgentService, AgentUnavailableError, configured_run
from backend.api.main import create_app
from backend.api.runtime import runtime_info
from backend.data.readiness import FILES, _validate, validate_dataset
from backend.data.store import PROCESSED_DIR
from backend.llm.gateway import BudgetExceeded, NoToolCall
from backend.llm.gemini import GeminiGateway, shared_limiter
from backend.llm.redaction import safe_error
from backend.tools import real
from contracts.api import CreateRun


@pytest.fixture
def configured(monkeypatch):
    for name, value in {
        "LLM_MODE": "gemini",
        "WHACK_AGENT_TOOLS": "registry",
        "WHACK_TOOLS": "real",
        "GEMINI_API_KEY": "secret-test-key",
        "GEMINI_MODEL": "test-model",
        "GEMINI_RPM": "1000",
    }.items():
        monkeypatch.setenv(name, value)


def test_packaged_data_ready_and_runtime_contract(configured):
    with TestClient(create_app()) as api:
        result = api.get("/runtime")
        assert result.status_code == 200
        body = result.json()
        assert body["ready"] and body["execution_mode"] == "gemini_real"
        assert body["dataset_version"] == validate_dataset(PROCESSED_DIR)["dataset_version"]
        assert "2021\u201322" in body["historical_coverage"]
        assert "secret-test-key" not in result.text


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_MODE", "fake"),
        ("LLM_MODE", "bad"),
        ("WHACK_TOOLS", "mock"),
        ("WHACK_TOOLS", "auto"),
        ("WHACK_AGENT_TOOLS", "stub"),
        ("GEMINI_RPM", "0"),
        ("GEMINI_RPM", "bad"),
        ("GEMINI_API_KEY", ""),
        ("GEMINI_MODEL", ""),
    ],
)
def test_no_mock_fallback(configured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert not runtime_info().ready
    with pytest.raises(AgentUnavailableError):
        configured_run("unavailable", CreateRun(question="Why?"))


@pytest.mark.parametrize("failure", ["missing", "corrupt", "sealed_missing"])
def test_invalid_data_is_unavailable(configured, monkeypatch, tmp_path, failure):
    from backend.tools import data

    for name in (*FILES, "manifest.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROCESSED_DIR / name, target)
    if failure == "missing":
        (tmp_path / "team_games.parquet").unlink()
    elif failure == "sealed_missing":
        (tmp_path / "sealed/games.parquet").unlink()
    else:
        with (tmp_path / "games.parquet").open("ab") as out:
            out.write(b"corrupt")
    monkeypatch.setattr(data, "_data_dir", tmp_path)
    assert not runtime_info().ready
    assert "missing or invalid" in runtime_info().unavailable_reason


def test_readiness_does_not_deserialize_sealed(monkeypatch):
    import pandas as pd

    original = pd.read_parquet

    def checked(path, *args, **kwargs):
        assert "sealed" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", checked)
    _validate.cache_clear()
    assert validate_dataset(PROCESSED_DIR)


def test_server_caps_browser_budget(configured):
    run = configured_run(
        "limited", CreateRun(question="Why?", budget={"max_llm_calls": 999, "max_tests": 999})
    )
    assert run.context.llm.max_calls == 60
    assert run.context.tools.max_tests == 12
    assert run._initial["budget"]["max_tests"] == 12


def test_run_history_isolated_in_threads_and_gate(configured, monkeypatch):
    from backend.tests.test_tools_real import spec

    monkeypatch.setattr(real, "compare", lambda *a, **k: {"effect": 1.0, "extra": {}})
    first, second = RegistryTools(), RegistryTools()

    async def execute():
        return await asyncio.to_thread(first.run, "compare_groups", {"spec": spec()})

    result = asyncio.run(execute())
    assert result.status == "ok"
    assert first.state.hypotheses and not second.state.hypotheses
    assert (
        second.run("out_of_sample", {"test_id": result.extra["test_id"]}).reason == "unknown_test_id"
    )
    reused = spec(metric="margin")
    assert first.run("compare_groups", {"spec": reused}).reason == "hypothesis_already_registered"
    assert second.run("compare_groups", {"spec": reused}).status == "ok"
    first.state.holdout_used.add(spec()["id"])
    assert spec()["id"] not in second.state.holdout_used
    assert first.run("holdout_confirm", {"hypothesis_id": spec()["id"]}).reason == "already_used"


def test_test_budget_includes_failed_calls_and_gate_followups(configured):
    tools = RegistryTools(max_tests=2)
    assert tools.run("regress", {}).status == "error"
    assert tools.run("out_of_sample", {"test_id": "missing"}).status == "error"
    result = tools.run("regress", {})
    assert result.reason == "test_budget_exhausted" and tools.tests == 2


def test_holdout_reserves_all_three_computations(configured):
    tools = RegistryTools(max_tests=2)
    assert (
        tools.run("holdout_confirm", {"hypothesis_id": "missing"}).reason == "test_budget_exhausted"
    )
    assert not tools.state.holdout_used


def test_reused_session_identifier_rejected():
    async def check():
        service = AgentService()
        service.sessions["same"] = object()
        with pytest.raises(ValueError, match="already exists"):
            await service.start("same", CreateRun(question="Why?"), None)

    asyncio.run(check())


class Reply(BaseModel):
    answer: str


class FakeClient:
    def __init__(self, response=None, failure=None):
        self.count = 0
        self.response = response or SimpleNamespace(text='{"answer":"ok"}', function_calls=[])
        self.failure = failure
        self.aio = SimpleNamespace(models=self)

    async def generate_content(self, **kwargs):
        self.count += 1
        if self.failure:
            raise self.failure
        return self.response


def test_gateway_hard_logical_and_outbound_limits():
    async def check():
        client = FakeClient()
        gateway = GeminiGateway(api_key="unused", model_id="test", rpm=1000, client=client)
        for _ in range(60):
            await gateway.structured("test", Reply, system="", context={})
        with pytest.raises(BudgetExceeded):
            await gateway.search("query", system="")
        assert gateway.calls == client.count == 60
        gateway.calls = 0
        gateway.attempts = 120
        with pytest.raises(BudgetExceeded):
            await gateway.search("query", system="")
        assert client.count == 60
        assert shared_limiter(1000) is shared_limiter(1000)

    asyncio.run(check())


def test_gateway_bounded_retries_and_redaction(monkeypatch, configured):
    async def no_sleep(*args):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    async def check():
        client = FakeClient(
            failure=errors.ServerError(503, {"error": {"message": "secret-test-key"}})
        )
        gateway = GeminiGateway(api_key="unused", model_id="test", rpm=1000, client=client)
        with pytest.raises(RuntimeError, match="HTTP 503") as caught:
            await gateway.search("query", system="")
        assert "secret-test-key" not in str(caught.value)
        assert client.count == gateway.attempts == 4
        gateway.attempts = 119
        with pytest.raises(BudgetExceeded):
            await gateway.search("query", system="")
        assert gateway.attempts == 120 and client.count == 5

    asyncio.run(check())
    assert "secret-test-key" not in safe_error(RuntimeError("secret-test-key"))


def test_malformed_responses_remain_failures():
    async def check():
        gateway = GeminiGateway(
            api_key="unused",
            model_id="test",
            rpm=1000,
            client=FakeClient(SimpleNamespace(text="bad", function_calls=[])),
        )
        with pytest.raises(ValidationError):
            await gateway.structured("test", Reply, system="", context={})
        with pytest.raises(NoToolCall):
            await gateway.call_tool("test", [], system="", context={})
        assert gateway.calls == 2

    asyncio.run(check())


def test_sdk_cannot_retry_behind_gateway_budget(monkeypatch):
    from backend.llm import gemini
    captured = {}
    def construct(**kwargs):
        captured.update(kwargs)
        return FakeClient()
    monkeypatch.setattr(gemini.genai, "Client", construct)
    GeminiGateway(api_key="unused", model_id="test")
    assert captured["http_options"].retry_options.attempts == 1
    assert captured["http_options"].timeout == 60_000
