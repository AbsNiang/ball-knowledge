"""Explicit integration-test app. Never selected by the production entrypoint."""

from backend.agent.graph import AgentRun
from backend.agent.stubs import StubTools, b2b_script
from backend.api.agent import AgentService
from backend.api.main import create_app
from backend.gate.gates import StatsGate
from backend.llm.fake import FakeLLM


def factory(run_id, request):
    tools = StubTools()
    return AgentRun(
        request.question,
        run_id=run_id,
        mode=request.mode,
        llm=FakeLLM(b2b_script),
        tools=tools,
        gate=StatsGate(tools.run),
    )


app = create_app(AgentService(factory))
