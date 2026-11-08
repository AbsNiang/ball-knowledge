"""The research loop as a LangGraph stateful router.

    START -> intake -> router <-> {premise, premise_test, seed, research, execute, judge,
                                   supervise, conclude, human_review}
                       router  -> summarize -> END

One run: the premise is checked first; a web search seeds up to three ideas; each seed is
researched in turn (research -> execute -> judge -> supervise) until the supervisor concludes
it; then every seed is aggregated into the answer. `AgentRun` owns one run: its event log and
its checkpointed LangGraph thread.
"""

import uuid
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from backend.agent.context import AgentContext
from backend.agent.ports import Gate, Tools
from backend.agent.record import Recorder
from backend.agent.router import ROUTES, router
from backend.agent.specialists import SPECIALISTS, intake
from backend.agent.state import AgentState, Mode, initial_state
from backend.events.store import EventLog
from backend.llm.gateway import LLMGateway
from backend.llm.redaction import safe_error
from contracts.events import RunCreated
from contracts.models import Budget

# Each router hop is two supersteps; budgets bound the LLM work, this bounds everything else.
RECURSION_LIMIT = 2_000


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("intake", intake)
    builder.add_node("router", router, destinations=ROUTES)
    for name, node in SPECIALISTS.items():
        builder.add_node(name, node)
        builder.add_edge(name, END if name == "summarize" else "router")
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "router")
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


class AgentRun:
    def __init__(
        self,
        question: str,
        *,
        llm: LLMGateway,
        tools: Tools,
        gate: Gate,
        mode: Mode = "live",
        budget: dict[str, int] | None = None,
        run_id: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.log = EventLog(self.run_id)
        self.context = AgentContext(llm=llm, tools=tools, gate=gate, log=self.log)
        self.graph = build_graph(checkpointer)
        self.config = {
            "configurable": {"thread_id": self.run_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        self._initial = initial_state(self.run_id, question, mode, budget)
        self.pending_review: dict[str, Any] | None = None

    async def start(self) -> dict[str, Any] | None:
        """Run until the answer is written, or until a decision needs human review.

        Standalone runs record `run_created` themselves; through the API, C's server does.
        """
        limits = self._initial["budget"]
        created = RunCreated(
            question=self._initial["question"],
            mode=self._initial["mode"],
            budget=Budget(max_llm_calls=limits["max_llm_calls"], max_tests=limits["max_tests"]),
        )
        self.log.append("system", created)
        return await self._invoke(self._initial)

    async def resume(self, action: str, note: str | None = None) -> dict[str, Any] | None:
        """Answer the pending review ('approve' or 'reject' with a 10+ character note)."""
        return await self._invoke(Command(resume={"action": action, "note": note}))

    def set_mode(self, mode: Mode) -> None:
        """Takes effect at the router's next step (plan.md §9.5)."""
        self.context.inbox.append({"type": "set_mode", "mode": mode})

    @property
    def state(self) -> AgentState:
        return self.graph.get_state(self.config).values

    async def _invoke(self, payload: Any) -> dict[str, Any] | None:
        try:
            out = await self.graph.ainvoke(payload, self.config, context=self.context)
        except Exception as exc:
            rec = Recorder(self.log)
            rec.error("agent_error", safe_error(exc))
            rec.status("failed")
            raise
        interrupts = out.get("__interrupt__") or []
        self.pending_review = interrupts[0].value if interrupts else None
        return self.pending_review
