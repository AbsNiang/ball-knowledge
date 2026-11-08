"""Stateful router (https://docs.langchain.com/oss/python/langchain/multi-agent/router).

Every step runs through the router. It applies pauses and mode changes, then dispatches exactly
one step with `Command(goto=...)`; every step returns to the router.

The LLM roles make the choices (the supervisor decides when a seed is done, the researcher which
tool to call); the router enforces the rules no LLM can override:
- a decision queued for a human (approval mode) is reviewed before anything runs after it;
- a research step is paid for up front, so the budget never cuts one off halfway, and the final
  summary always keeps its reserve;
- a seed gets at most `MAX_STEPS_PER_SEED` steps;
- once the budget runs out, unfinished seeds get rule-based labels and the run summarizes.

State persists across invocations through the checkpointer, which lets approval mode pause on
`interrupt()` and resume later.
"""

import copy
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Command

from backend.agent.context import AgentContext
from backend.agent.record import Recorder
from backend.agent.state import (
    BUDGET_LABEL,
    MAX_STEPS_PER_SEED,
    REPORT_RESERVE,
    STEP_COST,
    AgentState,
    Budget,
    stopping_failures,
)
from contracts.api import ModePayload, SetMode

ROUTES = (
    "premise",
    "premise_test",
    "seed",
    "research",
    "execute",
    "judge",
    "supervise",
    "conclude",
    "summarize",
    "human_review",
)
# Worst-case LLM calls a phase commits to. Phases not listed are pre-paid or cost nothing.
PHASE_COST = {"premise": 2, "seed": 2, "research": STEP_COST}
_BEFORE_SEEDS = frozenset({"premise", "premise_test", "seed"})


def over_budget(budget: Budget) -> bool:
    llm_left = budget["max_llm_calls"] - budget["llm_calls"]
    return llm_left <= REPORT_RESERVE or budget["tests"] >= budget["max_tests"]


def affordable(budget: Budget, phase: str) -> bool:
    cost = PHASE_COST.get(phase, 0)
    return cost == 0 or budget["llm_calls"] + cost + REPORT_RESERVE <= budget["max_llm_calls"]


def exhaust(state: AgentState) -> dict[str, Any]:
    """Budget hit: stop researching (plan.md [A11]). A seed whose primary test already failed a
    stopping gate keeps the label that failure fixed; every other unfinished seed is
    `Inconclusive (budget)`. Before any seed exists, the run ends at the premise."""
    if state["phase"] in _BEFORE_SEEDS or not state["seeds"]:
        premise = copy.deepcopy(state["premise"])
        premise["label"] = premise["label"] or BUDGET_LABEL
        return {
            "exhausted": True,
            "premise": premise,
            "outcome": state["outcome"] or "budget",
            "phase": "summarize",
        }
    seeds = copy.deepcopy(state["seeds"])
    for seed in seeds.values():
        if seed["status"] in ("queued", "active") and not stopping_failures(seed["gates"]):
            seed["forced_label"] = BUDGET_LABEL
    return {"exhausted": True, "seeds": seeds, "phase": "conclude"}


def route(state: AgentState) -> tuple[dict[str, Any], str]:
    """Pick the next step. Returns (state update, node name)."""
    if state["approval"]:
        return {}, "human_review"
    phase, budget = state["phase"], state["budget"]
    unaffordable = over_budget(budget) or not affordable(budget, phase)
    if phase in PHASE_COST and not state["exhausted"] and unaffordable:
        update = exhaust(state)
        return update, update["phase"]
    current = state["seeds"].get(state["current"] or "")
    at_limit = current is not None and len(current["steps"]) >= MAX_STEPS_PER_SEED
    if phase in ("research", "supervise") and at_limit:  # no more research, so no need to ask
        return {"phase": "conclude"}, "conclude"
    return {}, phase


async def router(state: AgentState, runtime: Runtime[AgentContext]) -> Command:
    ctx = runtime.context
    # Pause between steps; let an in-flight tool or LLM call finish without losing evidence.
    rec = Recorder(ctx.log)
    if not ctx.running.is_set():
        rec.status("paused")
        await ctx.running.wait()

    mode = state["mode"]
    for message in ctx.take_inbox():
        if message.get("type") == "set_mode" and message.get("mode") in ("live", "approval"):
            rec.intervention(SetMode(type="set_mode", payload=ModePayload(mode=message["mode"])))
            if message["mode"] != mode:
                mode = message["mode"]
                rec.mode(mode)
    budget = Budget(**{**state["budget"], "llm_calls": ctx.llm.calls})
    update: dict[str, Any] = {"mode": mode, "budget": budget}
    if mode == "live" and state["approval"]:
        update["approval"] = []

    routed, goto = route({**state, **update})
    rec.budget(budget["llm_calls"], budget["tests"])
    return Command(update={**update, **routed}, goto=goto)
