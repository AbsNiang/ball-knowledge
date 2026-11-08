"""Development stand-ins so the research loop runs end to end with zero LLM quota.

- `b2b_script`: scripted answers for every LLM role (premise, search, seeds, researcher tool
  calls, judge, supervisor, summary) for the back-to-back question.
- `StubTools`: canned `ToolResult`s for that question. The numbers and search results are
  invented: they exercise the plumbing and say nothing about the NBA. Person A's real gate
  (`backend/gate`) judges them.
"""

import hashlib
import json
import random
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from backend.agent.ports import ToolResult
from backend.agent.schemas import (
    Aggregate,
    ClaimDecision,
    DeclaredTest,
    Filter,
    HypothesisSpec,
    JudgedClaim,
    Judgment,
    SeedIdea,
    SeedSet,
    SeedSummary,
    SliceSpec,
    Supervision,
)
from backend.agent.state import STOPPING_GATES
from backend.agent.toolkit import LEDGER_TOOLS
from backend.llm.gateway import SearchHit, SearchResult
from backend.tools.hashing import hash_args

B2B_QUESTION = "Why do teams on the second night of a back-to-back underperform?"
CONFIRMATORY_ORDER = (
    "declare_primary_test",
    "run_primary_test",
    "replicate_primary",
    "market_check",
    "holdout_confirm",
)


def _slice(*filters: tuple[str, str, Any]) -> SliceSpec:
    return SliceSpec(filters=[Filter(col=c, op=op, value=v) for c, op, v in filters])


B2B = ("b2b", "eq", True)
B2B_CLAIM = HypothesisSpec(
    id="b2b_gap",
    kind="premise",
    metric="ats_margin",
    direction="a_lt_b",
    statement="Teams on the 2nd night of a back-to-back have a lower ATS margin",
    group_a=_slice(B2B),
    primary_test="compare_groups",
)
B2B_SOURCES = [
    SearchHit(title="Synthetic: travel and back-to-backs", url="https://example.com/b2b-travel"),
    SearchHit(title="Synthetic: rest advantage", url="https://example.com/rest-advantage"),
]
B2B_SEEDS = [
    SeedIdea(
        id="travel_fatigue",
        kind="explanation",
        statement="Travel fatigue: the gap is larger after a long trip into the 2nd game",
        rationale="Reporting links back-to-back slumps to overnight travel.",
        sources=[B2B_SOURCES[0].url],
        testable_with_data=True,
    ),
    SeedIdea(
        id="rested_opponent",
        kind="confounder",
        statement="Confounder: back-to-back teams more often face a rested opponent",
        rationale="If the opponent is rested, the gap may be about rest difference, not fatigue.",
        sources=[B2B_SOURCES[1].url],
        testable_with_data=True,
    ),
    SeedIdea(
        id="altitude",
        kind="explanation",
        statement="Playing the 2nd night at altitude (Denver, Utah) drives the gap",
        rationale="Own reasoning: altitude compounds fatigue. Few games, so likely underpowered.",
        testable_with_data=True,
    ),
]
# The primary test each seed declares (its id is set by the system).
B2B_TESTS: dict[str, DeclaredTest] = {
    "travel_fatigue": DeclaredTest(
        statement=B2B_SEEDS[0].statement,
        kind="explanation",
        metric="ats_margin",
        direction="a_lt_b",
        group_a=_slice(B2B, ("travel_km", "gte", 1000)),
        group_b=_slice(B2B, ("travel_km", "lte", 999)),
        primary_test="compare_groups",
    ),
    "rested_opponent": DeclaredTest(
        statement=B2B_SEEDS[1].statement,
        kind="confounder",
        metric="ats_margin",
        direction="a_lt_b",
        group_a=_slice(B2B),
        primary_test="regress",
        controls=["opp_rest_days"],
    ),
    "altitude": DeclaredTest(
        statement=B2B_SEEDS[2].statement,
        kind="explanation",
        metric="ats_margin",
        direction="a_lt_b",
        group_a=_slice(B2B, ("altitude_m", "gte", 1200)),
        group_b=_slice(B2B, ("altitude_m", "lte", 1199)),
        primary_test="compare_groups",
    ),
}


def b2b_script(decision: str, schema: type[BaseModel], context: dict[str, Any]) -> Any:
    """Scripted answers for `FakeLLM`. Wrap it to inject other answers in tests."""
    match decision:
        case "premise":
            return ClaimDecision(
                testable=True,
                claim=B2B_CLAIM,
                rationale="'Underperform' read as ATS margin: more power than cover rate, and it "
                "measures performance against the closing line.",
            )
        case "search":
            return SearchResult(
                query=context["query"],
                summary="Synthetic search: coverage blames overnight travel and rested "
                "opponents for back-to-back slumps.",
                sources=B2B_SOURCES,
            )
        case "seed":
            return SeedSet(
                seeds=B2B_SEEDS,
                rationale="A travel mechanism and a rested-opponent confounder from the search, "
                "and a small-n altitude idea from reasoning.",
            )
        case "research":
            return _research(context)
        case "judge":
            return _judgment(context)
        case "supervise":
            return _supervision(context)
        case "summarize":
            seeds = context["seeds"]
            lines = [f"{s['verdict']}: {s['statement']}" for s in seeds]
            return Aggregate(
                answer="; ".join(lines) or "No seed was researched.",
                seeds=[SeedSummary(seed_id=s["seed_id"], summary=s["verdict"]) for s in seeds],
                rationale="Summary of the verdict table.",
            )
    raise ValueError(f"no scripted answer for decision {decision!r}")


def _done(context: dict[str, Any]) -> list[str]:
    return [step["tool"] for step in context["steps"]]


def _wants_placebo(context: dict[str, Any]) -> bool:
    done = _done(context)
    return (
        context["seed"]["kind"] == "explanation"
        and "replicate_primary" in done
        and "placebo" not in done
    )


def _wants_search(context: dict[str, Any]) -> bool:
    return context["seed"]["id"] == "travel_fatigue" and "web_search" not in _done(context)


def _research(context: dict[str, Any]) -> dict[str, Any]:
    """Declare, test, replicate, a placebo for mechanisms, the market, the holdout, then read."""
    seed_id, offered = context["seed"]["id"], context["offered_tools"]
    if "declare_primary_test" in offered:
        test = B2B_TESTS.get(seed_id) or B2B_TESTS["travel_fatigue"]
        return {
            "tool": "declare_primary_test",
            "args": {"test": test.model_dump(mode="json")},
            "reason": "Pre-register the test that decides this seed.",
        }
    for tool in ("run_primary_test", "replicate_primary"):
        if tool in offered:
            return {"tool": tool, "args": {}, "reason": f"Next confirmatory step: {tool}."}
    if _wants_placebo(context):
        group_a = context["primary_test"]["group_a"]
        filters = [{**f, "value": False} if f["col"] == "b2b" else f for f in group_a["filters"]]
        return {
            "tool": "placebo",
            "args": {
                "test_id": context["primary_test_id"],
                "placebo_slice": {**group_a, "filters": filters},
            },
            "reason": "Without the back-to-back the gap should vanish if this is the cause.",
        }
    for tool in ("market_check", "holdout_confirm"):
        if tool in offered:
            return {"tool": tool, "args": {}, "reason": f"Next confirmatory step: {tool}."}
    if _wants_search(context):
        return {
            "tool": "web_search",
            "args": {"query": "NBA back-to-back travel fatigue evidence"},
            "reason": "See what reporting says about the mechanism.",
        }
    return {
        "tool": "web_search",
        "args": {"query": context["seed"]["statement"]},
        "reason": "More.",
    }


def _judgment(context: dict[str, Any]) -> Judgment:
    step, gates = context["step"], context["verdict_gates"] or {}
    failed = [g for g in STOPPING_GATES if gates.get(g) == "fail"]
    held = all(gates.get(g) == "pass" for g in ("G1", "G2", "G3", "G4"))
    if "G1" in failed or not (failed or held):
        stance = "insufficient"  # underpowered or not yet tested: not evidence against it
    else:
        stance = "refuted" if failed else "supported"
    cited = step.get("result_id") or step.get("id")
    return Judgment(
        step_review="sound",
        step_rationale=f"{step['tool']} bears directly on the seed.",
        synthesis=f"Verdict gates so far: {gates or 'none yet'}.",
        stance=stance,
        confidence="medium",
        claims=[JudgedClaim(text=f"{step['tool']} result recorded", evidence=[cited])],
        self_check="Synthetic numbers; nothing cuts against the gate outcomes.",
    )


def _supervision(context: dict[str, Any]) -> Supervision:
    offered = context["tools_offered_next"]
    pending = [tool for tool in CONFIRMATORY_ORDER if tool in offered]
    if context["feedback"]:
        guidance = f"Fix the last call: {context['feedback']}"
    elif pending:
        guidance = f"Next: {pending[0]}."
    elif _wants_placebo(context):
        guidance = "Run a placebo where the effect should vanish."
    elif _wants_search(context):
        guidance = "Check what reporting says about the mechanism."
    else:
        return Supervision(action="conclude", rationale="No remaining step can change the verdict.")
    return Supervision(action="continue", guidance=guidance, rationale="More to establish.")


# (hypothesis id, tool) -> canned numbers. ats_margin effects are in points.
B2B_RESULTS: dict[tuple[str, str], dict[str, Any]] = {
    ("b2b_gap", "compare_groups"): {
        "n": 36730,
        "effect": -1.21,
        "ci_low": -1.68,
        "ci_high": -0.74,
        "p_value": 0.0002,
        "extra": {"n_a": 4810, "n_b": 31920, "mde": 0.42},
    },
    ("travel_fatigue", "compare_groups"): {
        "n": 4810,
        "effect": -1.52,
        "ci_low": -2.41,
        "ci_high": -0.63,
        "p_value": 0.0011,
        "extra": {"n_a": 1650, "n_b": 3160, "mde": 0.88},
    },
    ("travel_fatigue", "out_of_sample"): {
        "n": 1680,
        "effect": -1.07,
        "ci_low": -2.30,
        "ci_high": 0.16,
        "p_value": 0.09,
        "extra": {"same_sign": True, "retention": 0.70},
    },
    ("travel_fatigue", "market_test"): {
        "n": 1650,
        "effect": 0.521,
        "ci_low": 0.497,
        "ci_high": 0.545,
        "p_value": 0.44,
        "extra": {"side": "opp", "roi": -0.005, "breakeven": 0.5238},
    },
    ("rested_opponent", "regress"): {
        "n": 36730,
        "effect": -0.38,
        "ci_low": -1.02,
        "ci_high": 0.26,
        "p_value": 0.24,
        "extra": {"n_a": 4810, "n_b": 31920, "mde": 0.45},
    },
    ("altitude", "compare_groups"): {
        "n": 4810,
        "effect": -2.64,
        "ci_low": -5.12,
        "ci_high": -0.16,
        "p_value": 0.037,
        "extra": {"n_a": 141, "n_b": 4669, "mde": 2.95},
    },
    ("travel_fatigue", "placebo"): {
        "n": 9800,
        "effect": -0.21,
        "ci_low": -0.95,
        "ci_high": 0.53,
        "p_value": 0.58,
    },
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class StubTools:
    """Canned results keyed by (hypothesis id, tool). Takes Person A's argument shapes."""

    def __init__(self, results: Mapping[tuple[str, str], dict[str, Any]] = B2B_RESULTS) -> None:
        self._results = results
        self._tests: dict[str, str] = {}  # test_id -> hypothesis id
        self._slices: dict[str, str] = {}  # group_a hash -> hypothesis id (for market_test)
        self._holdout_used: set[str] = set()

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        if tool in ("describe_dataset", "define_slice", "power"):
            return self._data(tool, args)
        if tool == "holdout_confirm":
            key = args["hypothesis_id"]
            if key in self._holdout_used:
                return ToolResult(
                    tool=tool,
                    args=args,
                    status="error",
                    reason="already_used",
                    args_hash=hash_args(args),
                )
            self._holdout_used.add(key)
        elif "spec" in args:
            key = args["spec"]["id"]
            self._slices[_hash(args["spec"]["group_a"])] = key
        elif tool == "market_test":
            key = self._slices.get(_hash(args["slice"]), "?")
        else:
            key = self._tests.get(args.get("test_id"), "?")
        canned = self._results.get((key, tool)) or self._fallback(tool, args)
        result = ToolResult(
            tool=tool,
            args=args,
            args_hash=hash_args(args),
            slice_hash=_hash(args.get("spec", args))[:16],
            **{**canned, "extra": dict(canned.get("extra", {}))},
        )
        if tool in LEDGER_TOOLS and "spec" in args:
            test_id = "t_" + _hash(args)[:10]
            result.extra["test_id"] = test_id
            self._tests[test_id] = key
        return result

    @staticmethod
    def _data(tool: str, args: dict[str, Any]) -> ToolResult:
        rng = random.Random(_hash([tool, args]))
        extra: dict[str, Any] = {}
        n = None
        if tool == "define_slice":
            n = rng.randint(400, 9000)
            extra = {"n_rows": n, "n_games": n // 2}
        elif tool == "power":
            extra = {"mde": 0.5}
        else:
            extra = {"seasons": "2007-08 to 2023-24 (synthetic)", "rows": 36730}
        return ToolResult(tool=tool, args=args, args_hash=hash_args(args), n=n, extra=extra)

    @staticmethod
    def _fallback(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Deterministic weak results for anything unscripted."""
        rng = random.Random(_hash([tool, args]))
        effect = round(rng.uniform(-0.6, 0.6), 2)
        if tool == "market_test":
            return {"n": 1000, "effect": 0.50, "ci_low": 0.47, "ci_high": 0.53, "p_value": 0.9}
        extra = {"same_sign": False, "retention": 0.2} if tool != "compare_groups" else {}
        extra |= {"n_a": 1000, "n_b": 1000, "mde": 0.5}
        return {
            "n": 2000,
            "effect": effect,
            "ci_low": effect - 0.8,
            "ci_high": effect + 0.8,
            "p_value": round(rng.uniform(0.2, 0.9), 3),
            "extra": extra,
        }
