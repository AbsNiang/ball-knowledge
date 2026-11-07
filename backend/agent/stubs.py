"""Development stand-ins so the agent loop runs end to end with zero LLM quota.

- `b2b_script`: scripted LLM decisions (and an approving critic) for the back-to-back question.
- `StubTools` / `StubGate`: canned `ToolResult`s and a toy G1-G7 + verdict table, until Person A's
  `backend/tools` and `backend/gate` land. The numbers are invented: they exercise the plumbing
  and say nothing about the NBA.
"""

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from backend.agent.ports import CheckResult, GateStatus, ToolResult
from backend.agent.schemas import (
    ClaimDecision,
    CriticChecklist,
    CriticReview,
    Filter,
    HypothesisRevision,
    HypothesisSet,
    HypothesisSpec,
    Interpretation,
    Narrative,
    PlanDecision,
    ReportDraft,
    SliceSpec,
)
from backend.agent.state import LEDGER_TOOLS
from backend.tools.hashing import hash_args

B2B_QUESTION = "Why do teams on the second night of a back-to-back underperform?"
NOT_REPLICATED = "Inconclusive (not replicated)"


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
B2B_HYPOTHESES = [
    HypothesisSpec(
        id="travel_fatigue",
        kind="explanation",
        metric="ats_margin",
        direction="a_lt_b",
        statement="Travel fatigue: the gap is larger after a long trip into the 2nd game",
        group_a=_slice(B2B, ("travel_km", "gte", 1000)),
        group_b=_slice(B2B, ("travel_km", "lte", 999)),
        primary_test="compare_groups",
    ),
    HypothesisSpec(
        id="rested_opponent",
        kind="confounder",
        metric="ats_margin",
        direction="a_lt_b",
        statement="Confounder: back-to-back teams more often face a rested opponent",
        group_a=_slice(B2B),
        primary_test="regress",
        controls=["opp_rest_days"],
    ),
    HypothesisSpec(
        id="altitude",
        kind="explanation",
        metric="ats_margin",
        direction="a_lt_b",
        statement="Playing the 2nd night at altitude (Denver, Utah) drives the gap",
        group_a=_slice(B2B, ("altitude_m", "gte", 1200)),
        group_b=_slice(B2B, ("altitude_m", "lte", 1199)),
        primary_test="compare_groups",
    ),
    HypothesisSpec(
        id="chance",
        kind="chance",
        metric="ats_margin",
        direction="two_sided",
        statement="The gap is noise from testing many seasons and slices",
        group_a=_slice(B2B),
        primary_test="compare_groups",
    ),
]


def b2b_script(decision: str, schema: type[BaseModel], context: dict[str, Any]) -> BaseModel:
    """Scripted answers for `FakeLLM`. Wrap it to inject critic objections in tests."""
    match decision:
        case "premise":
            return ClaimDecision(
                testable=True,
                claim=B2B_CLAIM,
                rationale="'Underperform' read as ATS margin: more power than cover rate, and it "
                "measures performance against the closing line.",
            )
        case "hypothesize":
            return HypothesisSet(
                hypotheses=B2B_HYPOTHESES,
                rationale="A travel mechanism, a rested-opponent confounder, a small-n altitude "
                "mechanism, and the chance baseline.",
            )
        case "revise_hypothesis":
            return HypothesisRevision(
                hypothesis=context["previous"], rationale=f"Answered: {context['feedback']}"
            )
        case "plan_test":
            spec = context["hypothesis"]
            return PlanDecision(
                tool=spec["primary_test"],
                controls=spec["controls"],
                rationale=f"{spec['primary_test']} on {spec['metric']} tests the declared "
                f"direction {spec['direction']}.",
            )
        case "choose_test":
            return _next_follow_up(context["hypothesis"], context["menu"])
        case "interpret":
            gates = context["gates"]
            held = sorted(g for g, v in gates.items() if v == "pass")
            missed = sorted(g for g, v in gates.items() if v != "pass")
            return Interpretation(
                summary=f"Passed {', '.join(held) or 'none'}; failed {', '.join(missed) or 'none'}",
                claims_made=[f"{g} passed" for g in held],
                limitations=["Stub numbers."],
                cited_checks=sorted(gates),
                rationale="Reads the gate outcomes only.",
            )
        case "narrate_verdict":
            return Narrative(
                text=f"{context['verdict']}: {context['hypothesis']}.",
                rationale="Restates the rule-based label.",
            )
        case "report":
            lines = [f"{v['verdict']}: {v['statement']}" for v in context["verdicts"]]
            return ReportDraft(answer="; ".join(lines) or "No verdicts.", rationale="Summary.")
        case "critic":
            return CriticReview(
                verdict="approve",
                checklist=CriticChecklist(),
                severity="low",
                evidence=[context["node_id"]],
                rationale="No checklist problems found.",
            )
    raise ValueError(f"no scripted answer for decision {decision!r}")


def _next_follow_up(spec: dict[str, Any], menu: Mapping[str, str]) -> dict[str, Any]:
    """Replicate first, then a placebo for mechanisms, then the market, then the holdout."""
    if "out_of_sample" in menu:
        return {"action": "out_of_sample", "rationale": "Replicate before anything else."}
    if "placebo" in menu and spec["kind"] == "explanation":
        filters = [
            {**f, "value": False} if f["col"] == "b2b" else f for f in spec["group_a"]["filters"]
        ]
        return {
            "action": "placebo",
            "placebo_slice": {**spec["group_a"], "filters": filters},
            "rationale": "Without the back-to-back the gap should vanish if this is the cause.",
        }
    if "market_test" in menu:
        return {"action": "market_test", "rationale": "Replicated; is it already in the line?"}
    if "holdout_confirm" in menu:
        return {"action": "holdout_confirm", "rationale": "G1-G6 passed; confirm on the holdout."}
    return {"action": "conclude", "rationale": "No remaining test would change the verdict."}


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


ALPHA, BH_Q, N_MIN, RETENTION, BREAKEVEN = 0.05, 0.05, 200, 0.5, 0.5238
SESOI = {"ats_margin": 1.0, "margin": 1.0}  # binary and ROI metrics: 0.03


class StubGate:
    """Toy version of plan.md §9.1-9.2. G4 is provisional BH over this run's ledger."""

    def __init__(self) -> None:
        self._ledger: dict[str, float] = {}

    def evaluate(
        self, node_id: str, result: ToolResult, spec: Mapping[str, Any]
    ) -> list[CheckResult]:
        def check(gate_id: str, ok: bool, why: str) -> CheckResult:
            return CheckResult(
                node_id=node_id,
                checker="gate",
                gate_id=gate_id,
                status="pass" if ok else "fail",
                rationale=why,
            )

        primary = result.tool in ("compare_groups", "regress", "persistence")
        gate_ids = {"out_of_sample": ["G5"], "market_test": ["G6"], "holdout_confirm": ["G7"]}
        gate_ids = gate_ids.get(result.tool, ["G1", "G2", "G3", "G4"] if primary else [])
        if result.status == "error":
            return [check(g, False, f"Tool error: {result.reason}") for g in gate_ids or ["error"]]

        x = result.extra
        if result.tool in LEDGER_TOOLS:
            self._ledger[x.get("test_id", node_id)] = result.p_value
        if primary:
            n_a, n_b, mde = x.get("n_a", 0), x.get("n_b", 0), x.get("mde", float("inf"))
            sesoi = SESOI.get(spec["metric"], 0.03)
            sign = {"a_lt_b": result.effect < 0, "a_gt_b": result.effect > 0}
            ci = f"[{result.ci_low:+.2f}, {result.ci_high:+.2f}]"
            return [
                check(
                    "G1",
                    min(n_a, n_b) >= N_MIN and mde <= sesoi,
                    f"n_a={n_a}, n_b={n_b}; MDE {mde:.2f} vs SESOI {sesoi}",
                ),
                check(
                    "G2",
                    (result.ci_low > 0 or result.ci_high < 0) and sign.get(spec["direction"], True),
                    f"95% CI {ci}, declared {spec['direction']}",
                ),
                check("G3", result.p_value < ALPHA, f"p={result.p_value:.4f} vs {ALPHA}"),
                check(
                    "G4",
                    self._bh_rejects(x.get("test_id", node_id)),
                    f"BH at q={BH_Q} over {len(self._ledger)} ledger entries",
                ),
            ]
        if result.tool in ("out_of_sample", "holdout_confirm"):
            ok = bool(x.get("same_sign")) and x.get("retention", 0) >= RETENTION
            why = f"same_sign={x.get('same_sign')}, retention={x.get('retention')}"
            return [check(gate_ids[0], ok, why)]
        if result.tool == "market_test":
            return [
                check(
                    "G6",
                    result.ci_low > BREAKEVEN,
                    f"cover CI low {result.ci_low:.3f} vs break-even {BREAKEVEN}",
                )
            ]
        return []

    def _bh_rejects(self, key: str) -> bool:
        ps = sorted(self._ledger.values())
        m = len(ps)
        cutoff = max((p for k, p in enumerate(ps, 1) if p <= k / m * BH_Q), default=-1.0)
        return self._ledger[key] <= cutoff

    def verdict(
        self,
        kind: str,
        gates: Mapping[str, GateStatus],
        competing: Sequence[Mapping[str, GateStatus]],
    ) -> str:
        if kind == "chance":
            # plan.md [A6]: supported if the premise or any competitor fails G2-G5.
            lost = any(g.get(x) == "fail" for g in competing for x in ("G2", "G3", "G4", "G5"))
            return "Chance supported" if lost else "Chance refuted"
        if gates.get("G1") != "pass":
            return "Inconclusive"
        if any(gates.get(g) != "pass" for g in ("G2", "G3", "G4")) or gates.get("G5") == "fail":
            return "Luck"
        if "G5" not in gates:  # the orchestrator concluded before replicating
            return NOT_REPLICATED
        if gates.get("G6") != "pass" or gates.get("G7") != "pass":
            return "Real but priced in"
        return "Real and beats market (provisional)"
