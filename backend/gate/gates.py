"""The stats gate: deterministic G1-G7 checks on tool results (plan.md v2 section 9.1). No LLM.

Which gates a result gets depends on what it is:

| Result                                                        | Gates              |
|---------------------------------------------------------------|--------------------|
| primary test: compare_groups, regress, persistence, or        | G1-G4              |
| market_test when it is the hypothesis's primary test          |                    |
| out_of_sample                                                 | G5                 |
| market_test as a follow-up                                    | G6                 |
| holdout_confirm                                               | G7                 |
| placebo, power and the data tools                             | none (diagnostics) |

- A gate whose input is missing fails and names the missing field. A tool error fails every
  gate the result would have had.
- G4 is provisional. `reevaluate()` reruns Benjamini-Hochberg over the final ledger and returns
  the results that flipped, for the agent to log and apply before the report.
- G6's second half (the effect survives `spread` as a control) runs `regress` through the same
  tool runner the agent uses. The call and its numbers go in the check's evidence, and its
  p-value is a ledger entry.
- "Survives" means the controlled effect's 95% CI excludes 0 in the declared direction (for a
  two-sided hypothesis, the direction of its primary effect).
- G7 cannot be overridden (`overridable`).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.gate import config
from backend.gate.ledger import Ledger
from backend.gate.verdict import Gates, GateStatus, compute_verdict, resolve_chance
from backend.tools.registry import dispatch
from backend.tools.schemas import ToolResult
from backend.tools.stats import Z_95, min_detectable_effect

GateId = Literal["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
RunTool = Callable[[str, dict[str, Any]], ToolResult]
Outcome = tuple[GateId, bool, str, list[str]]  # gate, passed, rationale, evidence

PRIMARY_TOOLS = frozenset({"compare_groups", "regress", "persistence", "market_test"})
_POINT_METRICS = frozenset({"ats_margin", "margin"})
_GATES_FOR: dict[str, tuple[GateId, ...]] = {
    "primary": ("G1", "G2", "G3", "G4"),
    "replication": ("G5",),
    "market": ("G6",),
    "holdout": ("G7",),
    "diagnostic": (),
}


class GateCheck(BaseModel):
    node_id: str
    gate_id: GateId
    status: GateStatus
    rationale: str
    evidence: list[str] = Field(default_factory=list)


def overridable(gate_id: str) -> bool:
    """plan.md section 9.4: a human may override G1-G6 with a note, never G7."""
    return gate_id != "G7"


def sesoi(tool: str, metric: str) -> float:
    """The smallest effect of interest, in the units the tool reports its effect in."""
    if tool == "persistence":
        return config.SESOI_CORRELATION
    return config.SESOI_POINTS if metric in _POINT_METRICS else config.SESOI_RATE


def role(tool: str, spec: Mapping[str, Any]) -> str:
    if tool == "out_of_sample":
        return "replication"
    if tool == "holdout_confirm":
        return "holdout"
    if tool == "market_test" and spec.get("primary_test") != "market_test":
        return "market"
    return "primary" if tool in PRIMARY_TOOLS else "diagnostic"


def excludes_zero(low: float | None, high: float | None, direction: str) -> bool:
    if low is None or high is None:
        return False
    if direction == "a_gt_b":
        return low > 0
    if direction == "a_lt_b":
        return high < 0
    return low > 0 or high < 0


def _fmt(value: Any) -> str:
    return f"{value:+.4g}" if isinstance(value, int | float) else "missing"


def _num(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, int | float) else "missing"


def _ci(low: Any, high: Any) -> str:
    return f"[{_fmt(low)}, {_fmt(high)}]"


class StatsGate:
    """One per run: it owns the run's ledger."""

    def __init__(self, run_tool: RunTool = dispatch) -> None:
        self._run_tool = run_tool
        self.ledger = Ledger()
        self._g4: dict[str, tuple[str, GateStatus]] = {}  # node id -> (ledger key, latest G4)
        self._effects: dict[str, float] = {}  # hypothesis id -> primary effect

    # -- the Gate port ------------------------------------------------------------------------

    def evaluate(
        self, node_id: str, result: ToolResult, spec: Mapping[str, Any]
    ) -> list[GateCheck]:
        kind = role(result.tool, spec)
        gate_ids = _GATES_FOR[kind]
        if not gate_ids:
            return []
        if result.status != "ok":
            why = f"Tool error: {result.reason}"
            return [
                GateCheck(node_id=node_id, gate_id=g, status="fail", rationale=why)
                for g in gate_ids
            ]
        run = {
            "primary": self._primary,
            "replication": self._replication,
            "market": self._market,
            "holdout": self._holdout,
        }[kind]
        return [
            GateCheck(
                node_id=node_id,
                gate_id=gate_id,
                status="pass" if passed else "fail",
                rationale=why,
                evidence=evidence,
            )
            for gate_id, passed, why, evidence in run(node_id, result, spec)
        ]

    def verdict(self, kind: str, gates: Gates, competing: Sequence[tuple[str, Gates]]) -> str:
        """`compute_verdict`, or the chance rule; `competing` is only read for chance."""
        return resolve_chance(competing) if kind == "chance" else compute_verdict(gates)

    def reevaluate(self) -> list[GateCheck]:
        """G4 over the final ledger: a new check for every primary test whose G4 flipped."""
        flips = []
        rejected = self.ledger.rejected()
        for node_id, (key, before) in self._g4.items():
            now: GateStatus = "pass" if key in rejected else "fail"
            if now == before:
                continue
            self._g4[node_id] = (key, now)
            flips.append(
                GateCheck(
                    node_id=node_id,
                    gate_id="G4",
                    status=now,
                    rationale=f"Re-evaluated over the final ledger of {len(self.ledger)} entries: "
                    f"p={self.ledger.p_value(key):.4g} is {'now' if now == 'pass' else 'no longer'} "
                    f"rejected by BH at q={config.BH_Q} ({before} -> {now}).",
                )
            )
        return flips

    # -- G1-G4: primary tests -----------------------------------------------------------------

    def _primary(self, node_id: str, result: ToolResult, spec: Mapping[str, Any]) -> list[Outcome]:
        if result.effect is not None and spec.get("id"):
            self._effects[spec["id"]] = result.effect
        key = self.ledger.add(result)
        outcomes = [self._power(result, spec), self._uncertainty(result, spec)]

        p = result.p_value
        g3 = p is not None and p < config.ALPHA
        outcomes.append(("G3", g3, f"p={_num(p)} vs alpha {config.ALPHA}", []))

        if key is None:
            outcomes.append(("G4", False, "No p-value, so no ledger entry.", []))
        else:
            g4 = self.ledger.rejects(key)
            self._g4[node_id] = (key, "pass" if g4 else "fail")
            why = (
                f"BH at q={config.BH_Q} over {len(self.ledger)} ledger entries: p={_num(p)} "
                f"{'rejected' if g4 else 'not rejected'}. Provisional until the report."
            )
            outcomes.append(("G4", g4, why, [f"ledger {key}"]))
        return outcomes

    def _power(self, result: ToolResult, spec: Mapping[str, Any]) -> Outcome:
        x = result.extra
        if result.tool in ("compare_groups", "regress"):
            n_a, n_b = x.get("n_a"), x.get("n_b")
            if n_a is None or n_b is None:
                return "G1", False, "The tool output has no n_a/n_b.", []
            smallest, sizes = min(n_a, n_b), f"n_a={n_a}, n_b={n_b}"
        else:
            if result.n is None:
                return "G1", False, "The tool output has no n.", []
            smallest, sizes = result.n, f"n={result.n}"
        mde = x.get("mde")
        if mde is None and result.ci_low is not None and result.ci_high is not None:
            mde = min_detectable_effect((result.ci_high - result.ci_low) / (2 * Z_95))
        if mde is None:
            return "G1", False, f"{sizes}; the tool output has no MDE or CI.", []
        bar = sesoi(result.tool, str(spec.get("metric")))
        passed = smallest >= config.N_MIN and mde <= bar
        why = f"{sizes} (need {config.N_MIN} each); MDE at 80% power {mde:.4g} vs SESOI {bar}"
        return "G1", passed, why, []

    def _uncertainty(self, result: ToolResult, spec: Mapping[str, Any]) -> Outcome:
        low, high = result.ci_low, result.ci_high
        if result.tool == "market_test":  # the CI is the bet's cover rate
            passed = low is not None and low > config.BREAKEVEN
            why = f"cover rate 95% CI {_ci(low, high)} must clear break-even {config.BREAKEVEN}"
            return "G2", passed, why, []
        direction = str(spec.get("direction", "two_sided"))
        passed = excludes_zero(low, high, direction)
        return "G2", passed, f"95% CI {_ci(low, high)} must exclude 0 ({direction})", []

    # -- G5-G7: follow-ups --------------------------------------------------------------------

    def _replication(
        self, _node: str, result: ToolResult, _spec: Mapping[str, Any]
    ) -> list[Outcome]:
        passed, why = _replicates(result.extra, "validation")
        return [("G5", passed, why, [])]

    def _market(self, _node: str, result: ToolResult, spec: Mapping[str, Any]) -> list[Outcome]:
        self.ledger.add(result)
        side = result.args.get("side", "?")
        low = result.ci_low
        rate_ok = low is not None and low > config.BREAKEVEN
        rate_why = f"{side} cover CI {_ci(low, result.ci_high)} vs break-even {config.BREAKEVEN}"

        controlled = self._spread_control(spec)
        self.ledger.add(controlled)
        if controlled.status != "ok":
            spread_ok, spread_why = False, f"spread-controlled regress failed: {controlled.reason}"
        else:
            spread_ok, spread_why = self._survives(controlled.ci_low, controlled.ci_high, spec)
        effect, ci = _fmt(controlled.effect), _ci(controlled.ci_low, controlled.ci_high)
        evidence = [f"regress {controlled.args_hash} with spread as a control: {effect}, CI {ci}"]
        return [("G6", rate_ok and spread_ok, f"{rate_why}; {spread_why}", evidence)]

    def _holdout(self, _node: str, result: ToolResult, spec: Mapping[str, Any]) -> list[Outcome]:
        x = result.extra
        replicated, replication_why = _replicates(x, "holdout")

        market = x.get("market") or {}
        low, high = market.get("ci_low"), market.get("ci_high")
        wanted = "opp" if spec.get("direction") == "a_lt_b" else "team"
        if market.get("error") or low is None or high is None:
            market_ok, market_why = (
                False,
                f"holdout market check unavailable ({market.get('error') or 'missing'})",
            )
        else:
            if market.get("side", "team") != wanted:  # holdout_confirm reports the team side
                low, high = 1 - high, 1 - low
            market_ok = low > config.BREAKEVEN
            market_why = (
                f"holdout {wanted} cover CI {_ci(low, high)} vs break-even {config.BREAKEVEN}"
            )

        controlled = x.get("spread_controlled") or {}
        if controlled.get("error") or controlled.get("ci_low") is None:
            spread_ok = False
            spread_why = (
                f"holdout spread control unavailable ({controlled.get('error') or 'missing'})"
            )
        else:
            spread_ok, spread_why = self._survives(
                controlled["ci_low"], controlled.get("ci_high"), spec
            )

        passed = replicated and market_ok and spread_ok
        return [("G7", passed, f"{replication_why}; {market_why}; {spread_why}", [])]

    def _spread_control(self, spec: Mapping[str, Any]) -> ToolResult:
        controls = list(dict.fromkeys([*spec.get("controls", []), "spread"]))
        return self._run_tool("regress", {"spec": {**spec, "controls": controls}})

    def _survives(
        self, low: float | None, high: float | None, spec: Mapping[str, Any]
    ) -> tuple[bool, str]:
        direction = str(spec.get("direction", "two_sided"))
        if direction == "two_sided":
            effect = self._effects.get(str(spec.get("id")))
            if effect:
                direction = "a_gt_b" if effect > 0 else "a_lt_b"
        passed = excludes_zero(low, high, direction)
        verb = "survives" if passed else "does not survive"
        return passed, f"effect {verb} spread as a control: 95% CI {_ci(low, high)} ({direction})"


def _replicates(extra: Mapping[str, Any], where: str) -> tuple[bool, str]:
    same, retention = extra.get("same_sign"), extra.get("retention")
    passed = same is True and retention is not None and retention >= config.RETENTION
    why = f"{where}: same_sign={same}, retention={_num(retention)} (need >= {config.RETENTION})"
    return passed, why
