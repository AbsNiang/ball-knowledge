"""`compute_verdict`: the label as a pure function of gate results (plan.md v2 section 9.2).

First matching row wins:

| Condition                            | Label                                 |
|--------------------------------------|---------------------------------------|
| G1 fails                             | Inconclusive                          |
| any of G2-G5 fails                   | Luck                                  |
| G1-G4 pass, G5 never ran             | Inconclusive (not replicated)         |
| G1-G5 pass, G6 fails or never ran    | Real but priced in                    |
| G1-G6 pass, G7 fails or never ran    | Real but priced in (G7 fail: holdout_failed) |
| G1-G7 pass                           | Real and beats market (provisional)   |

"Inconclusive (not replicated)" is not in the plan's table, which assumes G5 always runs. The
agent can conclude before replicating, and a result that was never replicated can't be Real.
The premise rules (Premise false, Untestable) and the budget label are applied by the agent.

Chance hypotheses have no test of their own and are resolved by rule (plan.md [A6]).
"""

from collections.abc import Mapping, Sequence
from typing import Literal

GateStatus = Literal["pass", "fail"]
Gates = Mapping[str, GateStatus]

INCONCLUSIVE = "Inconclusive"
NOT_REPLICATED = "Inconclusive (not replicated)"
LUCK = "Luck"
PRICED_IN = "Real but priced in"
REAL = "Real and beats market (provisional)"
CHANCE_SUPPORTED = "Chance supported"
CHANCE_REFUTED = "Chance refuted"
CHANCE_UNDETERMINED = "Chance undetermined"

_SIGNAL_GATES = ("G2", "G3", "G4", "G5")


def compute_verdict(gates: Gates) -> str:
    if gates.get("G1") != "pass":
        return INCONCLUSIVE
    if any(gates.get(g) == "fail" for g in _SIGNAL_GATES):
        return LUCK
    if any(gates.get(g) != "pass" for g in ("G2", "G3", "G4")):
        return LUCK  # a primary test always gets G1-G4; a missing one is not a pass
    if "G5" not in gates:
        return NOT_REPLICATED
    if gates.get("G6") != "pass" or gates.get("G7") != "pass":
        return PRICED_IN
    return REAL


def resolve_chance(competing: Sequence[tuple[str, Gates]]) -> str:
    """`competing` is (kind, gates) for the premise and every non-chance hypothesis.

    Supported if any of them fails G2-G5; refuted if every substantive hypothesis passes G2-G5;
    otherwise (some never reached G5) undetermined.
    """
    if any(gates.get(g) == "fail" for _, gates in competing for g in _SIGNAL_GATES):
        return CHANCE_SUPPORTED
    substantive = [gates for kind, gates in competing if kind not in ("premise", "chance")]
    if substantive and all(gates.get(g) == "pass" for gates in substantive for g in _SIGNAL_GATES):
        return CHANCE_REFUTED
    return CHANCE_UNDETERMINED
