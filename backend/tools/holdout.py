"""`holdout_confirm`: the only code allowed to read the sealed holdout seasons.

It runs a hypothesis's frozen primary test once on the holdout. For G7 it also returns what the
G6 rule needs on the holdout: the slice's cover rate against break-even, and the effect with
`spread` as a control.
"""

from backend.data.store import read_sealed_holdout
from backend.tools import data, real
from backend.tools.hashing import hash_args
from backend.tools.schemas import HoldoutConfirmArgs, ToolResult
from backend.tools.state import current

_SUPPORTED = ("compare_groups", "regress", "market_test")


def _optional(compute):
    try:
        return compute()
    except real.ToolFailure as failure:
        return {"error": failure.reason}


@real.guarded("holdout_confirm")
def holdout_confirm(args: HoldoutConfirmArgs, raw: dict) -> ToolResult:
    spec = current().hypotheses.get(args.hypothesis_id)
    if spec is None:
        raise real.ToolFailure("unknown_hypothesis")
    if args.hypothesis_id in current().holdout_used:
        raise real.ToolFailure("already_used")
    if spec.primary_test not in _SUPPORTED:
        raise real.ToolFailure("not_supported_on_holdout", primary_test=spec.primary_test)

    frame = read_sealed_holdout(data.data_dir())
    frame["split"] = "holdout"
    current().holdout_used.add(args.hypothesis_id)  # used once the sealed rows have been read
    rng = real.rng_for(hash_args(raw))

    if spec.primary_test == "compare_groups":
        out = real.compare(frame, spec, rng, use_split=False)
    elif spec.primary_test == "regress":
        out = real.regression(frame, spec, use_split=False)
    else:
        out = real.market_stats(frame, spec.group_a, "team", rng, use_split=False)

    market = _optional(lambda: real.market_stats(frame, spec.group_a, "team", rng, use_split=False))
    with_spread = spec.model_copy(update={"controls": [*spec.controls, "spread"]})
    controlled = _optional(lambda: real.regression(frame, with_spread, use_split=False))

    explore_effect = real.latest_primary_effect(args.hypothesis_id, spec.primary_test)
    holdout_effect = out["effect"]
    out["extra"] |= {
        "primary_test": spec.primary_test,
        "seasons": sorted(frame["season"].unique().tolist()),
        "explore_effect": explore_effect,
        "same_sign": bool(
            explore_effect and holdout_effect and explore_effect * holdout_effect > 0
        ),
        "retention": real._round(holdout_effect / explore_effect) if explore_effect else None,
        "market": {
            key: market.get(key) for key in ("effect", "ci_low", "ci_high", "p_value", "error")
        }
        | {"rate": market.get("extra", {}).get("rate")},
        "spread_controlled": {
            key: controlled.get(key) for key in ("effect", "ci_low", "ci_high", "p_value", "error")
        },
    }
    return real.result("holdout_confirm", raw, {"holdout": raw}, **out)
