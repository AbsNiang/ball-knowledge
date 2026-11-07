"""Stub tools with real signatures and fake numbers, so the agent can be built against them.

NOTHING HERE IS A REAL RESULT. Every result has `extra["mock"] = True`. Numbers come from a hash
of the arguments, so the same arguments always give the same output and the values look plausible
(CIs and p-values agree with each other). None of this touches the data.

Three behaviours the agent has to handle are real, not faked: `out_of_sample` and `placebo` need a
prior test id, and `holdout_confirm` works once per hypothesis. That needs a little state, which
the real tools will also keep, and `reset_mock_state` clears it between runs and tests.
"""

import math
import random
from typing import get_args

from backend.data.seasons import EXPLORE_STARTS, SPECIAL_STARTS, VALIDATE_STARTS, season_label
from backend.gate.config import BREAKEVEN, SEED, SESOI_POINTS, SESOI_RATE
from backend.tools.hashing import hash_args
from backend.tools.schemas import (
    AddFeatureArgs,
    CompareGroupsArgs,
    DefineSliceArgs,
    DescribeDatasetArgs,
    HoldoutConfirmArgs,
    HypothesisSpec,
    MarketTestArgs,
    Metric,
    OutOfSampleArgs,
    PersistenceArgs,
    PlaceboArgs,
    PowerArgs,
    RegressArgs,
    ToolResult,
)

_Z_95 = 1.959964
_Z_80_POWER = 0.841621
_POINT_METRICS = ("ats_margin", "margin")
_POINT_SD = 12.0  # rough game-level spread of margins
_RATE_SD = 0.5  # the largest possible spread of a binary outcome
_NULL_HISTOGRAM_BINS = 21

_tests: dict[str, dict] = {}  # test_id -> the test's result fields
_hypotheses: dict[str, HypothesisSpec] = {}
_holdout_used: set[str] = set()


def reset_mock_state() -> None:
    """Forget prior tests and holdout use."""
    _tests.clear()
    _hypotheses.clear()
    _holdout_used.clear()


def _rng(tool: str, args_hash: str) -> random.Random:
    return random.Random(f"{SEED}:{tool}:{args_hash}")


def _result(tool: str, args: dict, **fields) -> ToolResult:
    extra = {"mock": True, **fields.pop("extra", {})}
    return ToolResult(tool=tool, args=args, args_hash=hash_args(args), extra=extra, **fields)


def _error(tool: str, args: dict, reason: str) -> ToolResult:
    return _result(tool, args, status="error", reason=reason)


def _unit_sd(metric: str) -> float:
    return _POINT_SD if metric in _POINT_METRICS else _RATE_SD


def _sesoi(metric: str) -> float:
    return SESOI_POINTS if metric in _POINT_METRICS else SESOI_RATE


def _min_detectable_effect(n_a: int, n_b: int, metric: str) -> float:
    standard_error = _unit_sd(metric) * math.sqrt(1 / n_a + 1 / n_b)
    return (_Z_95 + _Z_80_POWER) * standard_error


def _p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def _estimate(rng: random.Random, metric: str, n_a: int, n_b: int) -> dict:
    """Draw an effect, then derive a CI and p-value that agree with it."""
    standard_error = _unit_sd(metric) * math.sqrt(1 / n_a + 1 / n_b)
    effect = rng.gauss(0, 2 * standard_error)
    return {
        "effect": round(effect, 4),
        "ci_low": round(effect - _Z_95 * standard_error, 4),
        "ci_high": round(effect + _Z_95 * standard_error, 4),
        "p_value": round(_p_two_sided(effect / standard_error), 4),
    }


def _null_histogram(rng: random.Random, spread: float) -> dict:
    half_width = 4 * spread
    width = 2 * half_width / _NULL_HISTOGRAM_BINS
    edges = [-half_width + width * index for index in range(_NULL_HISTOGRAM_BINS + 1)]
    counts = [0] * _NULL_HISTOGRAM_BINS
    for _ in range(1000):
        position = int((rng.gauss(0, spread) + half_width) / width)
        counts[min(max(position, 0), _NULL_HISTOGRAM_BINS - 1)] += 1
    return {"edges": [round(edge, 4) for edge in edges], "counts": counts}


def _remember_test(result: ToolResult, metric: str) -> None:
    _tests[result.args_hash] = {"metric": metric, "effect": result.effect}


def describe_dataset(_args: DescribeDatasetArgs) -> ToolResult:
    def labels(starts: range) -> list[str]:
        return [season_label(start) for start in starts]

    return _result(
        "describe_dataset",
        {},
        extra={
            "splits": {
                "explore": labels(EXPLORE_STARTS),
                "validate": labels(VALIDATE_STARTS),
                "special": labels(SPECIAL_STARTS),
            },
            "metrics": list(get_args(Metric)),
            "columns": {
                "season": "string",
                "team": "string",
                "opp": "string",
                "is_home": "bool",
                "rest_days": "int",
                "b2b": "bool",
                "opp_b2b": "bool",
                "travel_km": "float",
                "tz_shift": "int",
                "altitude_m": "int",
                "month": "int",
                "is_bubble": "bool",
                "is_playoffs": "bool",
                "games_into_season": "int",
            },
        },
    )


def define_slice(args: DefineSliceArgs) -> ToolResult:
    raw = args.model_dump()
    slice_id = hash_args(raw)
    n_games = _rng("define_slice", slice_id).randint(250, 4000)
    return _result(
        "define_slice",
        raw,
        n=n_games * 2,
        slice_hash=slice_id,
        extra={"slice_id": slice_id, "n_rows": n_games * 2, "n_games": n_games, "n_teams": 30},
    )


def add_feature(args: AddFeatureArgs) -> ToolResult:
    return _result("add_feature", args.model_dump(), extra={"exists": True, "coverage": 0.99})


def _run_group_test(tool: str, spec: HypothesisSpec, raw: dict) -> ToolResult:
    args_hash = hash_args(raw)
    rng = _rng(tool, args_hash)
    n_a, n_b = rng.randint(300, 3000), rng.randint(300, 3000)
    estimate = _estimate(rng, spec.metric, n_a, n_b)
    spread = _unit_sd(spec.metric) * math.sqrt(1 / n_a + 1 / n_b)
    result = _result(
        tool,
        raw,
        n=n_a + n_b,
        slice_hash=args_hash,
        **estimate,
        extra={
            "test_id": args_hash,
            "n_a": n_a,
            "n_b": n_b,
            "mde": round(_min_detectable_effect(n_a, n_b, spec.metric), 4),
            "null_histogram": _null_histogram(rng, spread),
            "observed": estimate["effect"],
        },
    )
    _remember_test(result, spec.metric)
    _hypotheses[spec.id] = spec
    return result


def compare_groups(args: CompareGroupsArgs) -> ToolResult:
    return _run_group_test("compare_groups", args.spec, args.model_dump())


def regress(args: RegressArgs) -> ToolResult:
    result = _run_group_test("regress", args.spec, args.model_dump())
    result.extra.update({"model": "ols", "se": "cluster_game_id", "controls": args.spec.controls})
    return result


def out_of_sample(args: OutOfSampleArgs) -> ToolResult:
    raw = args.model_dump()
    prior = _tests.get(args.test_id)
    if prior is None:
        return _error("out_of_sample", raw, "unknown_test_id")
    rng = _rng("out_of_sample", args.test_id)
    n_a, n_b = rng.randint(150, 900), rng.randint(150, 900)
    estimate = _estimate(rng, prior["metric"], n_a, n_b)
    same_sign = estimate["effect"] * prior["effect"] > 0
    retention = estimate["effect"] / prior["effect"] if prior["effect"] else None
    return _result(
        "out_of_sample",
        raw,
        n=n_a + n_b,
        **estimate,
        extra={
            "n_a": n_a,
            "n_b": n_b,
            "same_sign": same_sign,
            "retention": None if retention is None else round(retention, 4),
        },
    )


def persistence(args: PersistenceArgs) -> ToolResult:
    raw = args.model_dump()
    args_hash = hash_args(raw)
    rng = _rng("persistence", args_hash)
    n_pairs = 240  # 30 teams across the 8 consecutive season pairs in explore
    correlation = round(rng.gauss(0, 0.1), 4)
    standard_error = 1 / math.sqrt(n_pairs - 3)
    result = _result(
        "persistence",
        raw,
        n=n_pairs,
        effect=correlation,
        ci_low=round(correlation - _Z_95 * standard_error, 4),
        ci_high=round(correlation + _Z_95 * standard_error, 4),
        p_value=round(_p_two_sided(correlation / standard_error), 4),
        extra={"test_id": args_hash},
    )
    _remember_test(result, args.metric)
    return result


def market_test(args: MarketTestArgs) -> ToolResult:
    raw = args.model_dump()
    args_hash = hash_args(raw)
    rng = _rng("market_test", args_hash)
    n = rng.randint(300, 3000)
    rate = 0.5 + rng.gauss(0, 0.02)
    standard_error = math.sqrt(rate * (1 - rate) / n)
    edge = rate - BREAKEVEN
    result = _result(
        "market_test",
        raw,
        n=n,
        effect=round(edge, 4),
        ci_low=round(rate - _Z_95 * standard_error, 4),
        ci_high=round(rate + _Z_95 * standard_error, 4),
        p_value=round(_p_two_sided(edge / standard_error), 4),
        slice_hash=args_hash,
        extra={
            "test_id": args_hash,
            "rate": round(rate, 4),
            "breakeven": BREAKEVEN,
            "roi": round(rate * (100 / 110) - (1 - rate), 4),
        },
    )
    _remember_test(result, "ats_cover")
    return result


def power(args: PowerArgs) -> ToolResult:
    mde = _min_detectable_effect(args.n_a, args.n_b, args.metric)
    return _result(
        "power",
        args.model_dump(),
        n=args.n_a + args.n_b,
        extra={"mde": round(mde, 4), "sesoi": _sesoi(args.metric)},
    )


def placebo(args: PlaceboArgs) -> ToolResult:
    raw = args.model_dump()
    prior = _tests.get(args.test_id)
    if prior is None:
        return _error("placebo", raw, "unknown_test_id")
    rng = _rng("placebo", hash_args(raw))
    n_a, n_b = rng.randint(300, 1500), rng.randint(300, 1500)
    return _result(
        "placebo",
        raw,
        n=n_a + n_b,
        **_estimate(rng, prior["metric"], n_a, n_b),
        extra={"n_a": n_a, "n_b": n_b},
    )


def holdout_confirm(args: HoldoutConfirmArgs) -> ToolResult:
    raw = args.model_dump()
    spec = _hypotheses.get(args.hypothesis_id)
    if spec is None:
        return _error("holdout_confirm", raw, "unknown_hypothesis")
    if args.hypothesis_id in _holdout_used:
        return _error("holdout_confirm", raw, "already_used")
    _holdout_used.add(args.hypothesis_id)
    rng = _rng("holdout_confirm", args.hypothesis_id)
    n_a, n_b = rng.randint(150, 700), rng.randint(150, 700)
    return _result(
        "holdout_confirm",
        raw,
        n=n_a + n_b,
        **_estimate(rng, spec.metric, n_a, n_b),
        extra={"n_a": n_a, "n_b": n_b},
    )
