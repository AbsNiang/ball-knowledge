"""The statistical tools, run on the built dataset (plan.md v2 section 7).

Every tool is deterministic given `(args, data version, seed)`: its random generator is seeded
from `SEED` and the arguments' hash. Tools return summaries, never data frames, and report
problems as `status="error"` instead of raising.

Statistics (section 7.3):
- Effect: mean of group A minus mean of group B, in the metric's units (rates for 0/1 metrics).
- P-value: 10,000 permutations of whole games' labels within seasons. One-sided when the spec
  declares a direction, two-sided otherwise.
- CI: 5,000 bootstrap resamples of whole games within seasons, percentile method.
- MDE: (z 0.975 + z 0.80) times the bootstrap standard error.

Prior tests, registered hypotheses and holdout use belong to the RegistryTools run context.
Standalone dispatch retains its own context for the CLI and tool tests.
"""

import itertools
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest

from backend.gate import config
from backend.tools import data, stats
from backend.tools.hashing import hash_args
from backend.tools.schemas import (
    AddFeatureArgs,
    CompareGroupsArgs,
    DefineSliceArgs,
    DescribeDatasetArgs,
    HypothesisSpec,
    MarketTestArgs,
    OutOfSampleArgs,
    PersistenceArgs,
    PlaceboArgs,
    PowerArgs,
    RegressArgs,
    SliceSpec,
    ToolResult,
)
from backend.tools.slices import SliceError, complement, metric_values, select
from backend.tools.state import current

_BINARY_METRICS = {"ats_cover", "win", "total_over"}
_SPREAD_CONTROLLED_METRICS = {"ats_margin", "ats_cover", "win", "roi"}
_ROLLING_COLUMNS = {"rolling_ats": "rolling_ats_10", "rolling_margin": "rolling_margin_10"}
_PRECOMPUTED_WINDOWS = [10]
_WIN_PAYOUT = 100 / 110
_COLUMN_TYPES = {
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
}

def reset_state() -> None:
    """Forget history in the current execution context."""
    current().tests.clear()
    current().hypotheses.clear()
    current().holdout_used.clear()


class ToolFailure(Exception):
    """A tool can't produce a result. Becomes `status="error"` with `reason`."""

    def __init__(self, reason: str, **extra) -> None:
        super().__init__(reason)
        self.reason = reason
        self.extra = extra


def guarded(name: str) -> Callable:
    """Turn expected failures into error results, so the agent never sees an exception."""

    def wrap(function: Callable) -> Callable:
        def run(args) -> ToolResult:
            raw = args.model_dump(mode="json")
            try:
                return function(args, raw)
            except ToolFailure as failure:
                return error(name, raw, failure.reason, failure.extra)
            except SliceError as failure:
                return error(name, raw, "invalid_filter", {"message": str(failure)})
            except FileNotFoundError as failure:
                return error(name, raw, "data_not_built", {"message": str(failure)})

        run.__name__ = name
        return run

    return wrap


def error(tool: str, raw: dict, reason: str, extra: dict | None = None) -> ToolResult:
    return ToolResult(
        tool=tool,
        args=raw,
        status="error",
        reason=reason,
        args_hash=hash_args(raw),
        extra={"mock": False, **(extra or {})},
    )


def result(tool: str, raw: dict, slice_parts: object, **fields) -> ToolResult:
    version = data.data_version()
    extra = {"mock": False, "data_version": version, "seed": config.SEED, **fields.pop("extra", {})}
    return ToolResult(
        tool=tool,
        args=raw,
        args_hash=hash_args(raw),
        slice_hash=hash_args({"data": version, "slice": slice_parts}),
        extra=extra,
        **fields,
    )


def rng_for(args_hash: str) -> np.random.Generator:
    return np.random.default_rng([config.SEED, int(args_hash, 16)])


def _round(value: float | None, digits: int = 5) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _directional_p(p_two_sided: float, effect: float, direction: str) -> float:
    if direction == "two_sided":
        return p_two_sided
    agrees = effect > 0 if direction == "a_gt_b" else effect < 0
    return p_two_sided / 2 if agrees else 1 - p_two_sided / 2


# Shared computations. Each takes a frame, so holdout_confirm can pass the sealed rows.


def groups(frame: pd.DataFrame, spec: HypothesisSpec, use_split: bool = True):
    group_a = select(frame, spec.group_a, use_split=use_split)
    if spec.group_b == "complement":
        group_b = complement(frame, spec.group_a, group_a, use_split=use_split)
    else:
        group_b = select(frame, spec.group_b, use_split=use_split)
    if (group_a & group_b).any():
        raise ToolFailure("groups_overlap", overlapping_rows=int((group_a & group_b).sum()))
    return group_a, group_b


def compare(frame: pd.DataFrame, spec: HypothesisSpec, rng, use_split: bool = True) -> dict:
    group_a, group_b = groups(frame, spec, use_split)
    labels = np.where(group_a, 1, np.where(group_b, 2, 0)).astype(np.int8)
    table = stats.build_table(frame, metric_values(frame, spec.metric), labels)
    n_a, n_b = table.counts()
    if n_a == 0 or n_b == 0:
        raise ToolFailure("empty_group", n_a=n_a, n_b=n_b)

    effect = stats.observed(table)
    mean_a = stats.observed(table, "mean")
    null = stats.permutation_null(table, config.N_PERMUTATIONS, rng)
    boots = stats.bootstrap(table, config.N_BOOTSTRAP, rng, "difference")
    ci_low, ci_high = stats.percentile_ci(boots)
    standard_error = float(np.nanstd(boots, ddof=1))
    return {
        "n": n_a + n_b,
        "effect": _round(effect),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "p_value": _round(stats.permutation_p(null, effect, spec.direction), 6),
        "extra": {
            "n_a": n_a,
            "n_b": n_b,
            "n_games": table.n_games,
            "mean_a": _round(mean_a),
            "mean_b": _round(mean_a - effect),
            "se": _round(standard_error),
            "mde": _round(stats.min_detectable_effect(standard_error)),
            "direction": spec.direction,
            "observed": _round(effect),
            "null_histogram": stats.histogram(null),
        },
    }


def _fit(y: np.ndarray, design: pd.DataFrame, clusters: np.ndarray, binary: bool):
    """Return (effect, se, ci_low, ci_high, p_two_sided) for the `group_a` column."""
    exog = sm.add_constant(design, has_constant="add")
    if np.linalg.matrix_rank(exog.to_numpy()) < exog.shape[1]:
        raise ToolFailure("collinear_controls", columns=list(design.columns))
    cluster_kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": clusters}}
    try:
        if binary:
            fitted = sm.Logit(y, exog).fit(disp=0, **cluster_kwargs)
            row = fitted.get_margeff(at="overall", dummy=True).summary_frame().loc["group_a"]
            effect, se, p_value, low, high = (float(row.iloc[i]) for i in (0, 1, 3, 4, 5))
        else:
            fitted = sm.OLS(y, exog).fit(**cluster_kwargs)
            effect, se = float(fitted.params["group_a"]), float(fitted.bse["group_a"])
            low, high = (float(v) for v in fitted.conf_int().loc["group_a"])
            p_value = float(fitted.pvalues["group_a"])
    except (np.linalg.LinAlgError, ValueError) as failure:
        raise ToolFailure("model_failed", message=str(failure)) from failure
    if not all(math.isfinite(v) for v in (effect, se, low, high, p_value)):
        raise ToolFailure("model_failed", message="non-finite estimate")
    return effect, se, low, high, p_value


def regression(frame: pd.DataFrame, spec: HypothesisSpec, use_split: bool = True) -> dict:
    group_a, group_b = groups(frame, spec, use_split)
    in_model = group_a | group_b
    rows = frame[in_model]
    y = metric_values(rows, spec.metric)

    controls = list(dict.fromkeys(spec.controls))
    spread_added = (
        spec.kind == "market"
        and spec.metric in _SPREAD_CONTROLLED_METRICS
        and "spread" not in controls
    )
    if spread_added:
        controls.append("spread")
    design = pd.DataFrame({"group_a": group_a[in_model].astype(float).to_numpy()})
    for control in controls:
        design[control] = rows[control].astype(float).to_numpy()

    usable = ~np.isnan(y) & design.notna().all(axis=1).to_numpy()
    n_a = int(((design["group_a"] == 1).to_numpy() & usable).sum())
    n_b = int(((design["group_a"] == 0).to_numpy() & usable).sum())
    if n_a == 0 or n_b == 0:
        raise ToolFailure("empty_group", n_a=n_a, n_b=n_b)

    y, design = y[usable], design[usable].reset_index(drop=True)
    clusters = pd.factorize(rows["game_id"].to_numpy()[usable])[0]
    binary = spec.metric in _BINARY_METRICS
    effect, se, low, high, p_two = _fit(y, design, clusters, binary)

    extra = {
        "n_a": n_a,
        "n_b": n_b,
        "rows_dropped": int((~usable).sum()),
        "model": "logit, average marginal effect" if binary else "ols",
        "se_type": "cluster-robust by game_id",
        "controls": controls,
        "spread_added": spread_added,
        "se": _round(se),
        "mde": _round(stats.min_detectable_effect(se)),
        "p_two_sided": _round(p_two, 6),
        "direction": spec.direction,
    }
    if controls:
        base = _fit(y, design[["group_a"]], clusters, binary)[0]
        extra["effect_without_controls"] = _round(base)
        extra["shrinkage"] = _round(1 - effect / base) if base else None
    return {
        "n": n_a + n_b,
        "effect": _round(effect),
        "ci_low": _round(low),
        "ci_high": _round(high),
        "p_value": _round(_directional_p(p_two, effect, spec.direction), 6),
        "extra": extra,
    }


def persistence_stats(frame: pd.DataFrame, metric: str, split: str, rng) -> dict:
    rows = frame[frame["split"] == split]
    values = pd.DataFrame(
        {"team": rows["team"], "season": rows["season"], "value": metric_values(rows, metric)}
    ).dropna()
    by_season = values.groupby(["team", "season"])["value"].mean().unstack("season")
    seasons = sorted(by_season.columns)

    pairs = []
    for block, (this, following) in enumerate(itertools.pairwise(seasons)):
        if int(following[:4]) - int(this[:4]) != 1:
            continue
        both = by_season[[this, following]].dropna()
        pairs.append(pd.DataFrame({"x": both[this], "y": both[following], "block": block}))
    if not pairs:
        raise ToolFailure("not_enough_seasons", seasons=seasons)
    paired = pd.concat(pairs, ignore_index=True)
    n = len(paired)
    if n < 4:
        raise ToolFailure("not_enough_seasons", seasons=seasons)

    x = paired["x"].to_numpy()
    y = paired["y"].to_numpy()
    x_z, y_z = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    correlation = float(np.mean(x_z * y_z))

    # Shuffle next-season values among teams within each season pair. That keeps the overall
    # mean and spread of y, so the shuffled correlation is a plain dot product.
    block = paired["block"].to_numpy()
    null = []
    for start in range(0, config.N_PERMUTATIONS, 250):
        size = min(250, config.N_PERMUTATIONS - start)
        order = np.argsort(rng.random((size, n)) + block, axis=1)
        null.append((y_z[order] * x_z).mean(axis=1))
    null = np.concatenate(null)

    fisher_se = 1 / math.sqrt(n - 3)
    centre = math.atanh(max(min(correlation, 0.999999), -0.999999))
    return {
        "n": n,
        "effect": _round(correlation),
        "ci_low": _round(math.tanh(centre - stats.Z_95 * fisher_se)),
        "ci_high": _round(math.tanh(centre + stats.Z_95 * fisher_se)),
        "p_value": _round(stats.permutation_p(null, correlation, "two_sided"), 6),
        "extra": {
            "n_pairs": n,
            "season_pairs": len(pairs),
            "mde": _round(stats.min_detectable_effect(fisher_se)),
            "null_histogram": stats.histogram(null),
        },
    }


def market_stats(frame, slice_spec: SliceSpec, side: str, rng, use_split: bool = True) -> dict:
    rows = frame[select(frame, slice_spec, use_split=use_split)]
    if side in ("team", "opp"):
        outcome = rows["ats_cover"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        pushes = int((rows["ats_margin"] == 0).sum())
    else:
        rows = rows.drop_duplicates("game_id")
        outcome = rows["total_over"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        pushes = int((rows["pts"] + rows["opp_pts"] == rows["close_total"]).sum())
    if side in ("opp", "under"):
        outcome = 1 - outcome

    wins, losses = int((outcome == 1).sum()), int((outcome == 0).sum())
    bets = wins + losses
    if bets == 0:
        raise ToolFailure("no_bets")
    table = stats.build_table(rows, outcome, np.ones(len(rows), dtype=np.int8))
    ci_low, ci_high = stats.percentile_ci(
        stats.bootstrap(table, config.N_BOOTSTRAP, rng, "mean")
    )
    test = binomtest(wins, bets, config.BREAKEVEN, alternative="greater")
    wilson = binomtest(wins, bets).proportion_ci(method="wilson")  # two-sided 95%
    rate = wins / bets
    return {
        "n": bets,
        "effect": _round(rate - config.BREAKEVEN),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "p_value": _round(test.pvalue, 6),
        "extra": {
            "side": side,
            "rate": _round(rate),
            "breakeven": config.BREAKEVEN,
            "roi": _round((wins * _WIN_PAYOUT - losses) / (bets + pushes)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "n_games": table.n_games,
            "wilson_ci": [_round(wilson.low), _round(wilson.high)],
        },
    }


def _with_split(spec: SliceSpec, split: str) -> SliceSpec:
    return spec.model_copy(update={"split": split})


def _remember(test_id: str, tool: str, args, effect: float | None) -> None:
    current().tests[test_id] = {"tool": tool, "args": args, "effect": effect}


def latest_primary_effect(hypothesis_id: str, tool: str) -> float | None:
    """The explore-split effect of a hypothesis's primary test, if it was run."""
    for record in reversed(list(current().tests.values())):
        spec = getattr(record["args"], "spec", None)
        if record["tool"] == tool and spec is not None and spec.id == hypothesis_id:
            return record["effect"]
    return None


# Tools


@guarded("describe_dataset")
def describe_dataset(_args: DescribeDatasetArgs, raw: dict) -> ToolResult:
    frame = data.public_frame()
    splits = {
        split: {
            "seasons": sorted(frame.loc[frame["split"] == split, "season"].unique().tolist()),
            "rows": int((frame["split"] == split).sum()),
        }
        for split in ("explore", "validate", "special")
    }
    coverage = {
        name: _round(frame.loc[frame["split"] == "explore", column].notna().mean(), 4)
        for name, column in (("rolling_ats", "rolling_ats_10"), ("travel_km", "travel_km"))
    }
    return result(
        "describe_dataset",
        raw,
        "dataset",
        n=len(frame),
        extra={
            "splits": splits,
            "metrics": ["ats_margin", "ats_cover", "margin", "win", "total_over", "roi"],
            "columns": _COLUMN_TYPES,
            "feature_coverage": coverage,
            "rolling_windows": _PRECOMPUTED_WINDOWS,
        },
    )


@guarded("define_slice")
def define_slice(args: DefineSliceArgs, raw: dict) -> ToolResult:
    frame = data.public_frame()
    rows = frame[select(frame, args.spec)]
    slice_id = hash_args(raw)
    return result(
        "define_slice",
        raw,
        raw,
        n=len(rows),
        extra={
            "slice_id": slice_id,
            "n_rows": len(rows),
            "n_games": int(rows["game_id"].nunique()),
            "n_teams": int(rows["team"].nunique()),
            "seasons": sorted(rows["season"].unique().tolist()),
        },
    )


@guarded("add_feature")
def add_feature(args: AddFeatureArgs, raw: dict) -> ToolResult:
    column = _ROLLING_COLUMNS.get(args.name, args.name)
    if args.name in _ROLLING_COLUMNS and args.params.get("k") not in _PRECOMPUTED_WINDOWS:
        raise ToolFailure("unsupported_window", available=_PRECOMPUTED_WINDOWS)
    frame = data.public_frame()
    explore = frame[frame["split"] == "explore"]
    return result(
        "add_feature",
        raw,
        column,
        extra={"exists": True, "column": column, "coverage": _round(explore[column].notna().mean(), 4)},
    )


def _group_test(tool: str, args, raw: dict, run) -> ToolResult:
    existing = current().hypotheses.get(args.spec.id)
    if existing is not None and tool == existing.primary_test and existing != args.spec:
        raise ToolFailure("hypothesis_already_registered")
    args_hash = hash_args(raw)
    out = run(data.public_frame(), args.spec, rng_for(args_hash))
    _remember(args_hash, tool, args, out["effect"])
    # The spec holdout_confirm will run is the one used by the hypothesis's own primary test, so
    # a secondary test under the same id can't replace it. Any spec registers if none has yet.
    if args.spec.id not in current().hypotheses:
        current().hypotheses[args.spec.id] = args.spec
    out["extra"] |= {"test_id": args_hash, "spec_hash": hash_args({"tool": tool, "args": raw})}
    return result(tool, raw, raw, **out)


@guarded("compare_groups")
def compare_groups(args: CompareGroupsArgs, raw: dict) -> ToolResult:
    return _group_test("compare_groups", args, raw, compare)


@guarded("regress")
def regress(args: RegressArgs, raw: dict) -> ToolResult:
    return _group_test("regress", args, raw, lambda frame, spec, _rng: regression(frame, spec))


@guarded("persistence")
def persistence(args: PersistenceArgs, raw: dict) -> ToolResult:
    args_hash = hash_args(raw)
    out = persistence_stats(data.public_frame(), args.metric, args.split, rng_for(args_hash))
    _remember(args_hash, "persistence", args, out["effect"])
    out["extra"] |= {"test_id": args_hash, "spec_hash": hash_args({"tool": "persistence", "args": raw})}
    return result("persistence", raw, raw, **out)


@guarded("market_test")
def market_test(args: MarketTestArgs, raw: dict) -> ToolResult:
    args_hash = hash_args(raw)
    out = market_stats(data.public_frame(), args.slice, args.side, rng_for(args_hash))
    _remember(args_hash, "market_test", args, out["effect"])
    out["extra"] |= {"test_id": args_hash, "spec_hash": hash_args({"tool": "market_test", "args": raw})}
    return result("market_test", raw, raw, **out)


@guarded("power")
def power(args: PowerArgs, raw: dict) -> ToolResult:
    frame = data.public_frame()
    values = metric_values(frame[frame["split"] == "explore"], args.metric)
    spread = float(np.nanstd(values, ddof=1))
    standard_error = spread * math.sqrt(1 / args.n_a + 1 / args.n_b)
    sesoi = config.SESOI_POINTS if args.metric in ("ats_margin", "margin") else config.SESOI_RATE
    return result(
        "power",
        raw,
        args.metric,
        n=args.n_a + args.n_b,
        extra={
            "mde": _round(stats.min_detectable_effect(standard_error)),
            "sesoi": sesoi,
            "row_sd": _round(spread),
            "note": "Treats rows as independent, so it is slightly optimistic for paired games.",
        },
    )


def _rerun(prior: dict, rng, *, split: str | None = None, group_a: SliceSpec | None = None):
    """Run a stored test again with its split or group A replaced."""
    args = prior["args"]
    frame = data.public_frame()
    if prior["tool"] in ("compare_groups", "regress"):
        spec = args.spec
        update = {}
        if split is not None:
            update["group_a"] = _with_split(spec.group_a, split)
            if spec.group_b != "complement":
                update["group_b"] = _with_split(spec.group_b, split)
        if group_a is not None:
            update["group_a"] = group_a
        spec = spec.model_copy(update=update)
        if prior["tool"] == "compare_groups":
            return compare(frame, spec, rng)
        return regression(frame, spec)
    if prior["tool"] == "market_test":
        slice_spec = group_a if group_a is not None else _with_split(args.slice, split)
        return market_stats(frame, slice_spec, args.side, rng)
    if group_a is not None:
        raise ToolFailure("placebo_not_supported", tool=prior["tool"])
    return persistence_stats(frame, args.metric, split, rng)


@guarded("out_of_sample")
def out_of_sample(args: OutOfSampleArgs, raw: dict) -> ToolResult:
    prior = current().tests.get(args.test_id)
    if prior is None:
        raise ToolFailure("unknown_test_id")
    out = _rerun(prior, rng_for(hash_args(raw)), split="validate")
    explore_effect = prior["effect"]
    validate_effect = out["effect"]
    out["extra"] |= {
        "explore_effect": explore_effect,
        "same_sign": bool(explore_effect and validate_effect and explore_effect * validate_effect > 0),
        "retention": _round(validate_effect / explore_effect) if explore_effect else None,
        "original_tool": prior["tool"],
    }
    return result("out_of_sample", raw, raw, **out)


@guarded("placebo")
def placebo(args: PlaceboArgs, raw: dict) -> ToolResult:
    prior = current().tests.get(args.test_id)
    if prior is None:
        raise ToolFailure("unknown_test_id")
    out = _rerun(prior, rng_for(hash_args(raw)), group_a=args.placebo_slice)
    out["extra"] |= {"original_effect": prior["effect"], "original_tool": prior["tool"]}
    return result("placebo", raw, raw, **out)
