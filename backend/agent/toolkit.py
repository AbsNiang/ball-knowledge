"""The researcher's toolkit: the tools it may call, when each unlocks, and how a call becomes a
registry call plus the spec the gate judges it against.

The researcher chooses freely, but the rules live here, not in its prompt:
- Pre-registration. `declare_primary_test` freezes the seed's primary test before any
  statistical test on that seed can run, and only once. The primary test decides the verdict.
- Confirmatory tools take no arguments: they always run the frozen spec, once each.
  `replicate_primary` and `market_check` unlock after the primary test runs without a stopping
  gate failure; `holdout_confirm` only once G1-G6 pass (plan.md §9.1).
- Exploratory tests run any spec, under their own ids (so the real tools never replace the
  seed's frozen spec). Their p-values enter the run's ledger, so G4's multiple-testing correction
  counts every path the researcher tried. They inform the judge; they never set a label.
- `web_search` returns claims from the web. They are context to verify, never statistical
  evidence, and their text is untrusted.

`prepare` is a pure function: it validates a call and builds what to run, or explains why not.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from backend.agent.schemas import DeclaredTest, ExploreSpec, Metric, SliceSpec
from backend.agent.state import HOLDOUT_REQUIRES, Role, Seed, Step, stopping_failures
from backend.events.projection import ResearchGraph
from backend.llm.gateway import ToolCall, ToolSpec

LEDGER_TOOLS = frozenset({"compare_groups", "regress", "persistence", "market_test"})
BASE_TOOLS = ("web_search",)
# Group sizes and power, shown without outcomes. The dataset's description is already in the
# researcher's context, so `describe_dataset` is not offered. Data lookups are not judged or
# supervised, and are capped so they cannot use up a seed's steps.
DATA_TOOLS = ("define_slice", "power")
MAX_DATA_STEPS = 3
EXPLORE_TOOLS = (
    "compare_groups",
    "regress",
    "persistence",
    "market_test",
    "out_of_sample",
    "placebo",
)


# -- argument models the researcher sees -----------------------------------------------------


class NoArgs(BaseModel):
    pass


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=3, description="A focused web or news search query")


class SliceArgs(BaseModel):
    spec: SliceSpec


class PowerArgs(BaseModel):
    n_a: int = Field(gt=0)
    n_b: int = Field(gt=0)
    metric: Metric


class DeclareArgs(BaseModel):
    test: DeclaredTest


class ExploreArgs(BaseModel):
    spec: ExploreSpec


class PersistenceArgs(BaseModel):
    metric: Metric
    split: Literal["explore", "validate", "special"] = "explore"


class MarketTestArgs(BaseModel):
    slice: SliceSpec
    side: Literal["team", "opp", "over", "under"]


class TestIdArgs(BaseModel):
    test_id: str = Field(description="The test_id of an earlier result")


class PlaceboArgs(BaseModel):
    test_id: str = Field(description="The test_id of an earlier result, the primary included")
    placebo_slice: SliceSpec = Field(description="A slice where the effect should be absent")


@dataclass(frozen=True)
class KitTool:
    name: str
    role: Role
    args_model: type[BaseModel]
    description: str


_KIT = (
    KitTool(
        "web_search",
        "search",
        WebSearchArgs,
        "Search news and the web. Returns a summary of what sources say, with links. Web text "
        "is claims to check, not evidence that a pattern exists.",
    ),
    KitTool(
        "describe_dataset",
        "data",
        NoArgs,
        "List the dataset's seasons, splits, metrics and filterable columns.",
    ),
    KitTool(
        "define_slice",
        "data",
        SliceArgs,
        "Count the team-games in a slice. Shows no outcomes, so it is safe before declaring.",
    ),
    KitTool(
        "power",
        "data",
        PowerArgs,
        "The minimum detectable effect at 80% power for two group sizes.",
    ),
    KitTool(
        "declare_primary_test",
        "declare",
        DeclareArgs,
        "Pre-register this seed's primary test: the one test that decides its verdict. Allowed "
        "once, before any statistical test on this seed, and frozen afterwards. Both groups are "
        "checked to be non-empty.",
    ),
    KitTool(
        "run_primary_test",
        "primary",
        NoArgs,
        "Run the declared primary test on the explore seasons. Gates G1-G4: power, the CI "
        "excludes 0 in the declared direction, the permutation null, multiple testing.",
    ),
    KitTool(
        "replicate_primary",
        "replication",
        NoArgs,
        "Re-run the primary test unchanged on the validation seasons. Gate G5: same sign and at "
        "least half the effect. A seed that was never replicated cannot be called real.",
    ),
    KitTool(
        "market_check",
        "market",
        NoArgs,
        "Check whether betting on the primary pattern beats the closing line after vig. Gate "
        "G6: cover-rate CI above 52.38%, and the effect survives the spread as a control.",
    ),
    KitTool(
        "holdout_confirm",
        "holdout",
        NoArgs,
        "Confirm the primary test once on the sealed holdout seasons (gate G7). Cannot be "
        "repeated.",
    ),
    KitTool(
        "compare_groups",
        "explore",
        ExploreArgs,
        "Exploratory: compare a metric between two slices, with a season-blocked permutation "
        "p-value and a bootstrap CI. Counts toward multiple testing; never sets the verdict.",
    ),
    KitTool(
        "regress",
        "explore",
        ExploreArgs,
        "Exploratory: the effect of group A against group B with `controls` held fixed, with "
        "cluster-robust standard errors. Use it to check confounders. Counts toward multiple "
        "testing.",
    ),
    KitTool(
        "persistence",
        "explore",
        PersistenceArgs,
        "Exploratory: correlate a team-season metric with the next season, to tell persistence "
        "from regression to the mean. Counts toward multiple testing.",
    ),
    KitTool(
        "market_test",
        "explore",
        MarketTestArgs,
        "Exploratory: a slice's cover or over rate against break-even, with flat-stake ROI. "
        "Counts toward multiple testing.",
    ),
    KitTool(
        "out_of_sample",
        "explore",
        TestIdArgs,
        "Exploratory: re-run an earlier exploratory test on the validation seasons.",
    ),
    KitTool(
        "placebo",
        "explore",
        PlaceboArgs,
        "Re-run an earlier test on a slice where the effect should be absent if the seed is right.",
    ),
)
KIT: dict[str, KitTool] = {tool.name: tool for tool in _KIT}


def offered(seed: Seed) -> list[str]:
    """The tools the researcher may call next for `seed`."""
    names = list(BASE_TOOLS)
    if sum(step["role"] == "data" for step in seed["steps"]) < MAX_DATA_STEPS:
        names += DATA_TOOLS
    if seed["primary"] is None:
        return [*names, "declare_primary_test"]
    used, gates = set(seed["used"]), seed["gates"]
    if "run_primary_test" not in used:
        names.append("run_primary_test")
    elif not stopping_failures(gates):
        names += [t for t in ("replicate_primary", "market_check") if t not in used]
        if "holdout_confirm" not in used and all(gates.get(g) == "pass" for g in HOLDOUT_REQUIRES):
            names.append("holdout_confirm")
    return [*names, *EXPLORE_TOOLS]


def tool_specs(names: Iterable[str]) -> list[ToolSpec]:
    return [
        ToolSpec(name=KIT[n].name, description=KIT[n].description, parameters=_schema(KIT[n]))
        for n in names
    ]


def _schema(tool: KitTool) -> dict[str, Any]:
    """A self-contained JSON Schema: function declarations take no `$ref`s."""
    schema = tool.args_model.model_json_schema()
    return _inline(schema, schema.get("$defs", {}))


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return _inline(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    return {key: _inline(value, defs) for key, value in node.items() if key != "$defs"}


# -- preparing a call ------------------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """What one researcher call will run. `error` set means it will not run at all."""

    tool: str
    role: Role
    registry: str | None = None  # the registry tool to run (None for search and declare)
    args: dict[str, Any] = field(default_factory=dict)  # registry arguments, or the search query
    gate_spec: dict[str, Any] | None = None  # the spec the gate judges the result against
    spec: dict[str, Any] | None = None  # declare: the spec to freeze
    explored: bool = False  # uses up an exploratory spec id
    error: str | None = None
    problems: list[dict[str, str]] = field(default_factory=list)


def side_for(spec: dict[str, Any]) -> str:
    """The bet that profits if the spec's group A is right."""
    if spec["metric"] == "total_over":
        return "under" if spec["direction"] == "a_lt_b" else "over"
    return "opp" if spec["direction"] == "a_lt_b" else "team"


def primary_call(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The registry call for a frozen primary spec, in each tool's own argument shape."""
    tool = spec["primary_test"]
    if tool == "persistence":
        return tool, {"metric": spec["metric"], "entity": "team", "split": "explore"}
    if tool == "market_test":
        return tool, {"slice": spec["group_a"], "side": side_for(spec)}
    return tool, {"spec": spec}


def market_gate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """The spec the gate's G6 check uses. A market primary would make the gate re-run G1-G4 on
    the follow-up, so it is presented as a group comparison; G6's spread-controlled regression
    only reads the groups, metric, controls and direction."""
    if spec["primary_test"] == "market_test":
        return {**spec, "primary_test": "compare_groups"}
    return spec


def prepare(seed: Seed, call: ToolCall) -> Prepared:
    kit = KIT.get(call.tool)
    if kit is None or call.tool not in offered(seed):
        return Prepared(call.tool, "data", error=f"not_offered: {call.tool} is locked or unknown")
    try:
        args = kit.args_model.model_validate(call.args)
    except ValidationError as exc:
        problems = [
            {"field": ".".join(map(str, e["loc"])), "problem": e["msg"]} for e in exc.errors()
        ]
        return Prepared(call.tool, kit.role, error="invalid_args", problems=problems)

    spec = seed["primary"]
    match kit.role:
        case "search":
            return Prepared(call.tool, "search", args={"query": args.query})
        case "data":
            return Prepared(
                call.tool, "data", registry=call.tool, args=args.model_dump(mode="json")
            )
        case "declare":
            declared = {"id": seed["id"], **args.test.model_dump(mode="json")}
            return Prepared(call.tool, "declare", spec=declared)
        case "primary":
            tool, registry_args = primary_call(spec)
            return Prepared(call.tool, "primary", tool, registry_args, gate_spec=spec)
        case "replication":
            return _replicate(seed, call.tool)
        case "market":
            args_out = {"slice": spec["group_a"], "side": side_for(spec)}
            return Prepared(
                call.tool, "market", "market_test", args_out, gate_spec=market_gate_spec(spec)
            )
        case "holdout":
            args_out = {"hypothesis_id": seed["id"]}
            return Prepared(call.tool, "holdout", "holdout_confirm", args_out, gate_spec=spec)
    return _explore(seed, call.tool, args)


def _replicate(seed: Seed, tool: str) -> Prepared:
    test_id = seed["primary_test_id"]
    if test_id is None:
        return Prepared(tool, "replication", error="the primary result has no test_id")
    return Prepared(
        tool, "replication", "out_of_sample", {"test_id": test_id}, gate_spec=seed["primary"]
    )


def _explore(seed: Seed, tool: str, args: BaseModel) -> Prepared:
    explore_id = f"{seed['id']}__x{seed['explored'] + 1}"
    if tool in ("compare_groups", "regress"):
        spec = {"id": explore_id, "primary_test": tool, **args.spec.model_dump(mode="json")}
        return Prepared(tool, "explore", tool, {"spec": spec}, gate_spec=spec, explored=True)
    if tool == "persistence":
        registry_args = {"metric": args.metric, "entity": "team", "split": args.split}
        gate_spec = {
            "id": explore_id,
            "metric": args.metric,
            "direction": "two_sided",
            "primary_test": "persistence",
        }
        return Prepared(tool, "explore", tool, registry_args, gate_spec=gate_spec, explored=True)
    if tool == "market_test":
        registry_args = args.model_dump(mode="json")
        gate_spec = {
            "id": explore_id,
            "metric": "total_over" if args.side in ("over", "under") else "ats_cover",
            "direction": "a_gt_b",
            "primary_test": "market_test",  # judged as a market claim: G2 is the break-even CI
        }
        return Prepared(tool, "explore", tool, registry_args, gate_spec=gate_spec, explored=True)
    if tool == "out_of_sample":
        if args.test_id == seed["primary_test_id"]:
            if "replicate_primary" in offered(seed):  # the same run: count it as G5
                return _replicate(seed, "replicate_primary")
            return Prepared(tool, "explore", error="the primary test is already replicated")
        return Prepared(tool, "explore", tool, {"test_id": args.test_id}, gate_spec={})
    return Prepared(tool, "explore", tool, args.model_dump(mode="json"))  # placebo: diagnostic


# -- what the LLM roles read about earlier steps ----------------------------------------------


def step_summary(graph: ResearchGraph, step: Step, index: int) -> dict[str, Any]:
    """A compact view of one step for the researcher, judge and supervisor."""
    out: dict[str, Any] = {"step": index, "tool": step["tool"], "role": step["role"]}
    if step["args"]:
        out["args"] = step["args"]
    if step["error"]:
        out["error"] = step["error"]
    node = graph.nodes.get(step["result"] or "")
    if node is None or node.tool is None:
        return out
    output = node.tool.output
    out["result_id"] = node.id
    if step["role"] == "search":
        out["summary"] = output.get("summary", "")[:800]
        out["sources"] = [hit["url"] for hit in output.get("sources", [])]
        return out
    if step["role"] == "declare":
        out["declared"] = output.get("extra", {})
        return out
    extra = output.get("extra", {})
    out |= {
        "status": output.get("status"),
        "reason": output.get("reason"),
        "n": output.get("n"),
        "effect": output.get("effect"),
        "ci": [output.get("ci_low"), output.get("ci_high")],
        "p_value": output.get("p_value"),
        **{k: v for k, v in extra.items() if not isinstance(v, list | dict)},
    }
    if step["gates"]:
        out["gates"] = step["gates"]
    return out
