"""The tool registry B passes to Gemini, and the single entry point for calling a tool.

`TOOL_REGISTRY` is a list of `{name, description, json_schema}`. `dispatch` validates arguments and
never raises, so a bad call from the model comes back as `status="error"`.

To swap in a real tool, point its entry at the new function. Names and arguments stay the same,
so nothing on the agent side has to change.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.tools import mock, schemas
from backend.tools.hashing import hash_args
from backend.tools.schemas import ToolResult


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], ToolResult]


_TOOLS: tuple[Tool, ...] = (
    Tool(
        "describe_dataset",
        "List seasons, splits, metrics and filterable columns. Call this first.",
        schemas.DescribeDatasetArgs,
        mock.describe_dataset,
    ),
    Tool(
        "define_slice",
        "Define a filtered subset of team-games and return its size.",
        schemas.DefineSliceArgs,
        mock.define_slice,
    ),
    Tool(
        "add_feature",
        "Confirm a precomputed feature is available and return its coverage.",
        schemas.AddFeatureArgs,
        mock.add_feature,
    ),
    Tool(
        "compare_groups",
        "Compare a metric between two slices with a season-blocked permutation test and "
        "bootstrap CI. Declare the direction before testing.",
        schemas.CompareGroupsArgs,
        mock.compare_groups,
    ),
    Tool(
        "regress",
        "Estimate an effect with controls, using OLS or logit and cluster-robust standard errors.",
        schemas.RegressArgs,
        mock.regress,
    ),
    Tool(
        "out_of_sample",
        "Re-run a prior test unchanged on the validation seasons.",
        schemas.OutOfSampleArgs,
        mock.out_of_sample,
    ),
    Tool(
        "persistence",
        "Correlate a team-season metric with the next season, to spot regression to the mean.",
        schemas.PersistenceArgs,
        mock.persistence,
    ),
    Tool(
        "market_test",
        "Compare a slice's cover or over rate with the break-even rate, and report flat-stake ROI.",
        schemas.MarketTestArgs,
        mock.market_test,
    ),
    Tool(
        "power",
        "Return the minimum detectable effect at 80% power for two sample sizes.",
        schemas.PowerArgs,
        mock.power,
    ),
    Tool(
        "placebo",
        "Run a prior test on a slice where no effect should exist.",
        schemas.PlaceboArgs,
        mock.placebo,
    ),
    Tool(
        "holdout_confirm",
        "Run a hypothesis's primary test on the sealed holdout seasons. Works once per hypothesis.",
        schemas.HoldoutConfirmArgs,
        mock.holdout_confirm,
    ),
)

_TOOLS_BY_NAME = {tool.name: tool for tool in _TOOLS}


def _inline_refs(node: Any, definitions: dict, seen: tuple[str, ...] = ()) -> Any:
    """Replace `$ref`s with their definitions, since Gemini needs a self-contained schema.

    `HypothesisSpec` refers to itself through `mechanism_prediction`, so a reference already being
    expanded becomes a plain object.
    """
    if isinstance(node, list):
        return [_inline_refs(item, definitions, seen) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in seen:
            return {"type": "object"}
        return _inline_refs(definitions[name], definitions, (*seen, name))
    return {
        key: _inline_refs(value, definitions, seen)
        for key, value in node.items()
        if key != "$defs"
    }


def _json_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    return _inline_refs(schema, schema.get("$defs", {}))


TOOL_REGISTRY: list[dict] = [
    {"name": tool.name, "description": tool.description, "json_schema": _json_schema(tool.args_model)}
    for tool in _TOOLS
]


def dispatch(name: str, args: dict | None = None) -> ToolResult:
    """Validate arguments and run a tool. Failures come back as `status="error"`."""
    raw = args or {}
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return ToolResult(
            tool=name, args=raw, status="error", reason="unknown_tool", args_hash=hash_args(raw)
        )
    try:
        return tool.handler(tool.args_model.model_validate(raw))
    except ValidationError as error:
        problems = [
            {"field": ".".join(map(str, item["loc"])), "problem": item["msg"]}
            for item in error.errors()
        ]
        reason, extra = "invalid_args", {"problems": problems}
    except Exception as error:  # noqa: BLE001 - the agent must never see an exception
        reason, extra = "internal_error", {"message": str(error)}
    return ToolResult(
        tool=name, args=raw, status="error", reason=reason, args_hash=hash_args(raw), extra=extra
    )
