import json

import pytest

from backend.tools.mock import reset_mock_state
from backend.tools.registry import TOOL_REGISTRY, dispatch
from backend.tools.schemas import SliceSpec, ToolResult

EXPECTED_TOOLS = {
    "describe_dataset",
    "define_slice",
    "add_feature",
    "compare_groups",
    "regress",
    "out_of_sample",
    "persistence",
    "market_test",
    "power",
    "placebo",
    "holdout_confirm",
}

B2B_SLICE = {"split": "explore", "filters": [{"col": "b2b", "op": "eq", "value": True}]}


def spec(hypothesis_id: str = "h1", **changes) -> dict:
    base = {
        "id": hypothesis_id,
        "statement": "Teams on a back-to-back underperform the spread.",
        "kind": "premise",
        "metric": "ats_margin",
        "group_a": B2B_SLICE,
        "group_b": "complement",
        "direction": "a_lt_b",
        "primary_test": "compare_groups",
    }
    return {**base, **changes}


def valid_calls() -> dict[str, dict]:
    return {
        "describe_dataset": {},
        "define_slice": {"spec": B2B_SLICE},
        "add_feature": {"name": "rolling_ats", "params": {"k": 10}},
        "compare_groups": {"spec": spec()},
        "regress": {"spec": spec(controls=["spread"])},
        "persistence": {"metric": "ats_cover", "entity": "team"},
        "market_test": {"slice": B2B_SLICE, "side": "team"},
        "power": {"n_a": 500, "n_b": 2000, "metric": "ats_margin"},
    }


@pytest.fixture(autouse=True)
def clean_state():
    reset_mock_state()


def test_registry_lists_every_tool_with_name_description_and_schema():
    assert {entry["name"] for entry in TOOL_REGISTRY} == EXPECTED_TOOLS
    for entry in TOOL_REGISTRY:
        assert set(entry) == {"name", "description", "json_schema"}
        assert entry["description"]
        assert entry["json_schema"]["type"] == "object"


def test_registry_schemas_are_self_contained_json():
    text = json.dumps(TOOL_REGISTRY)
    assert "$ref" not in text
    assert "$defs" not in text


@pytest.mark.parametrize(("name", "args"), valid_calls().items())
def test_every_tool_returns_an_ok_result_marked_as_mock(name, args):
    result = dispatch(name, args)
    assert isinstance(result, ToolResult)
    assert result.status == "ok"
    assert result.extra["mock"] is True
    assert result.args_hash


@pytest.mark.parametrize(("name", "args"), valid_calls().items())
def test_same_args_give_the_same_output(name, args):
    assert dispatch(name, args) == dispatch(name, json.loads(json.dumps(args)))


def test_different_args_give_different_numbers():
    first = dispatch("compare_groups", {"spec": spec(metric="ats_margin")})
    second = dispatch("compare_groups", {"spec": spec(metric="margin")})
    assert first.effect != second.effect
    assert first.extra["test_id"] != second.extra["test_id"]


def test_results_are_summaries_that_serialise_to_json():
    json.dumps(dispatch("compare_groups", {"spec": spec()}).model_dump())


def test_confidence_interval_and_p_value_agree():
    for index in range(40):
        result = dispatch("compare_groups", {"spec": spec(f"h{index}")})
        excludes_zero = result.ci_low > 0 or result.ci_high < 0
        assert excludes_zero == (result.p_value < 0.05)


def test_unknown_tool_returns_an_error():
    result = dispatch("delete_everything", {})
    assert (result.status, result.reason) == ("error", "unknown_tool")


def test_bad_arguments_return_an_error_and_never_raise():
    result = dispatch("power", {"n_a": -1, "n_b": 10, "metric": "ats_margin"})
    assert (result.status, result.reason) == ("error", "invalid_args")
    assert result.extra["problems"]


def test_arguments_outside_the_schema_are_rejected():
    assert dispatch("describe_dataset", {"verbose": True}).reason == "invalid_args"


def test_filter_on_a_non_whitelisted_column_is_rejected():
    args = {"spec": {"split": "explore", "filters": [{"col": "pts", "op": "eq", "value": 1}]}}
    assert dispatch("define_slice", args).reason == "invalid_args"


@pytest.mark.parametrize(
    ("op", "value"), [("in", []), ("in", 3), ("between", [1]), ("between", 2)]
)
def test_filter_value_must_fit_its_operator(op, value):
    args = {"spec": {"split": "explore", "filters": [{"col": "month", "op": op, "value": value}]}}
    assert dispatch("define_slice", args).reason == "invalid_args"


def test_rolling_feature_needs_a_window():
    assert dispatch("add_feature", {"name": "rolling_ats"}).reason == "invalid_args"


def test_holdout_is_not_a_valid_slice_split():
    args = {"spec": {"split": "holdout", "filters": []}}
    assert dispatch("define_slice", args).reason == "invalid_args"
    assert dispatch("market_test", {"slice": args["spec"], "side": "team"}).reason == "invalid_args"
    bad = spec(group_a=args["spec"])
    assert dispatch("compare_groups", {"spec": bad}).reason == "invalid_args"


def test_describe_dataset_does_not_reveal_the_holdout():
    text = json.dumps(dispatch("describe_dataset").model_dump()).lower()
    assert "holdout" not in text
    assert "2021-22" not in text


def test_out_of_sample_needs_a_prior_test():
    assert dispatch("out_of_sample", {"test_id": "nope"}).reason == "unknown_test_id"


def test_out_of_sample_reports_sign_and_retention():
    test_id = dispatch("compare_groups", {"spec": spec()}).extra["test_id"]
    result = dispatch("out_of_sample", {"test_id": test_id})
    assert result.status == "ok"
    assert isinstance(result.extra["same_sign"], bool)
    assert "retention" in result.extra


def test_placebo_needs_a_prior_test():
    args = {"test_id": "nope", "placebo_slice": B2B_SLICE}
    assert dispatch("placebo", args).reason == "unknown_test_id"


def test_holdout_second_call_fails():
    dispatch("compare_groups", {"spec": spec()})
    assert dispatch("holdout_confirm", {"hypothesis_id": "h1"}).status == "ok"
    second = dispatch("holdout_confirm", {"hypothesis_id": "h1"})
    assert (second.status, second.reason) == ("error", "already_used")


def test_holdout_use_is_counted_per_hypothesis():
    dispatch("compare_groups", {"spec": spec("h1")})
    dispatch("compare_groups", {"spec": spec("h2", metric="margin")})
    assert dispatch("holdout_confirm", {"hypothesis_id": "h1"}).status == "ok"
    assert dispatch("holdout_confirm", {"hypothesis_id": "h2"}).status == "ok"


def test_holdout_call_for_an_unknown_hypothesis_does_not_use_up_the_call():
    assert dispatch("holdout_confirm", {"hypothesis_id": "h1"}).reason == "unknown_hypothesis"
    dispatch("compare_groups", {"spec": spec()})
    assert dispatch("holdout_confirm", {"hypothesis_id": "h1"}).status == "ok"


def test_power_mde_shrinks_as_samples_grow():
    small = dispatch("power", {"n_a": 200, "n_b": 200, "metric": "ats_margin"})
    large = dispatch("power", {"n_a": 2000, "n_b": 2000, "metric": "ats_margin"})
    assert large.extra["mde"] < small.extra["mde"]


def test_hypothesis_with_a_nested_mechanism_is_accepted():
    nested = spec("h2", mechanism_prediction={"statement": "Fatigue.", "secondary_test": spec("h3")})
    assert dispatch("compare_groups", {"spec": nested}).status == "ok"


def test_slice_spec_defaults_to_explore():
    assert SliceSpec(filters=[]).split == "explore"
