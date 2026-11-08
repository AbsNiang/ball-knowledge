"""Pure rules of the research loop: tool locks, call preparation, the judge's audit, routing and
the Gemini gateway's parsing. No LangGraph and no network."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.agent.judge import audit
from backend.agent.router import exhaust, route
from backend.agent.schemas import JudgedClaim, Judgment
from backend.agent.state import (
    BUDGET_LABEL,
    MAX_STEPS_PER_SEED,
    initial_state,
    new_seed,
)
from backend.agent.toolkit import (
    BASE_TOOLS,
    DATA_TOOLS,
    EXPLORE_TOOLS,
    KIT,
    LEDGER_TOOLS,
    MAX_DATA_STEPS,
    market_gate_spec,
    offered,
    prepare,
    primary_call,
    side_for,
    tool_specs,
)
from backend.llm.gateway import NoToolCall, ToolCall, ToolSpec
from backend.llm.gemini import GeminiGateway

PASS = {"G1": "pass", "G2": "pass", "G3": "pass", "G4": "pass"}
SLICE = {"split": "explore", "filters": [{"col": "b2b", "op": "eq", "value": True}]}
SPEC = {
    "id": "travel",
    "statement": "Travel fatigue",
    "kind": "explanation",
    "metric": "ats_margin",
    "group_a": SLICE,
    "group_b": "complement",
    "direction": "a_lt_b",
    "primary_test": "compare_groups",
    "controls": [],
}
DECLARED = {k: v for k, v in SPEC.items() if k != "id"}


def seed(**fields):
    track = new_seed("travel", "hypothesis-5", statement="Travel fatigue", kind="explanation")
    track.update(fields)
    return track


def declared(**fields):
    used = ["declare_primary_test", *fields.pop("used", [])]
    return seed(primary=dict(SPEC), used=used, **fields)


def after_primary(gates, **fields):
    used = ["run_primary_test", *fields.pop("used", [])]
    return declared(gates=dict(gates), primary_test_id="t_1", used=used, **fields)


def call(tool, **args):
    return ToolCall(tool=tool, args=args, reason="because")


# -- what is offered -------------------------------------------------------------------------


def test_only_declaring_and_data_tools_before_the_primary_test_is_declared():
    assert offered(seed()) == [*BASE_TOOLS, *DATA_TOOLS, "declare_primary_test"]
    assert "describe_dataset" not in offered(declared())


def test_data_lookups_are_capped_per_seed():
    lookups = [{"role": "data"}] * MAX_DATA_STEPS
    assert not set(DATA_TOOLS) & set(offered(seed(steps=lookups)))


def test_declaring_unlocks_the_primary_test_and_exploration_only():
    names = offered(declared())
    assert "run_primary_test" in names and set(EXPLORE_TOOLS) <= set(names)
    assert not {
        "declare_primary_test",
        "replicate_primary",
        "market_check",
        "holdout_confirm",
    } & set(names)


def test_follow_ups_unlock_after_the_primary_test_holds():
    names = offered(after_primary(PASS))
    assert {"replicate_primary", "market_check"} <= set(names)
    assert "run_primary_test" not in names and "holdout_confirm" not in names


def test_the_holdout_unlocks_after_g1_to_g6_and_runs_once():
    gates = {**PASS, "G5": "pass", "G6": "pass"}
    assert "holdout_confirm" in offered(
        after_primary(gates, used=["replicate_primary", "market_check"])
    )
    assert "holdout_confirm" not in offered(after_primary({**gates, "G6": "fail"}))
    spent = after_primary(gates, used=["replicate_primary", "market_check", "holdout_confirm"])
    assert "holdout_confirm" not in offered(spent)


def test_a_stopping_failure_locks_confirmation_but_not_exploration():
    names = offered(after_primary({**PASS, "G2": "fail"}))
    assert not {"replicate_primary", "market_check", "holdout_confirm"} & set(names)
    assert set(EXPLORE_TOOLS) <= set(names)


def test_a_gate_error_fails_closed_and_locks_confirmation():
    names = offered(after_primary({"error": "fail"}))
    assert not {"replicate_primary", "market_check", "holdout_confirm"} & set(names)


def test_a_failed_market_gate_only_locks_the_holdout():
    names = offered(after_primary({**PASS, "G6": "fail"}, used=["market_check"]))
    assert "replicate_primary" in names and "holdout_confirm" not in names


# -- preparing calls -------------------------------------------------------------------------


def test_a_locked_tool_is_refused_without_running():
    prepared = prepare(seed(), call("compare_groups", spec=DECLARED))
    assert prepared.error.startswith("not_offered") and prepared.registry is None


def test_invalid_arguments_name_the_field():
    prepared = prepare(seed(), call("declare_primary_test", test={"statement": "x"}))
    assert prepared.error == "invalid_args"
    assert "test.metric" in {p["field"] for p in prepared.problems}


def test_declaring_freezes_the_spec_under_the_seed_id():
    prepared = prepare(seed(), call("declare_primary_test", test=DECLARED))
    assert prepared.role == "declare" and prepared.spec["id"] == "travel"
    assert prepared.spec["primary_test"] == "compare_groups"


def test_confirmatory_tools_ignore_arguments_and_run_the_frozen_spec():
    prepared = prepare(declared(), call("run_primary_test", spec="ignored"))
    assert (prepared.registry, prepared.args) == ("compare_groups", {"spec": SPEC})
    assert prepared.gate_spec == SPEC
    replication = prepare(after_primary(PASS), call("replicate_primary"))
    assert (replication.registry, replication.args) == ("out_of_sample", {"test_id": "t_1"})
    market = prepare(after_primary(PASS), call("market_check"))
    assert market.args == {"slice": SLICE, "side": "opp"}


def test_exploratory_specs_get_their_own_ids_so_the_frozen_spec_is_never_replaced():
    first = prepare(declared(), call("regress", spec={**DECLARED, "controls": ["opp_rest_days"]}))
    assert first.args["spec"]["id"] == "travel__x1" and first.explored
    later = prepare(declared(explored=4), call("compare_groups", spec=DECLARED))
    assert later.args["spec"]["id"] == "travel__x5"
    assert later.gate_spec["primary_test"] == "compare_groups"


def test_exploratory_market_and_persistence_tests_are_gated_as_their_own_claims():
    market = prepare(declared(), call("market_test", slice=SLICE, side="over"))
    assert market.gate_spec["primary_test"] == "market_test"
    assert market.gate_spec["metric"] == "total_over"
    persistence = prepare(declared(), call("persistence", metric="ats_cover"))
    assert persistence.args == {"metric": "ats_cover", "entity": "team", "split": "explore"}


def test_replicating_the_primary_by_its_test_id_counts_as_g5():
    prepared = prepare(after_primary(PASS), call("out_of_sample", test_id="t_1"))
    assert prepared.tool == "replicate_primary" and prepared.role == "replication"
    again = prepare(
        after_primary(PASS, used=["replicate_primary"]), call("out_of_sample", test_id="t_1")
    )
    assert again.error


def test_primary_calls_use_each_tools_own_argument_shape():
    assert primary_call(SPEC) == ("compare_groups", {"spec": SPEC})
    persistence = {**SPEC, "primary_test": "persistence"}
    assert primary_call(persistence)[1] == {
        "metric": "ats_margin",
        "entity": "team",
        "split": "explore",
    }
    market = {**SPEC, "primary_test": "market_test", "direction": "a_gt_b"}
    assert primary_call(market)[1] == {"slice": SLICE, "side": "team"}
    assert market_gate_spec(market)["primary_test"] == "compare_groups"
    assert side_for({**SPEC, "metric": "total_over"}) == "under"


def test_tool_declarations_are_self_contained():
    specs = tool_specs(KIT)
    assert all(spec.parameters["type"] == "object" for spec in specs)
    assert "$ref" not in json.dumps([spec.parameters for spec in specs])
    assert LEDGER_TOOLS <= set(KIT)


# -- the judge's audit -----------------------------------------------------------------------


def judgment(**fields):
    base = {
        "step_review": "sound",
        "step_rationale": "Fine.",
        "synthesis": "Holds.",
        "stance": "supported",
        "confidence": "high",
        "claims": [JudgedClaim(text="Backed", evidence=["test_result-3", "nope"])],
        "self_check": "Checked.",
    }
    return Judgment(**{**base, **fields})


def test_the_audit_keeps_cited_claims_and_trims_unknown_ids():
    judged, notes = audit(judgment(), {"test_result-3"}, PASS)
    assert judged["claims"] == [{"text": "Backed", "evidence": ["test_result-3"]}]
    assert (judged["stance"], judged["confidence"], notes) == ("supported", "high", [])


def test_the_audit_drops_uncited_claims():
    judged, notes = audit(judgment(claims=[JudgedClaim(text="Air", evidence=[])]), set(), PASS)
    assert judged["claims"] == [] and len(notes) == 1


def test_the_judge_cannot_upgrade_past_the_gates():
    judged, notes = audit(judgment(), {"test_result-3"}, {**PASS, "G5": "fail"})
    assert judged["stance"] == "mixed" and "G5" in notes[0]
    judged, _ = audit(judgment(stance="mixed"), {"test_result-3"}, {})
    assert judged["confidence"] == "medium"  # no primary test has held yet


# -- routing ---------------------------------------------------------------------------------


def state_with(*seeds, **overrides):
    state = initial_state("run", "question")
    for track in seeds:
        state["seeds"][track["id"]] = track
    state["order"] = [s["id"] for s in seeds]
    state.update(overrides)
    return state


def test_a_queued_human_review_comes_before_anything_else():
    _, goto = route(state_with(approval=["premise_check-2"], phase="premise_test"))
    assert goto == "human_review"


def test_a_research_step_must_fit_the_budget_with_the_summary_reserve():
    active = declared(status="active")
    budget = {"max_llm_calls": 60, "max_tests": 40, "llm_calls": 55, "tests": 3}
    update, goto = route(state_with(active, current="travel", phase="research", budget=budget))
    assert goto == "conclude" and update["exhausted"]
    assert update["seeds"]["travel"]["forced_label"] == BUDGET_LABEL


def test_exhaustion_keeps_a_label_a_failed_gate_already_fixed():
    failed = after_primary({**PASS, "G2": "fail"}, status="active")
    queued = new_seed("later", "hypothesis-6", statement="Later", kind="confounder")
    update = exhaust(state_with(failed, queued, current="travel", phase="research"))
    assert update["seeds"]["travel"]["forced_label"] is None
    assert update["seeds"]["later"]["forced_label"] == BUDGET_LABEL


def test_exhaustion_before_seeding_ends_at_the_premise():
    update = exhaust(state_with(phase="seed"))
    assert update["phase"] == "summarize" and update["outcome"] == "budget"
    assert update["premise"]["label"] == BUDGET_LABEL


def test_the_step_limit_concludes_without_another_research_or_supervisor_call():
    full = declared(status="active", steps=[{"tool": "web_search"}] * MAX_STEPS_PER_SEED)
    for phase in ("research", "supervise"):
        _, goto = route(state_with(full, current="travel", phase=phase))
        assert goto == "conclude"


# -- the Gemini gateway, against a fake client -----------------------------------------------


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def generate_content(self, *, model, contents, config):
        self.requests.append(config)
        return self.response


def gateway(response):
    models = FakeModels(response)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GeminiGateway(api_key="unused", model_id="gemini-test", rpm=100, client=client), models


TOOLS = [ToolSpec("web_search", "Search.", {"type": "object", "properties": {"query": {}}})]


def test_gemini_tool_calls_return_the_reason_separately_and_force_a_call():
    function_call = SimpleNamespace(name="web_search", args={"query": "b2b", "reason": "Why."})
    llm, models = gateway(SimpleNamespace(function_calls=[function_call], text=None))
    out = asyncio.run(llm.call_tool("research", TOOLS, system="s", context={}))
    assert (out.tool, out.args, out.reason) == ("web_search", {"query": "b2b"}, "Why.")
    config = models.requests[0]
    assert config.tool_config.function_calling_config.allowed_function_names == ["web_search"]
    declared_params = config.tools[0].function_declarations[0].parameters_json_schema
    assert declared_params["required"] == ["reason"] and llm.calls == 1


def test_gemini_reply_without_a_tool_call_raises():
    llm, _ = gateway(SimpleNamespace(function_calls=None, text="I'd rather chat."))
    with pytest.raises(NoToolCall):
        asyncio.run(llm.call_tool("research", TOOLS, system="s", context={}))


def test_gemini_search_returns_grounded_sources_once_each():
    web = SimpleNamespace(uri="https://a.example", title="A", domain="a.example")
    chunk = SimpleNamespace(web=web)
    grounding = SimpleNamespace(grounding_chunks=[chunk, chunk], web_search_queries=["b2b nba"])
    response = SimpleNamespace(
        text="Summary.", candidates=[SimpleNamespace(grounding_metadata=grounding)]
    )
    llm, models = gateway(response)
    out = asyncio.run(llm.search("Why b2b?", system="s"))
    assert out.summary == "Summary." and [hit.url for hit in out.sources] == ["https://a.example"]
    assert out.search_queries == ["b2b nba"]
    assert models.requests[0].tools[0].google_search is not None
