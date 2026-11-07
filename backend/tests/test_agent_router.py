"""Pure routing and check-folding rules. No LLM, no LangGraph, no event loop."""

from backend.agent.checks import CheckBatch
from backend.agent.ports import CheckResult
from backend.agent.router import fold_checks, follow_up_menu, route
from backend.agent.state import (
    BUDGET_LABEL,
    PREMISE,
    PREMISE_FALSE,
    REVIEW_LABEL,
    initial_state,
    new_track,
)
from backend.events.projection import ResearchGraph
from contracts.models import Edge

PRIMARY_PASS = {"G1": "pass", "G2": "pass", "G3": "pass", "G4": "pass"}


def step(node, stage, tool=None, gates=None):
    return {"node": node, "stage": stage, "tool": tool, "gates": gates}


SPEC = step("hypothesis-3", "spec")
PLAN = step("test_plan-4", "plan", "compare_groups")
PRIMARY = step("test_result-5", "primary", "compare_groups", PRIMARY_PASS)
CHOOSE_OOS = step("test_plan-6", "choose", "out_of_sample")
OOS = step("test_result-7", "follow_up", "out_of_sample", {"G5": "pass"})
CHOOSE_MARKET = step("test_plan-8", "choose", "market_test")
MARKET = step("test_result-9", "follow_up", "market_test", {"G6": "pass"})


def state_with(*tracks, **overrides):
    state = initial_state("run", "question")
    state["question_node"] = "question-1"
    for track in tracks:
        state["tracks"][track["id"]] = track
    state["order"] = [t["id"] for t in tracks if t["id"] != PREMISE]
    state.update(overrides)
    return state


def held_premise():
    return new_track(PREMISE, "premise", "done")


def hypothesis(stage, *steps, track_id="h1", kind="explanation", **fields):
    track = new_track(track_id, kind, stage)
    track["steps"] = [dict(s) for s in steps]
    track.update(fields)
    return track


def critic(node_id, status, rationale="Missing control for opponent rest."):
    result = CheckResult(node_id=node_id, checker="critic", status=status, rationale=rationale)
    return CheckBatch(node_id, "critic", [result])


def test_router_starts_with_the_premise_check():
    update, goto = route(state_with())
    assert (goto, update["current"]) == ("premise", PREMISE)


def test_router_hypothesizes_only_after_the_premise_holds():
    update, goto = route(state_with(held_premise()))
    assert (goto, update["current"]) == ("hypothesize", None)


def test_false_premise_goes_to_its_verdict_then_the_report():
    gates = {**PRIMARY_PASS, "G2": "fail"}
    premise = new_track(PREMISE, "premise", "gated")
    premise["steps"] = [step("premise_check-2", "spec"), step("r", "primary", "x", gates)]
    update, goto = route(state_with(premise))
    assert goto == "verdict"
    assert update["outcome"] == "premise_false"
    assert update["tracks"][PREMISE]["forced_label"] == PREMISE_FALSE

    settled = {**update["tracks"][PREMISE], "stage": "done"}
    assert route(state_with(settled, outcome="premise_false"))[1] == "report"


def test_nothing_builds_on_a_result_before_its_gates_land():
    pending = step("test_result-5", "primary", "compare_groups", None)
    track = hypothesis("choose", SPEC, PLAN, pending)
    update, goto = route(state_with(held_premise(), track, awaiting_gate=["test_result-5"]))
    assert (goto, update["waiting_on"]) == ("wait_checks", ["test_result-5"])


def test_the_llm_chooses_the_next_test_once_the_gates_pass():
    update, goto = route(state_with(held_premise(), hypothesis("choose", SPEC, PLAN, PRIMARY)))
    assert (goto, update["current"]) == ("choose_test", "h1")


def test_a_failed_gate_stops_testing_the_hypothesis():
    failed = step("test_result-5", "primary", "compare_groups", {**PRIMARY_PASS, "G2": "fail"})
    update, goto = route(state_with(held_premise(), hypothesis("choose", SPEC, PLAN, failed)))
    assert goto == "interpret"
    assert update["tracks"]["h1"]["stage"] == "interpret"


def test_a_failed_market_gate_only_locks_the_holdout():
    priced_in = step("test_result-9", "follow_up", "market_test", {"G6": "fail"})
    track = hypothesis("choose", SPEC, PLAN, PRIMARY, CHOOSE_MARKET, priced_in)
    assert route(state_with(held_premise(), track))[1] == "choose_test"
    assert follow_up_menu(track) == ("out_of_sample", "placebo", "conclude")


def test_the_holdout_unlocks_only_after_g1_to_g6_pass():
    replicated = hypothesis("choose", SPEC, PLAN, PRIMARY, CHOOSE_OOS, OOS)
    assert "holdout_confirm" not in follow_up_menu(replicated)

    beats_market = hypothesis("choose", SPEC, PLAN, PRIMARY, CHOOSE_OOS, OOS, CHOOSE_MARKET, MARKET)
    assert follow_up_menu(beats_market) == ("placebo", "holdout_confirm", "conclude")

    beats_market["holdout_used"] = True
    assert follow_up_menu(beats_market) == ("placebo", "conclude")


def test_no_llm_call_when_nothing_is_left_to_choose():
    placebo = step("test_result-11", "follow_up", "placebo", {})
    track = hypothesis(
        "choose",
        SPEC,
        PLAN,
        PRIMARY,
        CHOOSE_OOS,
        OOS,
        CHOOSE_MARKET,
        MARKET,
        placebo,
        holdout_used=True,
    )
    assert route(state_with(held_premise(), track))[1] == "interpret"


def test_chance_verdict_waits_until_competitors_settle():
    chance = hypothesis("verdict", track_id="chance", kind="chance")
    update, goto = route(state_with(held_premise(), chance, awaiting_critic=["verdict-9"]))
    assert (goto, update["waiting_on"]) == ("wait_checks", ["verdict-9"])


def test_budget_exhaustion_forces_rule_based_verdicts():
    state = state_with(held_premise(), hypothesis("plan", SPEC))
    state["budget"]["llm_calls"] = state["budget"]["max_llm_calls"] - 2
    update, goto = route(state)
    assert goto == "verdict"
    assert update["exhausted"]
    assert update["tracks"]["h1"]["forced_label"] == BUDGET_LABEL


def test_pending_critics_count_against_the_budget():
    state = state_with(held_premise(), hypothesis("plan", SPEC), awaiting_critic=["a", "b", "c"])
    state["budget"]["llm_calls"] = state["budget"]["max_llm_calls"] - 6
    update, goto = route(state)
    assert goto == "verdict" and update["exhausted"]


def test_approval_blocks_until_the_critic_lands_then_asks_the_human():
    track = hypothesis("primary", SPEC, PLAN)
    state = state_with(
        held_premise(),
        track,
        mode="approval",
        approval=["test_plan-4"],
        awaiting_critic=["test_plan-4"],
    )
    assert route(state)[1] == "wait_checks"
    state["awaiting_critic"] = []
    assert route(state)[1] == "human_review"


def test_critic_concern_rewinds_the_track_and_taints_what_was_built_on_it():
    track = hypothesis("choose", SPEC, PLAN, PRIMARY)
    built_on_plan = Edge(id="e", source="test_result-5", target="test_plan-4", type="derives_from")
    graph = ResearchGraph(edges={"e": built_on_plan})  # derives_from points child -> parent
    state = state_with(held_premise(), track, awaiting_critic=["test_plan-4"])
    draft = fold_checks(state, [critic("test_plan-4", "flag")], graph)

    rewound = draft.tracks["h1"]
    assert rewound["stage"] == "plan"
    assert rewound["steps"] == [SPEC]
    assert rewound["revisions"] == 1
    assert rewound["feedback"] == "Missing control for opponent rest."
    assert rewound["revising"] == "test_plan-4"
    assert ("test_plan-4", "flagged") in draft.node_states
    assert ("test_result-5", "tainted") in draft.node_states
    assert draft.awaiting_critic == []


def test_a_disputed_choice_is_remade_without_rerunning_earlier_tests():
    track = hypothesis("follow_up", SPEC, PLAN, PRIMARY, CHOOSE_OOS, OOS, CHOOSE_MARKET)
    state = state_with(held_premise(), track, awaiting_critic=["test_plan-6"])
    draft = fold_checks(state, [critic("test_plan-6", "flag")], ResearchGraph())
    assert draft.tracks["h1"]["stage"] == "choose"
    assert draft.tracks["h1"]["steps"] == [SPEC, PLAN, PRIMARY]


def test_revision_limit_ends_the_track_instead_of_looping():
    track = hypothesis("primary", SPEC, PLAN, revisions=2)
    state = state_with(held_premise(), track, awaiting_critic=["test_plan-4"])
    draft = fold_checks(state, [critic("test_plan-4", "fail")], ResearchGraph())
    assert draft.tracks["h1"]["stage"] == "verdict"
    assert draft.tracks["h1"]["forced_label"] == REVIEW_LABEL


def test_the_holdout_is_never_rerun_by_a_revision():
    holdout = step("test_result-11", "follow_up", "holdout_confirm", {"G7": "pass"})
    track = hypothesis("choose", SPEC, PLAN, PRIMARY, holdout, holdout_used=True)
    state = state_with(held_premise(), track, awaiting_critic=["test_plan-4"])
    draft = fold_checks(state, [critic("test_plan-4", "flag")], ResearchGraph())
    assert draft.tracks["h1"]["forced_label"] == REVIEW_LABEL
    assert draft.tracks["h1"]["revisions"] == 0


def test_a_rejected_premise_discards_every_hypothesis():
    premise = new_track(PREMISE, "premise", "done")
    premise["steps"] = [step("premise_check-2", "spec")]
    state = state_with(premise, hypothesis("plan", SPEC), awaiting_critic=["premise_check-2"])
    draft = fold_checks(state, [critic("premise_check-2", "fail")], ResearchGraph())
    assert set(draft.tracks) == {PREMISE}
    assert draft.order == []
    assert draft.tracks[PREMISE]["stage"] == "spec"


def test_checks_on_superseded_nodes_steer_nothing():
    track = hypothesis("plan", SPEC)
    state = state_with(held_premise(), track, awaiting_critic=["test_plan-4"])
    draft = fold_checks(state, [critic("test_plan-4", "fail")], ResearchGraph())
    assert draft.tracks["h1"] == track
    assert draft.node_states == []
    assert draft.awaiting_critic == []
