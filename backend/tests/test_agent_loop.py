"""End-to-end runs of the research loop with the fake LLM and stub tools, judged by the real gate
(`backend/gate`)."""

import asyncio
from itertools import pairwise

import pytest

from backend.agent.graph import AgentRun
from backend.agent.ports import RegistryTools
from backend.agent.schemas import ClaimDecision, JudgedClaim, Judgment
from backend.agent.state import BUDGET_LABEL, MAX_STEPS_PER_SEED, NOT_TESTED, REVIEW_LABEL
from backend.agent.stubs import B2B_QUESTION, B2B_RESULTS, StubTools, b2b_script
from backend.agent.toolkit import EXPLORE_TOOLS, LEDGER_TOOLS
from backend.events.projection import reduce
from backend.gate.gates import StatsGate
from backend.llm.fake import FakeLLM
from backend.tools import mock
from backend.tools.registry import _TOOLS_BY_NAME
from contracts.events import Event

B2B_VERDICTS = {
    "travel_fatigue": "Real but priced in",
    "rested_opponent": "Luck",
    "altitude": "Inconclusive",
}


def make_run(script=b2b_script, results=B2B_RESULTS, llm=None, tools=None, **kwargs):
    tools = tools or StubTools(results)
    return AgentRun(
        B2B_QUESTION,
        llm=llm or FakeLLM(script),
        tools=tools,
        gate=StatsGate(tools.run),
        run_id="test",
        **kwargs,
    )


def run_to_end(run, answer=lambda review: ("approve", None)):
    async def go():
        reviews, review = [], await run.start()
        while review is not None:
            reviews.append(review)
            review = await run.resume(*answer(review))
        return reviews

    return asyncio.run(go())


def script_with(**overrides):
    """b2b_script, except where an override returns an answer for that decision."""

    def script(decision, schema, context):
        override = overrides.get(decision)
        answer = override(context) if override else None
        return b2b_script(decision, schema, context) if answer is None else answer

    return script


def verdicts(run):
    return {row["track"]: row["verdict"] for row in run.state["report"]["verdicts"]}


def nodes(run, node_type):
    return [n for n in run.log.graph.nodes.values() if n.type == node_type]


def edges(run, edge_type):
    return [e for e in run.log.graph.edges.values() if e.type == edge_type]


def calls(run, decision):
    return [c for d, c in run.context.llm.transcript if d == decision]


def tools_run(run, seed_id):
    return [step["tool"] for step in run.state["seeds"][seed_id]["steps"]]


def test_live_run_researches_each_seed_and_gives_gated_verdicts():
    run = make_run()
    assert run_to_end(run) == []

    assert run.state["status"] == "completed"
    assert verdicts(run) == B2B_VERDICTS
    events = run.log.events()
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[-1].type == "run_status" and events[-1].payload.status == "completed"
    graph = run.log.graph
    assert all(n.lifecycle == "complete" for n in graph.nodes.values())
    assert all(any(c.checker == "judge" for c in n.checks) for n in nodes(run, "test_plan"))
    gated = [n for n in nodes(run, "test_result") if n.tool.tool_name in LEDGER_TOOLS]
    assert gated and all(any(c.checker == "gate" for c in n.checks) for n in gated)
    assert run.context.llm.calls <= run.state["budget"]["max_llm_calls"]

    (report,) = nodes(run, "report")  # the final aggregation is a node on the graph
    verdict_nodes = {row["verdict_node"] for row in run.state["report"]["verdicts"]}
    assert set(graph.parents(report.id)) == verdict_nodes
    assert report.llm.rationale.startswith(run.state["report"]["answer"])


def test_the_premise_is_tested_before_any_seed_exists():
    run = make_run()
    run_to_end(run)
    order = [n.type for n in run.log.graph.nodes.values()]
    first_seed = order.index("hypothesis")
    assert order.index("premise_check") < order.index("test_result") < first_seed
    assert order.index("judgment") < order.index("source") < first_seed


def test_false_premise_ends_the_run_without_seeding_or_explaining_noise():
    no_gap = {
        "n": 36730,
        "effect": -0.2,
        "ci_low": -0.7,
        "ci_high": 0.3,
        "p_value": 0.41,
        "extra": {"n_a": 4810, "n_b": 31920, "mde": 0.42},
    }
    run = make_run(results={**B2B_RESULTS, ("b2b_gap", "compare_groups"): no_gap})
    run_to_end(run)
    assert run.state["outcome"] == "premise_false"
    assert verdicts(run) == {"premise": "Premise false"}
    assert nodes(run, "hypothesis") == [] and nodes(run, "source") == []
    assert not calls(run, "search") and not calls(run, "research")


def test_untestable_question_names_the_missing_data():
    untestable = ClaimDecision(
        testable=False, missing_data=["player tracking"], rationale="Player-level."
    )
    run = make_run(script_with(premise=lambda context: untestable))
    run_to_end(run)
    assert run.state["outcome"] == "untestable"
    assert verdicts(run) == {"premise": "Untestable"}
    assert "player tracking" in run.state["report"]["verdicts"][0]["statement"]
    assert not nodes(run, "test_result")


def test_seeds_come_from_a_web_search_and_cite_its_sources():
    run = make_run()
    run_to_end(run)
    (source,) = [n for n in nodes(run, "source") if n.title.startswith(f"Search: {B2B_QUESTION}")]
    assert source.tool.output["sources"]
    premise = nodes(run, "premise_check")[0].id
    for seed in nodes(run, "hypothesis"):
        assert set(run.log.graph.parents(seed.id)) == {premise, source.id}
    seeds = run.state["seeds"]
    assert seeds["travel_fatigue"]["sources"] == [source.id]  # cited a search result
    assert seeds["altitude"]["sources"] == []  # from reasoning alone
    travel_judge = next(c for c in calls(run, "judge") if c["seed"]["id"] == "travel_fatigue")
    assert source.id in {item["id"] for item in travel_judge["evidence"]}


def test_seeds_are_researched_one_at_a_time_and_each_step_is_judged_then_supervised():
    run = make_run()
    run_to_end(run)
    seeds = [calls_["seed"]["id"] for calls_ in calls(run, "research")]
    assert seeds == sorted(seeds, key=list(B2B_VERDICTS).index)  # never interleaved
    sequence = [d for d, _ in run.context.llm.transcript if d in ("research", "judge", "supervise")]
    assert sequence[0] == "judge"  # the premise's review
    steps = sequence[1:]
    assert steps[:3] == ["research", "judge", "supervise"]
    assert tools_run(run, "travel_fatigue") == [
        "declare_primary_test",
        "run_primary_test",
        "replicate_primary",
        "placebo",
        "market_check",
        "web_search",
    ]
    assert tools_run(run, "rested_opponent") == ["declare_primary_test", "run_primary_test"]


def test_statistical_tests_stay_locked_until_the_primary_test_is_declared():
    run = make_run()
    run_to_end(run)
    for context in calls(run, "research"):
        offered = set(context["offered_tools"])
        if context["primary_test"] is None:
            assert "declare_primary_test" in offered
            assert not offered & (LEDGER_TOOLS | {"run_primary_test"})
        else:
            assert "declare_primary_test" not in offered


def test_the_holdout_runs_once_and_only_after_g1_to_g6_pass():
    market = B2B_RESULTS[("travel_fatigue", "market_test")]
    holdout = {
        "n": 1200,
        "effect": -1.2,
        "ci_low": -2.4,
        "ci_high": 0.0,
        "p_value": 0.05,
        "extra": {
            "same_sign": True,
            "retention": 0.79,
            # holdout_confirm reports the team side; the gate mirrors it for an a_lt_b claim.
            "market": {"ci_low": 0.41, "ci_high": 0.47, "rate": 0.44},
            "spread_controlled": {"effect": -1.0, "ci_low": -1.9, "ci_high": -0.1},
        },
    }
    with_spread = {"n": 4810, "effect": -1.3, "ci_low": -2.2, "ci_high": -0.4, "p_value": 0.004}
    results = {
        **B2B_RESULTS,
        ("travel_fatigue", "market_test"): {**market, "effect": 0.556, "ci_low": 0.531},
        ("travel_fatigue", "regress"): with_spread,  # the gate's spread control for G6
        ("travel_fatigue", "holdout_confirm"): holdout,
    }
    run = make_run(results=results)
    run_to_end(run)
    offered = [
        "holdout_confirm" in c["offered_tools"]
        for c in calls(run, "research")
        if c["seed"]["id"] == "travel_fatigue"
    ]
    assert offered == [False] * 5 + [True, False]  # unlocked by G6, then spent
    assert tools_run(run, "travel_fatigue").count("holdout_confirm") == 1
    assert verdicts(run)["travel_fatigue"] == "Real and beats market (provisional)"


def test_exploratory_tests_enter_the_ledger_but_never_set_the_label():
    strong = {
        "n": 4810,
        "effect": -2.5,
        "ci_low": -3.3,
        "ci_high": -1.7,
        "p_value": 0.0001,
        "extra": {"n_a": 2400, "n_b": 2410, "mde": 0.6},
    }
    probe = {
        "statement": "Rested opponents only",
        "kind": "confounder",
        "metric": "ats_margin",
        "direction": "a_lt_b",
        "group_a": {"filters": [{"col": "opp_b2b", "op": "eq", "value": False}]},
    }

    def research(context):
        done = [s["tool"] for s in context["steps"]]
        rested = context["seed"]["id"] == "rested_opponent"
        if rested and "run_primary_test" in done and "compare_groups" not in done:
            return {"tool": "compare_groups", "args": {"spec": probe}, "reason": "Probe."}
        return None

    def supervise(context):
        done = [s["tool"] for s in context["steps"]]
        if context["seed"]["id"] == "rested_opponent" and "compare_groups" not in done:
            return {"action": "continue", "guidance": "Probe one slice.", "rationale": "Why not."}
        return None

    results = {**B2B_RESULTS, ("rested_opponent__x1", "compare_groups"): strong}
    run = make_run(script_with(research=research, supervise=supervise), results=results)
    run_to_end(run)

    assert tools_run(run, "rested_opponent")[-1] == "compare_groups"
    assert verdicts(run)["rested_opponent"] == "Luck"  # the frozen primary decides
    probe_node = next(
        n
        for n in nodes(run, "test_result")
        if n.tool.args.get("spec", {}).get("id") == "rested_opponent__x1"
    )
    assert {c.gate_id for c in probe_node.checks if c.checker == "gate"} == {"G1", "G2", "G3", "G4"}
    ledger = run.context.gate.ledger
    assert len(ledger) == 7  # premise, 3 primaries, the market test, its spread control, the probe


def test_the_judge_cannot_call_a_failed_seed_supported_or_cite_unseen_evidence():
    def judge(context):
        if context["seed"]["id"] != "rested_opponent" or not context["verdict_gates"]:
            return None
        return Judgment(
            step_review="sound",
            step_rationale="Fine.",
            synthesis="It is real.",
            stance="supported",
            confidence="high",
            claims=[
                JudgedClaim(text="Backed", evidence=[context["step"]["result_id"]]),
                JudgedClaim(text="Invented", evidence=["test_result-999"]),
            ],
            self_check="None.",
        )

    run = make_run(script_with(judge=judge))
    run_to_end(run)
    judged = run.state["seeds"]["rested_opponent"]["judgment"]
    assert judged["stance"] == "mixed" and judged["confidence"] == "medium"
    assert [c["text"] for c in judged["claims"]] == ["Backed"]
    assert len(judged["audit"]) == 3
    assert verdicts(run)["rested_opponent"] == "Luck"


def flawed_premise(times):
    left = {"n": times}

    def judge(context):
        if context["seed"]["id"] == "premise" and left["n"]:
            left["n"] -= 1
            return Judgment(
                step_review="flawed",
                step_rationale="The claim measures wins, not performance against the line.",
                synthesis="Not a test of the question.",
                stance="insufficient",
                confidence="low",
                self_check="Confident the construct is wrong.",
            )
        return None

    return judge


def test_a_flawed_premise_is_revised_once_with_the_judges_feedback():
    run = make_run(script_with(judge=flawed_premise(1)))
    run_to_end(run)
    first, second = nodes(run, "premise_check")
    assert first.check_outcome == "failed"
    assert any(e.source == second.id and e.target == first.id for e in edges(run, "revises"))
    assert second.version == first.version + 1
    assert "measures wins" in calls(run, "premise")[1]["feedback"]
    assert verdicts(run) == B2B_VERDICTS


def test_a_premise_still_flawed_after_revision_ends_the_run():
    run = make_run(script_with(judge=flawed_premise(2)))
    run_to_end(run)
    assert run.state["outcome"] == "review"
    assert verdicts(run) == {"premise": REVIEW_LABEL}
    assert not nodes(run, "hypothesis")


def test_the_supervisor_decides_when_a_seed_is_done():
    def supervise(context):
        if context["seed"]["id"] == "travel_fatigue":
            return {"action": "conclude", "rationale": "Enough for now."}
        return None

    run = make_run(script_with(supervise=supervise))
    run_to_end(run)
    assert tools_run(run, "travel_fatigue") == ["declare_primary_test"]
    assert verdicts(run) == {**B2B_VERDICTS, "travel_fatigue": NOT_TESTED}
    verdict = run.log.graph.nodes[run.state["seeds"]["travel_fatigue"]["verdict_node"]]
    assert verdict.llm.rationale == "Enough for now."  # the supervisor's decision is its provenance


def test_the_step_limit_ends_a_seed_without_asking_the_supervisor():
    def research(context):
        if context["seed"]["id"] == "altitude" and context["primary_test"]:
            return {"tool": "web_search", "args": {"query": "altitude fatigue"}, "reason": "More."}
        return None

    def supervise(context):
        if context["seed"]["id"] == "altitude":
            return {"action": "continue", "guidance": "Keep reading.", "rationale": "Curious."}
        return None

    run = make_run(
        script_with(research=research, supervise=supervise), budget={"max_llm_calls": 100}
    )
    run_to_end(run)
    assert len(tools_run(run, "altitude")) == MAX_STEPS_PER_SEED
    altitude = [c for c in calls(run, "supervise") if c["seed"]["id"] == "altitude"]
    assert len(altitude) == MAX_STEPS_PER_SEED - 1
    assert verdicts(run)["altitude"] == NOT_TESTED


def test_a_data_lookup_costs_one_call_and_skips_the_judge_and_supervisor():
    looked = {"n": 0}

    def research(context):
        if context["seed"]["id"] == "travel_fatigue" and not looked["n"]:
            looked["n"] += 1
            return {"tool": "define_slice", "args": {"spec": {"filters": []}}, "reason": "Size."}
        return None

    baseline = make_run()
    run_to_end(baseline)
    run = make_run(script_with(research=research))
    run_to_end(run)
    lookup = run.state["seeds"]["travel_fatigue"]["steps"][0]
    assert lookup["role"] == "data" and lookup["judgment"] is None
    assert run.log.graph.nodes[lookup["plan"]].check_outcome == "passed"
    assert run.context.llm.calls == baseline.context.llm.calls + 1  # just the research call
    assert verdicts(run) == B2B_VERDICTS


def test_a_call_that_cannot_run_is_fed_back_to_the_researcher():
    tries = {"n": 0}

    def research(context):
        if context["seed"]["id"] == "travel_fatigue" and not tries["n"]:
            tries["n"] += 1
            bad = {"test": {"statement": "x", "kind": "explanation", "metric": "points"}}
            return {"tool": "declare_primary_test", "args": bad, "reason": "Typo."}
        return None

    run = make_run(script_with(research=research))
    run_to_end(run)
    first = run.state["seeds"]["travel_fatigue"]["steps"][0]
    assert first["error"].startswith("invalid_args")
    plan = run.log.graph.nodes[first["plan"]]
    assert plan.check_outcome == "failed"
    retry = [c for c in calls(run, "research") if c["seed"]["id"] == "travel_fatigue"][1]
    assert "did not run" in retry["feedback"] and "test.metric" in retry["feedback"]
    assert verdicts(run) == B2B_VERDICTS


def test_approval_mode_reviews_every_decision_before_it_takes_effect():
    run = make_run(mode="approval")

    async def go():
        first = await run.start()
        built_before_approval = nodes(run, "test_result")
        reviews, review = [first], await run.resume("approve")
        while review is not None:
            reviews.append(review)
            review = await run.resume("approve")
        return first, built_before_approval, reviews

    first, built_before_approval, reviews = asyncio.run(go())
    assert first["type"] == "premise_check"
    assert built_before_approval == []
    graph = run.log.graph
    decisions = [
        nid
        for nid, n in graph.nodes.items()
        if n.type in ("premise_check", "hypothesis", "test_plan")
    ]
    assert [r["node_id"] for r in reviews] == decisions
    assert all(any(c.checker == "human" for c in graph.nodes[n].checks) for n in decisions)
    assert not any("awaiting_approval" in n.flags for n in graph.nodes.values())
    statuses = [e.payload.status for e in run.log.events() if e.type == "run_status"]
    assert statuses.count("awaiting_approval") == len(reviews)
    assert verdicts(run) == B2B_VERDICTS


def test_a_rejected_seed_is_pruned_and_never_researched():
    run = make_run(mode="approval")

    def answer(review):
        if review["type"] == "hypothesis" and "altitude" in review["title"].lower():
            return "reject", "Too few games at altitude to learn anything."
        return "approve", None

    run_to_end(run, answer)
    altitude = run.state["seeds"]["altitude"]
    assert altitude["status"] == "pruned" and altitude["steps"] == []
    assert "pruned" in run.log.graph.nodes[altitude["node"]].flags
    assert verdicts(run) == {k: v for k, v in B2B_VERDICTS.items() if k != "altitude"}
    summary = calls(run, "summarize")[0]
    assert summary["pruned_by_reviewer"] == [altitude["statement"]]


def test_a_rejected_tool_call_never_runs_and_the_note_reaches_the_researcher():
    run = make_run(mode="approval")
    rejected = {}

    def answer(review):
        if review["type"] == "test_plan" and not rejected:
            rejected["node"] = review["node_id"]
            return "reject", "Check group sizes with define_slice first."
        return "approve", None

    run_to_end(run, answer)
    graph = run.log.graph
    assert graph.nodes[rejected["node"]].check_outcome == "failed"
    assert not [e for e in graph.edges.values() if e.target == rejected["node"]]  # nothing ran
    retry = [c for c in calls(run, "research") if c["seed"]["id"] == "travel_fatigue"][1]
    assert "define_slice first" in retry["feedback"]
    assert verdicts(run) == B2B_VERDICTS


def test_mode_switches_take_effect_at_the_next_step():
    run = make_run()
    run.set_mode("approval")

    async def go():
        review = await run.start()
        run.set_mode("live")
        return review, await run.resume("approve")

    review, after = asyncio.run(go())
    assert review["type"] == "premise_check"
    assert after is None
    assert run.state["status"] == "completed"


def test_budget_exhaustion_gives_unfinished_seeds_rule_based_labels():
    run = make_run(budget={"max_llm_calls": 20})
    run_to_end(run)
    assert run.state["status"] == "completed"
    assert run.context.llm.calls <= 20
    assert BUDGET_LABEL in verdicts(run).values()
    assert calls(run, "summarize")  # the summary's reserve was kept


def test_a_g4_flip_at_report_time_revises_the_verdict():
    # p=0.02 passes BH on the two-entry ledger it arrives on, not on the final one.
    first = B2B_RESULTS[("travel_fatigue", "compare_groups")]
    results = {**B2B_RESULTS, ("travel_fatigue", "compare_groups"): {**first, "p_value": 0.02}}
    run = make_run(results=results)
    run_to_end(run)

    assert verdicts(run)["travel_fatigue"] == "Luck"
    rechecks = [
        e.payload.check
        for e in run.log.events()
        if e.type == "check_result" and e.payload.check.rationale.startswith("Re-evaluated")
    ]
    seed = run.state["seeds"]["travel_fatigue"]
    assert [(c.node_id, c.gate_id, c.status) for c in rechecks] == [
        (seed["primary_result"], "G4", "fail")
    ]
    graph = run.log.graph
    assert graph.nodes[seed["primary_result"]].check_outcome == "failed"
    revised = graph.nodes[seed["verdict_node"]]
    assert revised.title.startswith("Luck: ") and revised.version == 2
    assert graph.nodes[seed["node"]].verdict.label == "Luck"


def test_event_log_replays_to_the_same_graph():
    run = make_run()
    run_to_end(run)
    assert reduce(run.log.events()) == run.log.graph

    again = make_run()
    run_to_end(again)

    def shape(graph):
        return {
            nid: (n.type, n.title, n.check_outcome, sorted(n.flags))
            for nid, n in graph.nodes.items()
        }, sorted(graph.edges)

    assert shape(again.log.graph) == shape(run.log.graph)


def test_the_event_log_is_valid_replay_input_for_the_api():
    run = make_run(mode="approval")
    run_to_end(run)
    lines = run.log.to_jsonl().splitlines()
    events = [Event.model_validate_json(line) for line in lines]  # what POST /replay does
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert all(a.ts <= b.ts for a, b in pairwise(events))
    assert events[0].type == "run_created"
    assert reduce(events) == run.log.graph
    verdicts_out = [e.payload for e in events if e.type == "verdict_computed"]
    assert {v.verdict.label for v in verdicts_out} == {"Real but priced in", "Luck", "Inconclusive"}


def test_edges_follow_the_contract_directions():
    run = make_run(script_with(judge=flawed_premise(1)))
    run_to_end(run)
    order = list(run.log.graph.nodes)  # creation order
    for edge in run.log.graph.edges.values():
        if edge.type in ("derives_from", "revises"):  # child -> parent, new -> old
            assert order.index(edge.source) > order.index(edge.target), edge
        else:  # supports / refutes: the primary test result -> its seed
            assert run.log.graph.nodes[edge.source].type == "test_result"
            assert run.log.graph.nodes[edge.target].type == "hypothesis"


def test_every_registry_call_fits_person_a_tool_schemas():
    run = make_run()
    run_to_end(run)
    registry_calls = [
        n.tool
        for n in nodes(run, "test_result")
        if n.tool.tool_name in _TOOLS_BY_NAME and n.tool.output["status"] == "ok"
    ]
    assert {c.tool_name for c in registry_calls} >= {
        "compare_groups",
        "regress",
        "out_of_sample",
        "placebo",
        "market_test",
    }
    for call in registry_calls:
        _TOOLS_BY_NAME[call.tool_name].args_model.model_validate(call.args)


def test_explore_tools_are_offered_only_after_declaring():
    run = make_run()
    run_to_end(run)
    first = calls(run, "research")[0]
    assert not set(first["offered_tools"]) & set(EXPLORE_TOOLS)


@pytest.fixture
def fresh_mock_tools():
    mock.reset_mock_state()
    yield
    mock.reset_mock_state()


def test_the_loop_runs_on_person_a_mock_tools(fresh_mock_tools):
    run = make_run(tools=RegistryTools())
    run_to_end(run)
    assert run.state["status"] == "completed"
    results = [
        n.tool.output for n in nodes(run, "test_result") if n.tool.tool_name in _TOOLS_BY_NAME
    ]
    assert results and all(r["status"] == "ok" for r in results), [
        (r["tool"], r["reason"]) for r in results if r["status"] != "ok"
    ]
