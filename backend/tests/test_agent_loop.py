"""End-to-end runs of the stateful router with the fake LLM, stub tools and stub gate."""

import asyncio

from backend.agent.graph import AgentRun
from backend.agent.schemas import ClaimDecision, CriticChecklist, CriticReview
from backend.agent.state import result_steps
from backend.agent.stubs import (
    B2B_QUESTION,
    B2B_RESULTS,
    NOT_REPLICATED,
    StubGate,
    StubTools,
    b2b_script,
)
from backend.events.projection import reduce
from backend.llm.fake import FakeLLM

B2B_VERDICTS = {
    "travel_fatigue": "Real but priced in",
    "rested_opponent": "Luck",
    "altitude": "Inconclusive",
    "chance": "Chance supported",
}


def make_run(script=b2b_script, results=B2B_RESULTS, llm=None, **kwargs):
    return AgentRun(
        B2B_QUESTION,
        llm=llm or FakeLLM(script),
        tools=StubTools(results),
        gate=StubGate(),
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


def verdicts(run):
    return {row["track"]: row["verdict"] for row in run.state["report"]["verdicts"]}


def nodes(run, node_type):
    return [n for n in run.log.graph.nodes.values() if n["type"] == node_type]


def edges(run, edge_type):
    return [e for e in run.log.graph.edges.values() if e["type"] == edge_type]


def objecting_critic(decision_type, times=1, verdict="concerns"):
    left = {"n": times}

    def script(decision, schema, context):
        if decision == "critic" and context["decision_type"] == decision_type and left["n"]:
            left["n"] -= 1
            return CriticReview(
                verdict=verdict,
                checklist=CriticChecklist(construct_validity="problem"),
                severity="medium",
                evidence=[context["node_id"]],
                rationale="Wrong comparison group.",
                suggested_fix="Compare with rested teams.",
            )
        return b2b_script(decision, schema, context)

    return script


class SlowPlanCritic(FakeLLM):
    """Delays critic reviews of test plans so the loop builds on them before they land."""

    async def structured(self, decision, schema, *, system, context, meta=None):
        if decision == "critic" and context["decision_type"] == "test_plan":
            await asyncio.sleep(0.05)
        return await super().structured(decision, schema, system=system, context=context)


def test_live_run_gives_every_hypothesis_a_gated_verdict():
    run = make_run()
    assert run_to_end(run) == []

    assert run.state["status"] == "completed"
    assert verdicts(run) == B2B_VERDICTS
    events = run.log.events()
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[-1].kind == "run_status" and events[-1].data["status"] == "completed"
    graph = run.log.graph
    decisions = [n for n in graph.nodes.values() if n.get("llm")]
    assert decisions
    assert all(any(c["checker"] == "critic" for c in n["checks"]) for n in decisions)
    gated = [n for n in nodes(run, "test_result") if n["tool_call"]["tool"] != "placebo"]
    assert all(any(c["checker"] == "gate" for c in n["checks"]) for n in gated)
    assert run.context.llm.calls <= run.state["budget"]["max_llm_calls"]


def menus_offered(run):
    return [list(c["menu"]) for d, c in run.context.llm.transcript if d == "choose_test"]


def tools_run(run, track_id):
    return [step["tool"] for step in result_steps(run.state["tracks"][track_id])]


def test_the_llm_picks_follow_ups_only_after_gates_pass():
    run = make_run()
    run_to_end(run)
    assert tools_run(run, "travel_fatigue") == [
        "compare_groups",
        "out_of_sample",
        "placebo",
        "market_test",
    ]
    assert tools_run(run, "rested_opponent") == ["regress"]  # G2 failed: testing stopped
    assert tools_run(run, "altitude") == ["compare_groups"]  # G1 failed: testing stopped
    menus = menus_offered(run)
    assert len(menus) == 3  # only travel_fatigue ever passed a gate
    assert all("holdout_confirm" not in menu for menu in menus)  # G6 failed: holdout locked


def test_the_holdout_runs_once_and_only_after_g1_to_g6_pass():
    market = B2B_RESULTS[("travel_fatigue", "market_test")]
    holdout = {
        "n": 1200,
        "effect": -1.2,
        "ci_low": -2.4,
        "ci_high": 0.0,
        "p_value": 0.05,
        "extra": {"same_sign": True, "retention": 0.79},
    }
    results = {
        **B2B_RESULTS,
        ("travel_fatigue", "market_test"): {**market, "effect": 0.556, "ci_low": 0.531},
        ("travel_fatigue", "holdout_confirm"): holdout,
    }
    run = make_run(results=results)
    run_to_end(run)
    assert ["holdout_confirm" in menu for menu in menus_offered(run)] == [False, False, False, True]
    assert tools_run(run, "travel_fatigue").count("holdout_confirm") == 1
    assert verdicts(run)["travel_fatigue"] == "Real and beats market (provisional)"


def test_concluding_before_replication_cannot_make_a_hypothesis_real():
    def script(decision, schema, context):
        if decision == "choose_test":
            return {"action": "conclude", "rationale": "Stop early."}
        return b2b_script(decision, schema, context)

    run = make_run(script)
    run_to_end(run)
    assert tools_run(run, "travel_fatigue") == ["compare_groups"]
    assert verdicts(run) == {**B2B_VERDICTS, "travel_fatigue": NOT_REPLICATED}


def test_false_premise_ends_the_run_without_explaining_noise():
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
    assert nodes(run, "hypothesis") == []


def test_untestable_question_names_the_missing_data():
    def script(decision, schema, context):
        if decision == "premise":
            return ClaimDecision(
                testable=False, missing_data=["player tracking"], rationale="Player-level."
            )
        return b2b_script(decision, schema, context)

    run = make_run(script)
    run_to_end(run)
    assert run.state["outcome"] == "untestable"
    assert verdicts(run) == {"premise": "Untestable"}
    assert nodes(run, "premise_check")[0]["missing_data"] == ["player tracking"]
    assert not nodes(run, "test_result")


def test_critic_concern_sends_the_track_back_with_its_feedback():
    run = make_run(objecting_critic("test_plan"))
    run_to_end(run)

    first, *_later = nodes(run, "test_plan")
    assert first["state"] == "flagged"
    assert any(e["source"] == first["id"] for e in edges(run, "revises"))
    assert run.state["tracks"]["travel_fatigue"]["revisions"] == 1
    replans = [c for d, c in run.context.llm.transcript if d == "plan_test" and c["feedback"]]
    assert "Wrong comparison group" in replans[0]["feedback"]
    assert verdicts(run) == B2B_VERDICTS


def test_late_rejection_taints_descendants_and_reruns_the_track():
    run = make_run(llm=SlowPlanCritic(objecting_critic("test_plan", verdict="reject")))
    run_to_end(run)

    graph = run.log.graph
    first = nodes(run, "test_plan")[0]
    downstream = graph.descendants(first["id"])
    assert first["state"] == "failed"
    assert downstream, "the loop should have built on the plan before the critic landed"
    assert all(graph.nodes[nid]["state"] == "tainted" for nid in downstream)
    reported = [row["verdict_node"] for row in run.state["report"]["verdicts"]]
    assert all(graph.nodes[nid]["state"] != "tainted" for nid in reported)
    assert verdicts(run) == B2B_VERDICTS


def test_revision_limit_ends_the_track_as_inconclusive_review():
    run = make_run(objecting_critic("test_plan", times=3))
    run_to_end(run)
    assert run.state["tracks"]["travel_fatigue"]["revisions"] == 2
    assert verdicts(run) == {**B2B_VERDICTS, "travel_fatigue": "Inconclusive (review)"}


def test_approval_mode_pauses_before_building_on_each_decision():
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
    decisions = [nid for nid, n in graph.nodes.items() if n.get("llm")]
    assert [r["node_id"] for r in reviews] == decisions
    assert all(any(c["checker"] == "human" for c in graph.nodes[n]["checks"]) for n in decisions)
    assert verdicts(run) == B2B_VERDICTS


def test_rejection_needs_a_note_and_triggers_a_revision():
    run = make_run(mode="approval")

    async def go():
        review = await run.start()
        retry = await run.resume("reject", "no")
        revised = await run.resume("reject", "Measure covers with ats_cover instead.")
        review_after = revised
        while review_after is not None:
            review_after = await run.resume("approve")
        return review, retry, revised

    review, retry, revised = asyncio.run(go())
    assert retry["node_id"] == review["node_id"] and "error" in retry
    assert revised["type"] == "premise_check" and revised["node_id"] != review["node_id"]
    first, second = nodes(run, "premise_check")
    assert first["state"] == "failed"
    assert any(
        e["source"] == first["id"] and e["target"] == second["id"] for e in edges(run, "revises")
    )
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


def test_budget_exhaustion_marks_unfinished_hypotheses_inconclusive():
    run = make_run(budget={"max_llm_calls": 20})
    run_to_end(run)
    assert run.state["status"] == "completed"
    assert run.context.llm.calls <= 20
    assert "Inconclusive (budget)" in verdicts(run).values()


def test_event_log_replays_to_the_same_graph():
    run = make_run()
    run_to_end(run)
    assert reduce(run.log.events()) == run.log.graph

    again = make_run()
    run_to_end(again)

    def shape(graph):
        return {nid: (n["type"], n["title"], n["state"]) for nid, n in graph.nodes.items()}, sorted(
            graph.edges
        )

    assert shape(again.log.graph) == shape(run.log.graph)
