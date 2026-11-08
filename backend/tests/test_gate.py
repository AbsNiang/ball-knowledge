"""The stats gate (G1-G7), its ledger and the verdict table (plan.md v2 sections 9.1-9.2)."""

import math

import numpy as np
import pytest

from backend.data.store import save_tables
from backend.data.tables import build_team_games
from backend.gate import config
from backend.gate.gates import StatsGate, overridable
from backend.gate.verdict import (
    CHANCE_REFUTED,
    CHANCE_SUPPORTED,
    CHANCE_UNDETERMINED,
    INCONCLUSIVE,
    LUCK,
    NOT_REPLICATED,
    PRICED_IN,
    REAL,
    compute_verdict,
    resolve_chance,
)
from backend.tests.test_tools_real import spec as planted_spec
from backend.tests.test_tools_real import synthetic_games
from backend.tools import data
from backend.tools.registry import dispatch, reset_tool_state
from backend.tools.schemas import ToolResult
from backend.tools.stats import Z_95, min_detectable_effect

SPEC = {
    "id": "h",
    "statement": "Back-to-back teams have a lower ATS margin",
    "kind": "explanation",
    "metric": "ats_margin",
    "group_a": {"split": "explore", "filters": [{"col": "b2b", "op": "eq", "value": True}]},
    "group_b": "complement",
    "direction": "a_lt_b",
    "primary_test": "compare_groups",
    "controls": [],
}


def primary(key=1, *, effect=-1.2, ci=(-1.7, -0.7), p=0.001, n_a=1000, n_b=1000, mde=0.5):
    return ToolResult(
        tool="compare_groups",
        args={"spec": {**SPEC, "statement": f"variant {key}"}},
        args_hash=f"p{key}",
        n=n_a + n_b,
        effect=effect,
        ci_low=ci[0],
        ci_high=ci[1],
        p_value=p,
        extra={"n_a": n_a, "n_b": n_b, "mde": mde},
    )


def follow_up(tool, **fields):
    extra = fields.pop("extra", {})
    return ToolResult(tool=tool, args=fields.pop("args", {}), args_hash=tool, extra=extra, **fields)


def no_tools(tool, args):
    raise AssertionError(f"the gate should not run {tool}")


def statuses(checks):
    return {check.gate_id: check.status for check in checks}


# -- which gates apply ------------------------------------------------------------------------


def test_a_primary_test_gets_g1_to_g4():
    checks = StatsGate(no_tools).evaluate("n1", primary(), SPEC)
    assert statuses(checks) == {"G1": "pass", "G2": "pass", "G3": "pass", "G4": "pass"}
    assert all(check.node_id == "n1" for check in checks)


def test_diagnostics_get_no_gates():
    gate = StatsGate(no_tools)
    assert (
        gate.evaluate("n", follow_up("placebo", effect=0.1, ci_low=-0.5, ci_high=0.7), SPEC) == []
    )
    assert gate.evaluate("n", follow_up("power", n=2000, extra={"mde": 0.4}), SPEC) == []


def test_a_tool_error_fails_every_gate_the_result_would_get():
    error = ToolResult(
        tool="compare_groups", args={}, status="error", reason="empty_group", args_hash="e"
    )
    checks = StatsGate(no_tools).evaluate("n", error, SPEC)
    assert statuses(checks) == {"G1": "fail", "G2": "fail", "G3": "fail", "G4": "fail"}
    assert all("empty_group" in check.rationale for check in checks)


def test_missing_inputs_fail_closed():
    bare = follow_up("compare_groups", args={"spec": SPEC}, effect=-1.0)
    assert set(statuses(StatsGate(no_tools).evaluate("n", bare, SPEC)).values()) == {"fail"}


# -- G1-G3 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change", [{"n_a": config.N_MIN - 1}, {"n_b": config.N_MIN - 1}, {"mde": 1.4}]
)
def test_g1_needs_both_groups_large_and_powered_for_the_smallest_effect(change):
    checks = StatsGate(no_tools).evaluate("n", primary(**change), SPEC)
    assert statuses(checks)["G1"] == "fail"


def test_g1_judges_power_in_the_metrics_units():
    rate_spec = {**SPEC, "metric": "ats_cover"}
    points = StatsGate(no_tools).evaluate("n", primary(mde=0.5), SPEC)
    rate = StatsGate(no_tools).evaluate("n", primary(mde=0.05), rate_spec)
    assert statuses(points)["G1"] == "pass"  # 0.5 points <= 1 point
    assert statuses(rate)["G1"] == "fail"  # 5 percentage points > 3


def test_g2_needs_the_ci_on_the_declared_side_of_zero():
    wrong_side = primary(effect=1.2, ci=(0.7, 1.7))
    assert statuses(StatsGate(no_tools).evaluate("n", wrong_side, SPEC))["G2"] == "fail"
    two_sided = {**SPEC, "direction": "two_sided"}
    assert statuses(StatsGate(no_tools).evaluate("n", wrong_side, two_sided))["G2"] == "pass"
    spans_zero = primary(ci=(-1.7, 0.2))
    assert statuses(StatsGate(no_tools).evaluate("n", spans_zero, SPEC))["G2"] == "fail"


def test_g3_needs_p_below_alpha():
    checks = StatsGate(no_tools).evaluate("n", primary(p=config.ALPHA), SPEC)
    assert statuses(checks)["G3"] == "fail"


# -- G4 and the ledger ------------------------------------------------------------------------


def test_only_a_changed_spec_adds_a_ledger_entry():
    gate = StatsGate(no_tools)
    gate.evaluate("a", primary(1), SPEC)
    gate.evaluate("b", primary(1), SPEC)  # identical spec: the same entry
    assert len(gate.ledger) == 1
    gate.evaluate("c", primary(2), SPEC)
    gate.evaluate(
        "d", follow_up("out_of_sample", extra={"same_sign": True, "retention": 0.8}), SPEC
    )
    assert len(gate.ledger) == 2  # out_of_sample is a diagnostic


def test_g4_corrects_over_every_test_in_the_run():
    gate = StatsGate(no_tools)
    assert statuses(gate.evaluate("first", primary(1, p=0.03), SPEC))["G4"] == "pass"  # alone
    for key in range(2, 6):
        gate.evaluate(f"n{key}", primary(key, p=0.6), SPEC)
    late = gate.evaluate("late", primary(9, p=0.03), SPEC)
    assert statuses(late)["G4"] == "fail"  # 0.03 > 1/6 * 0.05 once six tests have run


def test_g4_reevaluation_reports_only_the_flips():
    gate = StatsGate(no_tools)
    gate.evaluate("early", primary(1, p=0.02), SPEC)  # passes on a ledger of one
    for key in range(2, 4):
        gate.evaluate(f"big{key}", primary(key, p=0.6), SPEC)
    gate.evaluate("middle", primary(4, p=0.03), SPEC)  # fails on a ledger of four
    for key in range(5, 9):
        gate.evaluate(f"tiny{key}", primary(key, p=0.0001), SPEC)

    flips = gate.reevaluate()
    assert {check.node_id: check.status for check in flips} == {"middle": "pass"}
    assert "fail -> pass" in flips[0].rationale
    assert gate.reevaluate() == []  # already applied

    for key in range(9, 30):
        gate.evaluate(f"noise{key}", primary(key, p=0.9), SPEC)
    flipped = {check.node_id: check.status for check in gate.reevaluate()}
    assert flipped == {"early": "fail", "middle": "fail"}


# -- G5-G7 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"same_sign": True, "retention": 0.7}, "pass"),
        ({"same_sign": True, "retention": 0.4}, "fail"),
        ({"same_sign": False, "retention": 0.9}, "fail"),
        ({}, "fail"),
    ],
)
def test_g5_needs_the_same_sign_and_half_the_effect_on_validation(extra, expected):
    result = follow_up("out_of_sample", effect=-0.8, extra=extra)
    assert statuses(StatsGate(no_tools).evaluate("n", result, SPEC)) == {"G5": expected}


def spread_tool(ci=(-1.9, -0.3), status="ok"):
    calls = []

    def run_tool(tool, args):
        calls.append((tool, args))
        return ToolResult(
            tool=tool,
            args=args,
            status=status,
            reason=None if status == "ok" else "collinear_controls",
            args_hash="spread",
            effect=sum(ci) / 2,
            ci_low=ci[0],
            ci_high=ci[1],
            p_value=0.01,
        )

    return run_tool, calls


def market(ci_low=0.531):
    args = {"slice": SPEC["group_a"], "side": "opp"}
    return follow_up(
        "market_test", args=args, n=1650, effect=0.03, ci_low=ci_low, ci_high=0.58, p_value=0.02
    )


def test_g6_needs_the_cover_rate_and_a_spread_controlled_effect():
    run_tool, calls = spread_tool()
    gate = StatsGate(run_tool)
    checks = gate.evaluate("m", market(), SPEC)
    assert statuses(checks) == {"G6": "pass"}
    assert calls == [("regress", {"spec": {**SPEC, "controls": ["spread"]}})]
    assert "regress spread" in checks[0].evidence[0]
    assert len(gate.ledger) == 2  # the market test and the spread-controlled regress


@pytest.mark.parametrize(
    ("ci_low", "spread_ci", "status"),
    [
        (0.510, (-1.9, -0.3), "ok"),  # the bet does not clear break-even
        (0.531, (-1.9, 0.2), "ok"),  # the effect is the spread in disguise
        (0.531, (-1.9, -0.3), "error"),  # the control could not be run
    ],
)
def test_g6_fails_if_either_half_fails(ci_low, spread_ci, status):
    run_tool, _ = spread_tool(spread_ci, status)
    assert statuses(StatsGate(run_tool).evaluate("m", market(ci_low), SPEC)) == {"G6": "fail"}


def holdout(**changes):
    extra = {
        "same_sign": True,
        "retention": 0.8,
        "market": {"ci_low": 0.41, "ci_high": 0.47, "rate": 0.44},  # team side
        "spread_controlled": {"effect": -1.0, "ci_low": -1.9, "ci_high": -0.1},
        **changes,
    }
    return follow_up("holdout_confirm", args={"hypothesis_id": "h"}, effect=-1.0, extra=extra)


def test_g7_mirrors_the_team_side_for_a_claim_against_the_team():
    checks = StatsGate(no_tools).evaluate("h", holdout(), SPEC)
    assert statuses(checks) == {"G7": "pass"}
    assert "opp cover CI [+0.53, +0.59]" in checks[0].rationale


@pytest.mark.parametrize(
    "change",
    [
        {"retention": 0.3},
        {"market": {"ci_low": 0.47, "ci_high": 0.52}},  # opp side [0.48, 0.53]
        {"spread_controlled": {"effect": -0.2, "ci_low": -1.1, "ci_high": 0.7}},
        {"market": {"error": "no_bets"}},
        {"market": None, "spread_controlled": None},  # the mock tools return neither
    ],
)
def test_g7_needs_replication_and_the_market_rule_on_the_holdout(change):
    assert statuses(StatsGate(no_tools).evaluate("h", holdout(**change), SPEC)) == {"G7": "fail"}


def test_only_g7_cannot_be_overridden():
    assert [g for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7") if not overridable(g)] == ["G7"]


# -- verdicts ---------------------------------------------------------------------------------

PASS4 = dict.fromkeys(("G1", "G2", "G3", "G4"), "pass")
PASS5 = {**PASS4, "G5": "pass"}
PASS6 = {**PASS5, "G6": "pass"}


@pytest.mark.parametrize(
    ("gates", "label"),
    [
        ({**PASS4, "G1": "fail"}, INCONCLUSIVE),
        ({**PASS5, "G1": "fail"}, INCONCLUSIVE),
        ({**PASS4, "G2": "fail"}, LUCK),
        ({**PASS4, "G3": "fail"}, LUCK),
        ({**PASS4, "G4": "fail"}, LUCK),
        ({**PASS4, "G5": "fail"}, LUCK),
        (PASS4, NOT_REPLICATED),
        (PASS5, PRICED_IN),
        ({**PASS5, "G6": "fail"}, PRICED_IN),
        (PASS6, PRICED_IN),
        ({**PASS6, "G7": "fail"}, PRICED_IN),
        ({**PASS6, "G7": "pass"}, REAL),
    ],
)
def test_compute_verdict_follows_the_table(gates, label):
    assert compute_verdict(gates) == label


def test_chance_is_resolved_by_the_competing_hypotheses():
    premise = ("premise", PASS4)
    assert resolve_chance([premise, ("explanation", {**PASS4, "G3": "fail"})]) == CHANCE_SUPPORTED
    assert (
        resolve_chance([premise, ("explanation", PASS5), ("confounder", PASS6)]) == CHANCE_REFUTED
    )
    assert resolve_chance([premise, ("explanation", PASS4)]) == CHANCE_UNDETERMINED  # no G5


# -- planted noise and planted signal (plan.md section 13.2) ---------------------------------


def _one_sided_p(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def run_chain(rng, true_effect, n=5000):
    """One a_gt_b ats_cover hypothesis with `n` rows a group, through the gates in the router's
    order: the primary test, then out_of_sample, then the market. Each draw is a fresh sample."""
    spec = {**SPEC, "metric": "ats_cover", "direction": "a_gt_b"}
    se = 0.5 * math.sqrt(2 / n)

    def draw():
        return true_effect + se * rng.standard_normal()

    def ci(effect):
        return effect - Z_95 * se, effect + Z_95 * se

    def run_tool(tool, args):  # the gate's spread-controlled regress
        effect = draw()
        low, high = ci(effect)
        return ToolResult(
            tool=tool,
            args=args,
            args_hash="s",
            effect=effect,
            ci_low=low,
            ci_high=high,
            p_value=0.5,
        )

    gate = StatsGate(run_tool)
    effect = draw()
    low, high = ci(effect)
    first = ToolResult(
        tool="compare_groups",
        args={"spec": spec},
        args_hash="p",
        n=2 * n,
        effect=effect,
        ci_low=low,
        ci_high=high,
        p_value=_one_sided_p(effect / se),
        extra={"n_a": n, "n_b": n, "mde": min_detectable_effect(se)},
    )
    gates = statuses(gate.evaluate("p", first, spec))
    if compute_verdict(gates) != NOT_REPLICATED:
        return compute_verdict(gates)
    replica = draw()
    replication = {"same_sign": replica * effect > 0, "retention": replica / effect}
    gates |= statuses(gate.evaluate("o", follow_up("out_of_sample", extra=replication), spec))
    if gates["G5"] == "pass":
        rate = 0.5 + true_effect / 2 + 0.5 / math.sqrt(n) * rng.standard_normal()
        bet = follow_up(
            "market_test", args={"side": "team"}, ci_low=rate - 0.014, ci_high=rate + 0.014
        )
        gates |= statuses(gate.evaluate("m", bet, spec))
    return compute_verdict(gates)


def test_planted_noise_rarely_reaches_a_real_verdict():
    rng = np.random.default_rng(config.SEED)
    labels = [run_chain(rng, 0.0) for _ in range(2000)]
    real = sum(label in (PRICED_IN, REAL) for label in labels) / len(labels)
    assert real <= 0.05
    assert labels.count(REAL) == 0


def test_a_planted_three_percentage_point_effect_is_detected():
    rng = np.random.default_rng(config.SEED)
    labels = [run_chain(rng, 0.03) for _ in range(500)]
    assert sum(label in (PRICED_IN, REAL) for label in labels) / len(labels) >= 0.7


# -- on Person A's real tools -----------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("built")
    games = synthetic_games()
    save_tables(games, build_team_games(games), directory)
    return directory


@pytest.fixture
def real_tools(dataset, monkeypatch):
    previous = data.data_dir()
    data.use_data_dir(dataset)
    monkeypatch.setenv("WHACK_TOOLS", "real")
    monkeypatch.setattr(config, "N_PERMUTATIONS", 2000)
    monkeypatch.setattr(config, "N_BOOTSTRAP", 1000)
    reset_tool_state()
    yield
    reset_tool_state()
    data.use_data_dir(previous)


def test_the_gate_reads_the_real_tools_output(real_tools):
    """The synthetic data plants a January home edge in the explore seasons only."""
    spec = planted_spec(kind="explanation")
    gate = StatsGate(dispatch)

    first = dispatch("compare_groups", {"spec": spec})
    gates = statuses(gate.evaluate("p", first, spec))
    assert gates["G2"] == gates["G3"] == gates["G4"] == "pass"
    assert gates["G1"] == "fail"  # 456 home January games: MDE about 1.8 points, above 1.0

    replica = dispatch("out_of_sample", {"test_id": first.extra["test_id"]})
    assert statuses(gate.evaluate("o", replica, spec)) == {"G5": "fail"}

    bet = dispatch("market_test", {"slice": spec["group_a"], "side": "team"})
    market_check = gate.evaluate("m", bet, spec)
    assert statuses(market_check) == {"G6": "pass"}, market_check[0].rationale

    sealed = dispatch("holdout_confirm", {"hypothesis_id": spec["id"]})
    (holdout_check,) = gate.evaluate("h", sealed, spec)
    assert holdout_check.status == "fail"
    assert "unavailable" not in holdout_check.rationale  # it read market and spread_controlled
