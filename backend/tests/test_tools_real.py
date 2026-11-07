"""The real tools, on a small synthetic dataset built through the actual pipeline.

Planted structure:
- Explore seasons: home teams score 6 more points in January, and the line doesn't know.
- Validate and holdout seasons: no January effect, so it must fail to replicate.
- Every season: each team has a fixed strength, so raw margins persist between seasons.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.data.join import GAMES_COLUMNS
from backend.data.store import save_tables
from backend.data.tables import build_team_games
from backend.data.teams import FRANCHISE_BY_TEAM_ID
from backend.gate import config
from backend.tools import data, stats
from backend.tools.registry import active_backend, dispatch, reset_tool_state

TEAMS = sorted(FRANCHISE_BY_TEAM_ID.values())
SEASONS = {2010: True, 2011: True, 2012: True, 2016: False, 2021: False}  # start -> planted
JANUARY_BOOST = 6.0
HOME_JANUARY = {
    "split": "explore",
    "filters": [{"col": "is_home", "op": "eq", "value": True}, {"col": "month", "op": "eq", "value": 1}],
}


def synthetic_games() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    strength = dict(zip(TEAMS, rng.normal(0, 4, len(TEAMS))))
    rows, number = [], 0
    for start, planted in SEASONS.items():
        day = pd.Timestamp(start, 11, 1)
        for _ in range(80):  # 80 days x 8 games, no team twice in a day
            teams = rng.permutation(TEAMS)[:16]
            for home, away in zip(teams[:8], teams[8:]):
                number += 1
                expected = 3 + strength[home] - strength[away]
                boost = JANUARY_BOOST if planted and day.month == 1 else 0
                margin = round(rng.normal(expected + boost, 12))
                margin = margin if margin != 0 else 1
                away_pts = int(rng.integers(95, 115))
                spread = -round(expected * 2) / 2  # the market knows strength and home edge only
                rows.append(
                    [
                        f"{start}{number:06d}",
                        day,
                        f"{start}-{(start + 1) % 100:02d}",
                        home,
                        away,
                        away_pts + margin,
                        away_pts,
                        spread,
                        2 * away_pts + margin + rng.choice([-6.5, -2, 0, 3.5]),
                        -150.0,
                        130.0,
                        False,
                        False,
                        False,
                    ]
                )
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=GAMES_COLUMNS)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("built")
    games = synthetic_games()
    save_tables(games, build_team_games(games), directory)
    return directory


@pytest.fixture(autouse=True)
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


def spec(hypothesis_id="h1", group_a=None, **changes) -> dict:
    base = {
        "id": hypothesis_id,
        "statement": "Home teams beat the spread in January.",
        "kind": "premise",
        "metric": "ats_margin",
        "group_a": group_a or HOME_JANUARY,
        "group_b": "complement",
        "direction": "a_gt_b",
        "primary_test": "compare_groups",
    }
    return {**base, **changes}


def team_games() -> pd.DataFrame:
    return data.public_frame()


def test_results_are_real_and_tied_to_the_data_version():
    result = dispatch("define_slice", {"spec": HOME_JANUARY})
    assert result.status == "ok"
    assert result.extra["mock"] is False
    assert result.extra["data_version"] == data.data_version()
    assert result.slice_hash


def test_define_slice_counts_match_the_rows():
    frame = team_games()
    expected = frame[(frame["split"] == "explore") & frame["is_home"] & (frame["month"] == 1)]
    result = dispatch("define_slice", {"spec": HOME_JANUARY})
    assert result.extra["n_rows"] == len(expected)
    assert result.extra["n_games"] == expected["game_id"].nunique()


def test_compare_groups_finds_the_planted_effect():
    result = dispatch("compare_groups", {"spec": spec()})
    assert result.status == "ok"
    assert result.effect == pytest.approx(JANUARY_BOOST, abs=1.5)
    assert result.ci_low > 0
    assert result.p_value < 0.01
    assert result.extra["n_a"] + result.extra["n_b"] == result.n


def test_compare_groups_effect_is_the_raw_difference_in_means():
    frame = team_games()
    explore = frame[frame["split"] == "explore"]
    in_a = explore["is_home"] & (explore["month"] == 1)
    expected = explore.loc[in_a, "ats_margin"].mean() - explore.loc[~in_a, "ats_margin"].mean()
    result = dispatch("compare_groups", {"spec": spec()})
    assert result.effect == pytest.approx(expected, abs=1e-4)


def test_compare_groups_finds_nothing_where_nothing_was_planted():
    first_half = {"split": "explore", "filters": [{"col": "team", "op": "in", "value": TEAMS[:15]}]}
    result = dispatch("compare_groups", {"spec": spec(group_a=first_half, direction="two_sided")})
    assert result.p_value > 0.05
    assert result.ci_low < 0 < result.ci_high


def test_binary_metric_drops_pushes_from_the_rate():
    frame = team_games()
    explore = frame[frame["split"] == "explore"]
    in_a = explore["is_home"] & (explore["month"] == 1)
    cover = explore["ats_cover"].astype("Float64")
    result = dispatch("compare_groups", {"spec": spec(metric="ats_cover")})
    assert result.extra["mean_a"] == pytest.approx(float(cover[in_a].mean()), abs=1e-4)
    assert result.extra["n_a"] == int(cover[in_a].notna().sum())


def test_same_arguments_give_identical_results():
    first = dispatch("compare_groups", {"spec": spec()})
    reset_tool_state()
    second = dispatch("compare_groups", {"spec": json.loads(json.dumps(spec()))})
    assert first == second


def test_overlapping_groups_are_refused():
    result = dispatch("compare_groups", {"spec": spec(group_b=HOME_JANUARY)})
    assert result.reason == "groups_overlap"


def test_empty_group_is_reported_not_raised():
    nobody = {"split": "explore", "filters": [{"col": "month", "op": "eq", "value": 7}]}
    result = dispatch("compare_groups", {"spec": spec(group_a=nobody)})
    assert result.reason == "empty_group"


def test_filter_value_of_the_wrong_type_is_refused():
    bad = {"split": "explore", "filters": [{"col": "month", "op": "eq", "value": "january"}]}
    assert dispatch("define_slice", {"spec": bad}).reason == "invalid_filter"


def test_regress_without_controls_matches_the_difference_in_means():
    compared = dispatch("compare_groups", {"spec": spec()})
    regressed = dispatch("regress", {"spec": spec(primary_test="regress")})
    assert regressed.effect == pytest.approx(compared.effect, abs=1e-4)
    assert regressed.extra["se_type"] == "cluster-robust by game_id"


def test_regress_reports_shrinkage_when_controls_are_added():
    result = dispatch("regress", {"spec": spec(primary_test="regress", controls=["rest_days"])})
    assert result.status == "ok"
    assert result.extra["controls"] == ["rest_days"]
    assert "effect_without_controls" in result.extra


def test_regress_adds_spread_for_market_claims():
    result = dispatch("regress", {"spec": spec(primary_test="regress", kind="market")})
    assert result.extra["spread_added"] is True
    assert "spread" in result.extra["controls"]


def test_regress_on_a_binary_metric_reports_a_rate_effect():
    result = dispatch("regress", {"spec": spec(primary_test="regress", metric="ats_cover")})
    assert result.extra["model"].startswith("logit")
    assert 0 < result.effect < 1


def test_regress_refuses_a_control_that_duplicates_the_group():
    home = {"split": "explore", "filters": [{"col": "is_home", "op": "eq", "value": True}]}
    result = dispatch("regress", {"spec": spec(group_a=home, primary_test="regress", controls=["is_home"])})
    assert result.reason == "collinear_controls"


def test_out_of_sample_shows_the_effect_does_not_replicate():
    test_id = dispatch("compare_groups", {"spec": spec()}).extra["test_id"]
    result = dispatch("out_of_sample", {"test_id": test_id})
    assert result.status == "ok"
    assert set(result.args) == {"test_id"}
    assert abs(result.effect) < 2.5
    assert result.extra["retention"] < 0.5


def test_placebo_runs_the_same_test_on_another_slice():
    test_id = dispatch("compare_groups", {"spec": spec()}).extra["test_id"]
    away = {"split": "explore", "filters": [{"col": "is_home", "op": "eq", "value": False}, {"col": "month", "op": "eq", "value": 1}]}
    result = dispatch("placebo", {"test_id": test_id, "placebo_slice": away})
    assert result.status == "ok"
    assert result.effect < 0  # January away teams are on the wrong end of the boost


def test_market_test_counts_wins_losses_and_pushes_by_hand():
    frame = team_games()
    rows = frame[(frame["split"] == "explore") & frame["is_home"] & (frame["month"] == 1)]
    wins = int((rows["ats_cover"] == 1).sum())
    losses = int((rows["ats_cover"] == 0).sum())
    pushes = int((rows["ats_margin"] == 0).sum())
    result = dispatch("market_test", {"slice": HOME_JANUARY, "side": "team"})
    assert (result.extra["wins"], result.extra["losses"], result.extra["pushes"]) == (wins, losses, pushes)
    assert result.extra["rate"] == pytest.approx(wins / (wins + losses), abs=1e-5)
    expected_roi = (wins * 100 / 110 - losses) / (wins + losses + pushes)
    assert result.extra["roi"] == pytest.approx(expected_roi, abs=1e-5)
    assert result.effect == pytest.approx(result.extra["rate"] - config.BREAKEVEN, abs=1e-5)
    low, high = result.extra["wilson_ci"]
    assert low < result.extra["rate"] < high < 1


def test_market_test_opp_side_mirrors_the_team_side():
    team = dispatch("market_test", {"slice": HOME_JANUARY, "side": "team"})
    opp = dispatch("market_test", {"slice": HOME_JANUARY, "side": "opp"})
    assert (opp.extra["wins"], opp.extra["losses"]) == (team.extra["losses"], team.extra["wins"])


def test_market_test_counts_each_game_once_for_totals():
    both_sides = {"split": "explore", "filters": [{"col": "month", "op": "eq", "value": 1}]}
    result = dispatch("market_test", {"slice": both_sides, "side": "over"})
    frame = team_games()
    games = frame[(frame["split"] == "explore") & (frame["month"] == 1)]["game_id"].nunique()
    assert result.extra["wins"] + result.extra["losses"] + result.extra["pushes"] == games


def test_persistence_separates_skill_from_noise():
    margin = dispatch("persistence", {"metric": "margin", "entity": "team"})
    cover = dispatch("persistence", {"metric": "ats_cover", "entity": "team"})
    assert margin.effect > 0.5 and margin.p_value < 0.01
    assert cover.p_value > 0.05
    assert margin.extra["n_pairs"] == 60  # 30 teams x 2 consecutive explore pairs


def test_persistence_needs_two_consecutive_seasons():
    assert dispatch("persistence", {"metric": "margin", "split": "validate"}).reason == "not_enough_seasons"


def test_power_uses_the_real_spread_of_the_metric():
    frame = team_games()
    sd = float(np.nanstd(frame[frame["split"] == "explore"]["ats_margin"], ddof=1))
    result = dispatch("power", {"n_a": 400, "n_b": 1600, "metric": "ats_margin"})
    expected = (stats.Z_95 + stats.Z_POWER_80) * sd * (1 / 400 + 1 / 1600) ** 0.5
    assert result.extra["mde"] == pytest.approx(expected, abs=1e-4)


def test_add_feature_reports_coverage_and_only_precomputed_windows():
    assert dispatch("add_feature", {"name": "rolling_ats", "params": {"k": 10}}).extra["coverage"] > 0.5
    assert dispatch("add_feature", {"name": "rolling_ats", "params": {"k": 5}}).reason == "unsupported_window"


def test_describe_dataset_lists_public_seasons_only():
    result = dispatch("describe_dataset")
    assert result.extra["splits"]["explore"]["seasons"] == ["2010-11", "2011-12", "2012-13"]
    assert "2021-22" not in json.dumps(result.model_dump())


def test_holdout_confirm_runs_the_frozen_spec_on_sealed_rows_once():
    dispatch("compare_groups", {"spec": spec()})
    # A secondary test on the same hypothesis keeps its declared primary test, as B's agent does.
    dispatch("regress", {"spec": spec(controls=["rest_days"])})
    result = dispatch("holdout_confirm", {"hypothesis_id": "h1"})
    assert result.status == "ok"
    assert result.extra["seasons"] == ["2021-22"]
    assert result.extra["primary_test"] == "compare_groups"  # the regress call didn't replace it
    assert result.extra["retention"] < 0.5  # nothing was planted in the holdout
    assert set(result.extra["market"]) >= {"rate", "ci_low"}
    again = dispatch("holdout_confirm", {"hypothesis_id": "h1"})
    assert (again.status, again.reason) == ("error", "already_used")


def test_holdout_confirm_needs_a_tested_hypothesis():
    assert dispatch("holdout_confirm", {"hypothesis_id": "h1"}).reason == "unknown_hypothesis"


def test_missing_dataset_is_an_error_not_an_exception(tmp_path):
    data.use_data_dir(tmp_path)
    assert dispatch("define_slice", {"spec": HOME_JANUARY}).reason == "data_not_built"


def test_backend_choice_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("WHACK_TOOLS", "mock")
    assert dispatch("define_slice", {"spec": HOME_JANUARY}).extra["mock"] is True
    monkeypatch.setenv("WHACK_TOOLS", "auto")
    assert active_backend() == "real"
    data.use_data_dir(tmp_path)
    assert active_backend() == "mock"


def test_within_season_shuffles_never_mix_seasons():
    # Group A only in season 0, group B only in season 1, with very different means. Shuffling
    # within seasons can't move a label across seasons, so every shuffle equals the observed.
    rows = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(40)],
            "season": ["2010-11"] * 20 + ["2011-12"] * 20,
            "is_home": True,
        }
    )
    values = np.array([10.0] * 20 + [0.0] * 20)
    labels = np.array([1] * 20 + [2] * 20, dtype=np.int8)
    table = stats.build_table(rows, values, labels)
    null = stats.permutation_null(table, 200, np.random.default_rng(0))
    assert np.allclose(null, stats.observed(table))


def test_permutation_test_rejects_about_five_percent_of_pure_noise():
    rng = np.random.default_rng(11)
    rejections = 0
    trials = 300
    for _ in range(trials):
        rows = pd.DataFrame(
            {
                "game_id": np.repeat([f"g{i}" for i in range(100)], 2),
                "season": np.repeat(["2010-11"] * 50 + ["2011-12"] * 50, 2),
                "is_home": np.tile([True, False], 100),
            }
        )
        values = rng.normal(0, 1, 200)
        labels = rng.choice(np.array([1, 2], dtype=np.int8), 200)
        table = stats.build_table(rows, values, labels)
        null = stats.permutation_null(table, 400, rng)
        rejections += stats.permutation_p(null, stats.observed(table), "two_sided") < 0.05
    assert 0.02 < rejections / trials < 0.09
