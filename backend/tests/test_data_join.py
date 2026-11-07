import pandas as pd
import pytest

from backend.data.join import join_odds
from backend.data.odds import ODDS_COLUMNS
from backend.data.results import RESULT_COLUMNS


def result(game_id, day, home, away, home_pts=110, away_pts=100, season="2010-11", bubble=False):
    return dict(
        zip(
            RESULT_COLUMNS,
            [game_id, pd.Timestamp(day), season, home, away, home_pts, away_pts, False, bubble],
        )
    )


def odds(
    day,
    home,
    away,
    home_final=110,
    away_final=100,
    spread=-5.0,
    total=210.0,
    neutral=False,
    season="2010-11",
):
    values = [
        pd.Timestamp(day),
        season,
        home,
        away,
        neutral,
        home_final,
        away_final,
        spread,
        total,
        -200.0,
        170.0,
    ]
    return dict(zip(ODDS_COLUMNS, values))


def run(results, odds_rows):
    return join_odds(
        pd.DataFrame(results, columns=RESULT_COLUMNS), pd.DataFrame(odds_rows, columns=ODDS_COLUMNS)
    )


def test_exact_match_attaches_the_line():
    joined = run([result("g1", "2010-11-01", "BOS", "MIA")], [odds("2010-11-01", "BOS", "MIA")])
    assert joined.games.loc[0, "close_spread_home"] == -5.0
    assert joined.join_rate == 1.0
    assert joined.unmatched_games.empty and joined.unused_odds.empty


def test_game_without_odds_stays_in_games_and_is_listed():
    joined = run(
        [result("g1", "2010-11-01", "BOS", "MIA"), result("g2", "2010-11-02", "CHI", "NYK")],
        [odds("2010-11-01", "BOS", "MIA")],
    )
    assert len(joined.games) == 2
    assert joined.join_rate == 0.5
    assert list(joined.unmatched_games["game_id"]) == ["g2"]


def test_neutral_game_listed_backwards_flips_line_and_moneylines():
    joined = run(
        [result("g1", "2020-08-05", "MIA", "BOS", 100, 95, "2019-20", bubble=True)],
        [odds("2020-08-05", "BOS", "MIA", 95, 100, spread=-4.0, neutral=True, season="2019-20")],
    )
    row = joined.games.iloc[0]
    assert row["close_spread_home"] == 4.0
    assert (row["ml_home"], row["ml_away"]) == (170.0, -200.0)
    assert joined.score_mismatches.empty


def test_swapped_teams_are_not_accepted_for_a_normal_game():
    joined = run(
        [result("g1", "2010-11-01", "MIA", "BOS")],
        [odds("2010-11-01", "BOS", "MIA")],
    )
    assert joined.join_rate == 0.0
    assert len(joined.unused_odds) == 1


def test_date_one_day_off_matches_when_scores_agree():
    joined = run([result("g1", "2010-11-02", "BOS", "MIA")], [odds("2010-11-01", "BOS", "MIA")])
    assert joined.join_rate == 1.0


def test_date_one_day_off_is_rejected_when_scores_disagree():
    joined = run(
        [result("g1", "2010-11-02", "BOS", "MIA")],
        [odds("2010-11-01", "BOS", "MIA", home_final=99)],
    )
    assert joined.join_rate == 0.0


def test_date_one_day_off_is_rejected_when_two_games_could_match():
    joined = run(
        [result("g1", "2010-11-01", "BOS", "MIA"), result("g2", "2010-11-03", "BOS", "MIA")],
        [odds("2010-11-02", "BOS", "MIA")],
    )
    assert joined.join_rate == 0.0


def test_exact_match_wins_over_a_nearby_game():
    joined = run(
        [result("g1", "2010-11-01", "BOS", "MIA"), result("g2", "2010-11-02", "BOS", "MIA")],
        [odds("2010-11-02", "BOS", "MIA", spread=-9.0), odds("2010-11-01", "BOS", "MIA")],
    )
    games = joined.games.set_index("game_id")
    assert games.loc["g2", "close_spread_home"] == -9.0
    assert games.loc["g1", "close_spread_home"] == -5.0


def test_score_disagreement_on_an_exact_match_is_reported_and_results_win():
    joined = run(
        [result("g1", "2010-11-01", "BOS", "MIA")],
        [odds("2010-11-01", "BOS", "MIA", home_final=111)],
    )
    assert len(joined.score_mismatches) == 1
    assert joined.games.loc[0, "home_pts"] == 110


def test_join_rate_only_counts_seasons_the_odds_cover():
    joined = run(
        [
            result("g1", "2010-11-01", "BOS", "MIA"),
            result("g2", "2023-11-01", "BOS", "MIA", season="2023-24"),
        ],
        [odds("2010-11-01", "BOS", "MIA")],
    )
    assert joined.join_rate == 1.0
    assert list(joined.rate_by_season["season"]) == ["2010-11"]
    assert len(joined.games) == 2


def test_bubble_games_are_neutral_and_normal_games_are_not():
    joined = run(
        [
            result("g1", "2020-08-05", "MIA", "BOS", season="2019-20", bubble=True),
            result("g2", "2020-03-01", "MIA", "BOS", season="2019-20"),
        ],
        [],
    )
    assert list(joined.games["is_neutral"]) == [True, False]


def test_rate_table_counts_games_with_usable_lines():
    joined = run(
        [result("g1", "2010-11-01", "BOS", "MIA"), result("g2", "2010-11-02", "CHI", "NYK")],
        [odds("2010-11-01", "BOS", "MIA"), odds("2010-11-02", "CHI", "NYK", spread=None)],
    )
    row = joined.rate_by_season.iloc[0]
    assert (row["matched"], row["with_spread"], row["with_total"]) == (2, 1, 2)
    assert row["join_rate"] == pytest.approx(1.0)
