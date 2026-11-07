import pandas as pd
import pytest

from backend.data.arenas import ARENAS, home_arena
from backend.data.join import GAMES_COLUMNS
from backend.data.seasons import season_label, split_of
from backend.data.tables import REST_CAP_DAYS, build_team_games
from backend.data.teams import FRANCHISE_BY_TEAM_ID, franchise_from_odds_name

FEATURES = [
    "rest_days",
    "b2b",
    "opp_rest_days",
    "opp_b2b",
    "travel_km",
    "tz_shift",
    "altitude_m",
    "games_into_season",
    "rolling_ats_10",
    "rolling_margin_10",
]


def game(
    game_id,
    day,
    home="BOS",
    away="MIA",
    home_pts=110,
    away_pts=100,
    spread=-5.0,
    total=210.0,
    season="2010-11",
    bubble=False,
):
    values = [
        game_id,
        pd.Timestamp(day),
        season,
        home,
        away,
        home_pts,
        away_pts,
        spread,
        total,
        -200.0,
        170.0,
        False,
        bubble,
        bubble,
    ]
    return dict(zip(GAMES_COLUMNS, values))


def team_games(*games) -> pd.DataFrame:
    return build_team_games(pd.DataFrame(list(games), columns=GAMES_COLUMNS))


def row(frame: pd.DataFrame, game_id: str, team: str) -> pd.Series:
    return frame[(frame["game_id"] == game_id) & (frame["team"] == team)].iloc[0]


def test_team_games_has_exactly_two_rows_per_game():
    frame = team_games(game("g1", "2010-11-01"), game("g2", "2010-11-03", "NYK", "CHI"))
    assert len(frame) == 4
    assert frame.groupby("game_id").size().eq(2).all()


def test_home_favourite_covers_and_away_row_mirrors_it():
    frame = team_games(game("g1", "2010-11-01", home_pts=110, away_pts=100, spread=-5.0))
    home, away = row(frame, "g1", "BOS"), row(frame, "g1", "MIA")
    assert (home["margin"], home["spread"], home["ats_margin"], home["ats_cover"]) == (10, -5, 5, 1)
    assert (away["margin"], away["spread"], away["ats_margin"], away["ats_cover"]) == (-10, 5, -5, 0)


def test_favourite_who_wins_by_less_than_the_spread_fails_to_cover():
    frame = team_games(game("g1", "2010-11-01", home_pts=103, away_pts=100, spread=-5.0))
    assert row(frame, "g1", "BOS")["ats_cover"] == 0
    assert row(frame, "g1", "BOS")["win"] == 1
    assert row(frame, "g1", "MIA")["ats_cover"] == 1
    assert row(frame, "g1", "MIA")["win"] == 0


def test_away_favourite_has_negative_spread_on_its_own_row():
    frame = team_games(game("g1", "2010-11-01", home_pts=100, away_pts=110, spread=4.0))
    assert row(frame, "g1", "MIA")["spread"] == -4.0
    assert row(frame, "g1", "MIA")["ats_margin"] == 6.0
    assert row(frame, "g1", "BOS")["spread"] == 4.0


def test_spread_push_is_null_for_both_teams_never_zero():
    frame = team_games(game("g1", "2010-11-01", home_pts=110, away_pts=100, spread=-10.0))
    assert frame["ats_cover"].isna().all()
    assert frame["ats_margin"].eq(0).all()


def test_total_push_is_null_and_totals_score_both_directions():
    frame = team_games(
        game("g1", "2010-11-01", home_pts=110, away_pts=100, total=210.0),
        game("g2", "2010-11-03", home_pts=115, away_pts=100, total=210.0),
        game("g3", "2010-11-05", home_pts=100, away_pts=100, total=210.0),
    )
    assert frame[frame["game_id"] == "g1"]["total_over"].isna().all()
    assert frame[frame["game_id"] == "g2"]["total_over"].eq(1).all()
    assert frame[frame["game_id"] == "g3"]["total_over"].eq(0).all()


def test_missing_line_gives_null_outcomes_not_losses():
    frame = team_games(game("g1", "2010-11-01", spread=None, total=None))
    assert frame["ats_cover"].isna().all()
    assert frame["total_over"].isna().all()
    assert frame["win"].notna().all()


def test_pick_em_spread_is_plain_zero_on_both_rows():
    frame = team_games(game("g1", "2010-11-01", spread=0.0))
    assert str(row(frame, "g1", "MIA")["spread"]) == "0.0"


def test_rest_days_count_days_off_and_flag_back_to_backs():
    frame = team_games(
        game("g1", "2010-11-01", "BOS", "MIA"),
        game("g2", "2010-11-02", "BOS", "CHI"),
        game("g3", "2010-11-05", "BOS", "NYK"),
    )
    assert row(frame, "g1", "BOS")["rest_days"] == REST_CAP_DAYS
    assert row(frame, "g2", "BOS")["rest_days"] == 0
    assert bool(row(frame, "g2", "BOS")["b2b"])
    assert row(frame, "g3", "BOS")["rest_days"] == 2
    assert not bool(row(frame, "g3", "BOS")["b2b"])


def test_opponent_rest_comes_from_the_opponent():
    frame = team_games(
        game("g1", "2010-11-01", "BOS", "MIA"),
        game("g2", "2010-11-02", "BOS", "CHI"),
    )
    assert row(frame, "g2", "CHI")["opp_rest_days"] == 0
    assert bool(row(frame, "g2", "CHI")["opp_b2b"])
    assert row(frame, "g2", "BOS")["opp_rest_days"] == REST_CAP_DAYS


def test_rest_resets_at_a_new_season():
    # The gap is under the cap on purpose, so only the season reset can give the capped value.
    frame = team_games(
        game("g1", "2010-04-10", season="2009-10"),
        game("g2", "2010-04-13", season="2010-11"),
    )
    assert row(frame, "g2", "BOS")["rest_days"] == REST_CAP_DAYS
    assert row(frame, "g2", "BOS")["games_into_season"] == 0


def test_long_layoff_is_capped():
    frame = team_games(
        game("g1", "2020-03-10", season="2019-20"),
        game("g2", "2020-07-30", season="2019-20", bubble=True),
    )
    assert row(frame, "g2", "BOS")["rest_days"] == REST_CAP_DAYS


def test_travel_runs_from_previous_venue_with_time_zones():
    frame = team_games(
        game("g1", "2010-11-01", "BOS", "MIA"),
        game("g2", "2010-11-03", "LAL", "BOS"),
    )
    trip = row(frame, "g2", "BOS")
    assert trip["travel_km"] == pytest.approx(4170, abs=60)
    assert trip["tz_shift"] == -3
    assert trip["altitude_m"] == home_arena("LAL", 2010).altitude_m


def test_first_game_of_season_measures_travel_from_home():
    frame = team_games(game("g1", "2010-11-01", "BOS", "MIA"))
    assert row(frame, "g1", "BOS")["travel_km"] == pytest.approx(0)
    assert row(frame, "g1", "MIA")["travel_km"] > 1500


def test_home_stand_has_no_travel():
    frame = team_games(
        game("g1", "2010-11-01", "BOS", "MIA"),
        game("g2", "2010-11-03", "BOS", "CHI"),
    )
    assert row(frame, "g2", "BOS")["travel_km"] == pytest.approx(0)
    assert row(frame, "g2", "BOS")["tz_shift"] == 0


def test_bubble_games_are_played_in_orlando_with_no_travel_between_them():
    frame = team_games(
        game("g1", "2020-08-01", "LAL", "BOS", season="2019-20", bubble=True),
        game("g2", "2020-08-03", "DEN", "BOS", season="2019-20", bubble=True),
    )
    assert row(frame, "g2", "BOS")["travel_km"] == pytest.approx(0)
    assert row(frame, "g2", "BOS")["altitude_m"] == 30


def test_rolling_form_excludes_the_current_game():
    games = [
        game(f"g{index}", f"2010-11-{index + 1:02d}", home_pts=120, away_pts=100)
        for index in range(1, 8)
    ]
    games[5] = game("g6", "2010-11-07", home_pts=150, away_pts=100)
    frame = team_games(*games)
    assert row(frame, "g6", "BOS")["rolling_margin_10"] == pytest.approx(20)
    assert row(frame, "g7", "BOS")["rolling_margin_10"] == pytest.approx(25)


def test_rolling_form_needs_a_minimum_history():
    games = [game(f"g{day}", f"2010-11-{day:02d}") for day in range(1, 7)]
    frame = team_games(*games)
    assert pd.isna(row(frame, "g5", "BOS")["rolling_margin_10"])
    assert row(frame, "g6", "BOS")["rolling_margin_10"] == pytest.approx(10)


def test_rolling_cover_skips_pushes_instead_of_counting_them_as_zero():
    covers = [
        game(f"g{day}", f"2010-11-{day:02d}", home_pts=110, away_pts=100, spread=-5.0)
        for day in range(1, 6)
    ]
    push = game("push", "2010-11-06", home_pts=110, away_pts=100, spread=-10.0)
    last = game("last", "2010-11-07", home_pts=110, away_pts=100, spread=-5.0)
    frame = team_games(*covers, push, last)
    assert row(frame, "last", "BOS")["rolling_ats_10"] == pytest.approx(1.0)


def test_rolling_form_does_not_carry_across_seasons():
    old = [
        game(f"old{day}", f"2010-03-{day:02d}", season="2009-10", home_pts=140)
        for day in range(1, 10)
    ]
    new = [game("new1", "2010-10-27", season="2010-11")]
    frame = team_games(*old, *new)
    assert pd.isna(row(frame, "new1", "BOS")["rolling_margin_10"])


def test_no_feature_changes_when_the_row_or_later_games_change():
    schedule = [
        game(f"g{day:02d}", f"2010-11-{day:02d}", home_pts=100 + day, away_pts=100, spread=-1.0)
        for day in range(1, 13)
    ]
    changed = [dict(item) for item in schedule]
    for item in changed[7:]:  # the outcome of g08 and everything after it
        item["home_pts"] += 40
        item["close_spread_home"] = -30.0
        item["close_total"] = 150.0

    def features_through_g08(frame: pd.DataFrame) -> pd.DataFrame:
        kept = frame[frame["game_id"] <= "g08"].sort_values(["game_id", "team"])
        return kept[FEATURES].reset_index(drop=True)

    original, altered = team_games(*schedule), team_games(*changed)
    assert len(features_through_g08(original)) == 16
    assert not original["margin"].equals(altered["margin"])
    pd.testing.assert_frame_equal(features_through_g08(original), features_through_g08(altered))


def test_every_franchise_has_an_arena_in_every_season():
    for team in FRANCHISE_BY_TEAM_ID.values():
        for start in range(2007, 2027):
            assert home_arena(team, start).franchise == team


def test_arena_eras_do_not_overlap():
    for start in range(2007, 2027):
        teams = [
            arena.franchise for arena in ARENAS if arena.first_start <= start <= arena.last_start
        ]
        assert len(teams) == len(set(teams)) == 30


def test_relocations_use_the_arena_of_the_time():
    assert "Seattle" in home_arena("OKC", 2007).name
    assert home_arena("OKC", 2008).name == "Paycom Center"
    assert "Tampa" in home_arena("TOR", 2020).name
    assert home_arena("TOR", 2021).name == "Scotiabank Arena"


@pytest.mark.parametrize(
    ("name", "franchise"),
    [
        ("Seattle", "OKC"),
        ("OklahomaCity", "OKC"),
        ("NewJersey", "BKN"),
        ("Brooklyn", "BKN"),
        ("NewOrleans", "NOP"),
        ("Charlotte", "CHA"),
        ("LALakers", "LAL"),
        ("LA Clippers", "LAC"),
        ("GoldenState", "GSW"),
        ("Team LeBron", None),
    ],
)
def test_odds_names_map_to_one_franchise_through_renames(name, franchise):
    assert franchise_from_odds_name(name) == franchise


def test_season_splits_match_the_agreed_boundaries():
    expected = {
        2007: "explore",
        2015: "explore",
        2016: "validate",
        2018: "validate",
        2019: "special",
        2020: "special",
        2021: "holdout",
        2025: "holdout",
    }
    for start, split in expected.items():
        assert split_of(season_label(start)) == split


def test_season_label_wraps_the_century_digits():
    assert season_label(2009) == "2009-10"
    assert season_label(2099) == "2099-00"
