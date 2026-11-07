from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backend.data.odds import (
    assign_dates,
    parse_line,
    parse_moneyline,
    parse_season_odds,
    resolve_close,
    season_from_filename,
)


def odds_rows(*rows: tuple) -> pd.DataFrame:
    columns = ["Date", "Rot", "VH", "Team", "Final", "Close", "ML"]
    frame = pd.DataFrame(rows, columns=columns).astype(object)
    frame.columns = [column.lower() for column in columns]
    return frame


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7.5", 7.5),
        (7.5, 7.5),
        ("-3", 3.0),
        ("7½", 7.5),
        ("pk", 0.0),
        ("PK", 0.0),
        ("NL", None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_line_handles_archive_formats(value, expected):
    assert parse_line(value) == expected


def test_parse_moneyline_keeps_sign_and_rejects_missing():
    assert parse_moneyline("-150") == -150.0
    assert parse_moneyline(130) == 130.0
    assert parse_moneyline("NL") is None
    assert parse_moneyline(0) is None


def test_close_home_favourite_gives_negative_spread():
    assert resolve_close(away_close=200.5, home_close=7.0) == (-7.0, 200.5)


def test_close_away_favourite_gives_positive_spread():
    assert resolve_close(away_close=4.5, home_close=210.0) == (4.5, 210.0)


def test_close_pick_em_gives_zero_spread_either_way_round():
    assert resolve_close(away_close=0.0, home_close=205.0) == (0.0, 205.0)
    assert resolve_close(away_close=205.0, home_close=0.0) == (0.0, 205.0)


@pytest.mark.parametrize("pair", [(200.0, 205.0), (3.0, 5.0), (None, None), (80.0, 80.0)])
def test_close_ambiguous_pair_gives_no_line(pair):
    assert resolve_close(*pair) == (None, None)


def test_close_keeps_the_readable_half_when_one_cell_is_missing():
    assert resolve_close(away_close=None, home_close=210.0) == (None, 210.0)
    assert resolve_close(away_close=None, home_close=6.0) == (-6.0, None)


def test_dates_roll_into_the_new_year():
    days = assign_dates([1028, 1231, 101, 415], 2010)
    assert days == [date(2010, 10, 28), date(2010, 12, 31), date(2011, 1, 1), date(2011, 4, 15)]


def test_dates_in_bubble_season_stay_in_the_second_year():
    days = assign_dates([1022, 311, 730, 1011], 2019)
    assert days == [date(2019, 10, 22), date(2020, 3, 11), date(2020, 7, 30), date(2020, 10, 11)]


def test_invalid_date_becomes_none_without_stopping_later_rows():
    assert assign_dates(["bad", 1028], 2010) == [None, date(2010, 10, 28)]


def test_parse_pairs_rows_into_one_game_with_relocated_team_mapped():
    raw = odds_rows(
        (1030, 501, "V", "Seattle", 95, 205, 180),
        (1030, 502, "H", "Denver", 110, 7.5, -220),
    )
    games = parse_season_odds(raw, "2007-08").games
    row = games.iloc[0]
    assert (row["away"], row["home"]) == ("OKC", "DEN")
    assert row["date"] == pd.Timestamp(2007, 10, 30)
    assert row["close_spread_home"] == -7.5
    assert row["close_total"] == 205
    assert (row["ml_home"], row["ml_away"]) == (-220, 180)
    assert (row["home_final"], row["away_final"]) == (110, 95)
    assert not row["neutral"]


def test_parse_marks_neutral_games():
    raw = odds_rows(
        (1022, 499, "V", "Boston", 100, 4, -180),
        (1022, 500, "H", "Miami", 95, 215, 160),
        (115, 499, "V", "Boston", 100, 4, -180),
        (115, 500, "H", "Miami", 95, 215, 160),
        (805, 501, "N", "Boston", 100, 4, -180),
        (805, 502, "N", "Miami", 95, 215, 160),
    )
    games = parse_season_odds(raw, "2019-20").games
    assert list(games["neutral"]) == [False, False, True]
    assert games.iloc[2]["date"] == pd.Timestamp(2020, 8, 5)


def test_parse_records_unknown_team_and_keeps_going():
    raw = odds_rows(
        (1030, 501, "V", "Team LeBron", 95, 205, 180),
        (1030, 502, "H", "Denver", 110, 7, -220),
        (1030, 503, "V", "Boston", 95, 205, 180),
        (1030, 504, "H", "Miami", 110, 7, -220),
    )
    parsed = parse_season_odds(raw, "2010-11")
    assert len(parsed.games) == 1
    assert [issue["kind"] for issue in parsed.issues] == ["unknown_team"]


def test_parse_recovers_alignment_after_an_orphan_row():
    raw = odds_rows(
        (1030, 499, "H", "Chicago", 95, 205, 180),
        (1031, 501, "V", "Boston", 95, 205, 180),
        (1031, 502, "H", "Miami", 110, 7, -220),
    )
    parsed = parse_season_odds(raw, "2010-11")
    assert len(parsed.games) == 1
    assert parsed.games.iloc[0]["home"] == "MIA"
    assert [issue["kind"] for issue in parsed.issues] == ["unpaired_row"]


def test_misdated_pair_still_pairs_by_rotation_without_shifting_later_games():
    # Real rows from the 2010-11 archive: CHA's row says Jan 29, LAC's says Jan 30.
    raw = odds_rows(
        (1230, 501, "V", "Boston", 100, 190, 150),
        (1230, 502, "H", "Miami", 110, 5, -170),
        (129, 515, "V", "Charlotte", 88, 192, 235),
        (130, 516, "H", "LAClippers", 103, 6, -275),
        (130, 801, "V", "Miami", 108, 202, -105),
        (130, 802, "H", "OklahomaCity", 103, 1, -115),
    )
    parsed = parse_season_odds(raw, "2010-11")
    pairs = list(zip(parsed.games["away"], parsed.games["home"]))
    assert pairs == [("BOS", "MIA"), ("CHA", "LAC"), ("MIA", "OKC")]
    assert parsed.games.iloc[1]["date"] == pd.Timestamp(2011, 1, 29)
    assert [issue["kind"] for issue in parsed.issues] == ["dates_disagree"]


def test_home_row_before_away_row_is_never_paired():
    raw = odds_rows(
        (130, 516, "H", "LAClippers", 103, 6, -275),
        (130, 801, "V", "Miami", 108, 202, -105),
    )
    parsed = parse_season_odds(raw, "2010-11")
    assert parsed.games.empty


def test_rows_with_non_consecutive_rotations_are_not_paired():
    raw = odds_rows(
        (130, 801, "V", "Miami", 108, 202, -105),
        (130, 804, "H", "OklahomaCity", 103, 1, -115),
    )
    assert parse_season_odds(raw, "2010-11").games.empty


def test_parse_records_missing_close_but_keeps_the_game():
    raw = odds_rows(
        (1030, 501, "V", "Boston", 95, "NL", 180),
        (1030, 502, "H", "Miami", 110, "NL", -220),
    )
    parsed = parse_season_odds(raw, "2010-11")
    assert len(parsed.games) == 1
    assert pd.isna(parsed.games.iloc[0]["close_spread_home"])
    assert [issue["kind"] for issue in parsed.issues] == ["unusable_close"]


def test_season_comes_from_file_name():
    assert season_from_filename(Path("nba odds 2007-08.xlsx")) == "2007-08"
    with pytest.raises(ValueError):
        season_from_filename(Path("odds.xlsx"))
