"""Acceptance checks on the real dataset (plan.md v2 section 6.4).

These skip until `python -m backend.data.build` has run on real downloads.
"""

import json

import pandas as pd
import pytest

from backend.data.join import MIN_JOIN_RATE
from backend.data.seasons import HOLDOUT_FIRST_START, season_start
from backend.data.store import GAMES_FILE, PROCESSED_DIR, TEAM_GAMES_FILE

needs_real_data = pytest.mark.skipif(
    not (PROCESSED_DIR / GAMES_FILE).exists(), reason="dataset not built yet"
)

# Five public games checked independently; see the linked validation notes.
HAND_CHECKED_GAMES: list[tuple[str, str, float, int | None]] = [
    ("0041500401", "GSW", 9.0, 1),  # 104-89, GS -6
    ("0041500402", "GSW", 27.0, 1),  # 110-77, GS -6
    ("0041500404", "CLE", -13.5, 0),  # 97-108, CLE -2.5
    ("0041500406", "CLE", 12.0, 1),  # 115-101, CLE -2
    ("0041500407", "GSW", -8.5, 0),  # 89-93, archive GS -4.5
]
# Independent scores/lines and the Game 7 line movement are documented in
# docs/live-research-validation.md. The archive is the authoritative line per game.


@needs_real_data
def test_join_rate_meets_the_threshold_in_every_season():
    rates = pd.DataFrame(json.loads((PROCESSED_DIR / "manifest.json").read_text())["join_rates"])
    assert (rates["join_rate"] >= MIN_JOIN_RATE).all(), rates


@needs_real_data
def test_team_games_is_twice_games():
    games = pd.read_parquet(PROCESSED_DIR / GAMES_FILE)
    team_games = pd.read_parquet(PROCESSED_DIR / TEAM_GAMES_FILE)
    assert len(team_games) == 2 * len(games)


@needs_real_data
def test_public_tables_stop_before_the_holdout():
    frame = pd.read_parquet(PROCESSED_DIR / TEAM_GAMES_FILE)
    assert frame["season"].map(season_start).max() < HOLDOUT_FIRST_START


@needs_real_data
def test_no_team_plays_twice_on_one_day():
    frame = pd.read_parquet(PROCESSED_DIR / TEAM_GAMES_FILE)
    assert not frame.duplicated(["team", "date"]).any()
    assert frame["rest_days"].min() >= 0


@needs_real_data
def test_spreads_and_totals_are_in_believable_ranges():
    games = pd.read_parquet(PROCESSED_DIR / GAMES_FILE)
    assert games["close_spread_home"].abs().max() <= 30
    assert games["close_total"].dropna().between(150, 300).all()


@needs_real_data
def test_home_teams_are_favoured_more_often_than_not():
    games = pd.read_parquet(PROCESSED_DIR / GAMES_FILE)
    assert games["close_spread_home"].mean() < 0


@needs_real_data
def test_ats_cover_rate_is_close_to_a_coin_flip():
    frame = pd.read_parquet(PROCESSED_DIR / TEAM_GAMES_FILE)
    assert frame["ats_cover"].astype("float64").mean() == pytest.approx(0.5, abs=0.001)


@needs_real_data
@pytest.mark.skipif(len(HAND_CHECKED_GAMES) < 5, reason="five real games not hand-checked yet")
def test_ats_signs_on_five_hand_checked_real_games():
    frame = pd.read_parquet(PROCESSED_DIR / TEAM_GAMES_FILE).set_index(["game_id", "team"])
    for game_id, team, ats_margin, ats_cover in HAND_CHECKED_GAMES:
        row = frame.loc[(game_id, team)]
        assert row["ats_margin"] == ats_margin
        assert (pd.isna(row["ats_cover"]) and ats_cover is None) or row["ats_cover"] == ats_cover
