"""Load game results into one normalised shape, whichever source they came from.

Sources: the Kaggle NBA games dataset (Nathan Lauga) or a saved `nba_api` league game log. Both
are parsed from their expected column names and need checking against real downloads.

Output columns: game_id, date, season, home, away, home_pts, away_pts, is_playoffs, is_bubble.
"""

from pathlib import Path

import pandas as pd

from backend.data.seasons import FIRST_SEASON_START, is_bubble_day, season_label
from backend.data.teams import franchise_from_team_id

_REGULAR_SEASON = "2"
_PLAYOFFS = "4"
_PLAY_IN = "5"
_KEPT_GAME_TYPES = {_REGULAR_SEASON, _PLAYOFFS, _PLAY_IN}

RESULT_COLUMNS = [
    "game_id",
    "date",
    "season",
    "home",
    "away",
    "home_pts",
    "away_pts",
    "is_playoffs",
    "is_bubble",
]


def _season_start_from_game_id(game_id: str) -> int:
    """Read the season from an NBA game id, whose format is 00TYYNNNNN."""
    two_digit_year = int(game_id[3:5])
    return (1900 if two_digit_year >= 46 else 2000) + two_digit_year


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep real games in the study period and add the derived columns."""
    frame = frame.dropna(subset=["home_pts", "away_pts", "home", "away"]).copy()
    frame = frame[frame["game_id"].str[2].isin(_KEPT_GAME_TYPES)]
    starts = frame["game_id"].map(_season_start_from_game_id)
    frame = frame[starts >= FIRST_SEASON_START].copy()
    frame["season"] = starts[frame.index].map(season_label)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    # Play-in games come after the regular season, so they count as postseason.
    frame["is_playoffs"] = frame["game_id"].str[2].isin({_PLAYOFFS, _PLAY_IN})
    frame["is_bubble"] = frame["date"].map(lambda day: is_bubble_day(day.date()))
    frame["home_pts"] = frame["home_pts"].astype(int)
    frame["away_pts"] = frame["away_pts"].astype(int)
    frame = frame.drop_duplicates(subset="game_id")
    return frame[RESULT_COLUMNS].sort_values(["date", "game_id"]).reset_index(drop=True)


def load_lauga_games(path: Path) -> pd.DataFrame:
    """Load `games.csv` from the Kaggle dataset."""
    raw = pd.read_csv(path)
    frame = pd.DataFrame(
        {
            "game_id": raw["GAME_ID"].astype("int64").astype(str).str.zfill(10),
            "date": raw["GAME_DATE_EST"],
            "home": raw["HOME_TEAM_ID"].map(franchise_from_team_id),
            "away": raw["VISITOR_TEAM_ID"].map(franchise_from_team_id),
            "home_pts": raw["PTS_home"],
            "away_pts": raw["PTS_away"],
        }
    )
    return _finalise(frame)


def load_nba_api_gamelog(path: Path) -> pd.DataFrame:
    """Load a saved `LeagueGameLog` CSV, which has one row per team per game."""
    raw = pd.read_csv(path, dtype={"GAME_ID": str})
    raw["GAME_ID"] = raw["GAME_ID"].str.zfill(10)
    raw["team"] = raw["TEAM_ID"].map(franchise_from_team_id)
    is_home_row = raw["MATCHUP"].str.contains("vs.", regex=False)
    home = raw[is_home_row].set_index("GAME_ID")
    away = raw[~is_home_row].set_index("GAME_ID")
    joined = home.merge(away, left_index=True, right_index=True, suffixes=("_home", "_away"))
    frame = pd.DataFrame(
        {
            "game_id": joined.index,
            "date": joined["GAME_DATE_home"].to_numpy(),
            "home": joined["team_home"].to_numpy(),
            "away": joined["team_away"].to_numpy(),
            "home_pts": joined["PTS_home"].to_numpy(),
            "away_pts": joined["PTS_away"].to_numpy(),
        }
    )
    return _finalise(frame)


def pull_nba_api_gamelog(start_years: range, out_path: Path) -> None:
    """Download game logs with `nba_api` and save them for `load_nba_api_gamelog`."""
    from nba_api.stats.endpoints import leaguegamelog  # optional, only needed to pull

    pages = []
    for start_year in start_years:
        for season_type in ("Regular Season", "Playoffs", "PlayIn"):
            log = leaguegamelog.LeagueGameLog(
                season=season_label(start_year),
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
            )
            pages.append(log.get_data_frames()[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pages, ignore_index=True).to_csv(out_path, index=False)
