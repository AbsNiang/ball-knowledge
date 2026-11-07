"""Build `team_games`, two rows per game, and its as-of-tip-off features.

Sign rules, defined here and nowhere else:
- `close_spread_home` is negative when home is favoured.
- `spread` is from the team's perspective, so the away row is `-close_spread_home`.
- `ats_margin = margin + spread`.
- `ats_cover` and `total_over` are NULL on a push, and also when there is no line. They are never
  0 in those cases, so pushes drop out of rates instead of counting as losses.

Every feature uses only games before the row's own, within the same season. `month`,
`is_playoffs`, `is_bubble` and `is_neutral` are extra columns the slice filters need.
"""

import numpy as np
import pandas as pd

from backend.data.arenas import BUBBLE_ARENA, Arena, home_arena
from backend.data.seasons import season_start

REST_CAP_DAYS = 7
ROLLING_WINDOW = 10
ROLLING_MIN_GAMES = 5
EARTH_RADIUS_KM = 6371.0

TEAM_GAMES_COLUMNS = [
    "game_id",
    "date",
    "season",
    "month",
    "team",
    "opp",
    "is_home",
    "is_playoffs",
    "is_bubble",
    "is_neutral",
    "pts",
    "opp_pts",
    "margin",
    "spread",
    "ats_margin",
    "ats_cover",
    "win",
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
    "close_total",
    "total_over",
]


def build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Expand `games` into two rows per game and add every feature."""
    frame = pd.concat([_side(games, is_home=True), _side(games, is_home=False)], ignore_index=True)
    frame = frame.sort_values(["team", "date", "game_id"], ignore_index=True)
    frame = _add_rest(frame)
    frame = _add_opponent_rest(frame)
    frame = _add_venue_features(frame, games)
    frame = _add_form(frame)
    return frame.sort_values(["date", "game_id", "is_home"], ignore_index=True)[TEAM_GAMES_COLUMNS]


def _side(games: pd.DataFrame, is_home: bool) -> pd.DataFrame:
    """Build the rows for one side of every game."""
    us, them = ("home", "away") if is_home else ("away", "home")
    frame = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "date": games["date"],
            "season": games["season"],
            "month": games["date"].dt.month,
            "team": games[us],
            "opp": games[them],
            "is_home": is_home,
            "is_playoffs": games["is_playoffs"],
            "is_bubble": games["is_bubble"],
            "is_neutral": games["is_neutral"],
            "pts": games[f"{us}_pts"],
            "opp_pts": games[f"{them}_pts"],
            "close_total": pd.to_numeric(games["close_total"]).astype("float64"),
        }
    )
    frame["margin"] = frame["pts"] - frame["opp_pts"]
    spread_home = pd.to_numeric(games["close_spread_home"]).astype("float64")
    frame["spread"] = spread_home if is_home else -spread_home
    frame["spread"] = frame["spread"] + 0.0  # turns -0.0 into 0.0 for pick'ems
    frame["ats_margin"] = frame["margin"] + frame["spread"]
    frame["ats_cover"] = _binary_or_null(frame["ats_margin"])
    frame["win"] = (frame["margin"] > 0).astype(int)
    frame["total_over"] = _binary_or_null(frame["pts"] + frame["opp_pts"] - frame["close_total"])
    return frame


def _binary_or_null(difference: pd.Series) -> pd.Series:
    """Return 1 above zero, 0 below it, and NULL at zero or when there is no line."""
    result = pd.Series(pd.NA, index=difference.index, dtype="Int8")
    result[difference > 0] = 1
    result[difference < 0] = 0
    return result


def _add_rest(frame: pd.DataFrame) -> pd.DataFrame:
    """Add days off before each game.

    Back-to-back is 0 days. The first game of a season is set to the cap, and the cap applies
    everywhere so the months-long bubble layoff can't dominate a regression.
    """
    previous = frame.groupby(["team", "season"])["date"].shift(1)
    rest = (frame["date"] - previous).dt.days - 1
    frame["rest_days"] = rest.fillna(REST_CAP_DAYS).clip(upper=REST_CAP_DAYS).astype(int)
    frame["b2b"] = frame["rest_days"] == 0
    frame["games_into_season"] = frame.groupby(["team", "season"]).cumcount()
    return frame


def _add_opponent_rest(frame: pd.DataFrame) -> pd.DataFrame:
    opponent = frame[["game_id", "team", "rest_days", "b2b"]].rename(
        columns={"team": "opp", "rest_days": "opp_rest_days", "b2b": "opp_b2b"}
    )
    return frame.merge(opponent, on=["game_id", "opp"], how="left", validate="one_to_one")


def _venue(home: str, season: str, is_bubble: bool) -> Arena:
    return BUBBLE_ARENA if is_bubble else home_arena(home, season_start(season))


def _add_venue_features(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add altitude, plus travel and time zone change from the previous game's venue.

    The first game of a season is measured from the team's home arena.
    """
    venue_by_game = {
        row.game_id: _venue(row.home, row.season, row.is_bubble)
        for row in games[["game_id", "home", "season", "is_bubble"]].itertuples(index=False)
    }
    venues = frame["game_id"].map(venue_by_game)
    frame["_lat"] = venues.map(lambda arena: arena.lat)
    frame["_lon"] = venues.map(lambda arena: arena.lon)
    frame["_tz"] = venues.map(lambda arena: arena.utc_offset)
    frame["altitude_m"] = venues.map(lambda arena: arena.altitude_m)

    previous = frame.groupby(["team", "season"])[["_lat", "_lon", "_tz"]].shift(1)
    first_game = previous["_lat"].isna()
    bases = [
        home_arena(team, season_start(season))
        for team, season in zip(frame.loc[first_game, "team"], frame.loc[first_game, "season"])
    ]
    previous.loc[first_game, "_lat"] = [arena.lat for arena in bases]
    previous.loc[first_game, "_lon"] = [arena.lon for arena in bases]
    previous.loc[first_game, "_tz"] = [arena.utc_offset for arena in bases]

    frame["travel_km"] = _haversine_km(
        previous["_lat"], previous["_lon"], frame["_lat"], frame["_lon"]
    )
    frame["tz_shift"] = (frame["_tz"] - previous["_tz"]).astype(int)
    return frame.drop(columns=["_lat", "_lon", "_tz"])


def _haversine_km(lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
    lat_a, lon_a, lat_b, lon_b = (
        np.radians(pd.to_numeric(values).to_numpy(dtype=float))
        for values in (lat_a, lon_a, lat_b, lon_b)
    )
    half_chord = (
        np.sin((lat_b - lat_a) / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half_chord))


def _add_form(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rolling form over the previous games, excluding the row's own game."""
    frame["rolling_ats_10"] = _rolling_before(frame, frame["ats_cover"].astype("float64"))
    frame["rolling_margin_10"] = _rolling_before(frame, frame["margin"].astype("float64"))
    return frame


def _rolling_before(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    groups = [frame["team"], frame["season"]]
    return values.groupby(groups).transform(
        lambda series: series.shift(1)
        .rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_GAMES)
        .mean()
    )
