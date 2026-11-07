"""Join game results to closing odds and build the `games` table.

Results are the source of truth, so every game stays in `games` whether or not it found odds.
Later features need the full schedule: a game without a line still counts as a game played.

Matching runs in order of confidence, and each result can be claimed once:
1. Same date, home and away.
2. Neutral-site odds only: same date with home and away swapped, flipping the line to match.
3. One day either side, because the archive's dates can differ from the results source. This
   needs a matching final score and a single candidate, since teams can meet on consecutive days.
"""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

MIN_JOIN_RATE = 0.98

GAMES_COLUMNS = [
    "game_id",
    "date",
    "season",
    "home",
    "away",
    "home_pts",
    "away_pts",
    "close_spread_home",
    "close_total",
    "ml_home",
    "ml_away",
    "is_playoffs",
    "is_bubble",
    "is_neutral",
]


@dataclass
class JoinResult:
    games: pd.DataFrame
    unmatched_games: pd.DataFrame  # games in covered seasons with no odds row
    unused_odds: pd.DataFrame  # odds rows that matched no game
    score_mismatches: pd.DataFrame  # matched pairs whose final scores disagree
    rate_by_season: pd.DataFrame  # season, games, matched, join_rate, with_spread, with_total

    @property
    def join_rate(self) -> float:
        games = int(self.rate_by_season["games"].sum())
        return float(self.rate_by_season["matched"].sum() / games) if games else 0.0


def _flip(odds: dict) -> dict:
    """Return an odds row with home and away exchanged."""
    spread = odds["close_spread_home"]
    return {
        **odds,
        "home": odds["away"],
        "away": odds["home"],
        "home_final": odds["away_final"],
        "away_final": odds["home_final"],
        "ml_home": odds["ml_away"],
        "ml_away": odds["ml_home"],
        "close_spread_home": None if pd.isna(spread) else (0.0 if spread == 0 else -spread),
    }


def _scores_agree(result: dict, odds: dict) -> bool:
    return result["home_pts"] == odds["home_final"] and result["away_pts"] == odds["away_final"]


def join_odds(results: pd.DataFrame, odds: pd.DataFrame) -> JoinResult:
    """Attach odds to results and report everything that didn't line up."""
    results = results.reset_index(drop=True)
    result_rows = results.to_dict("records")
    odds_rows = odds.to_dict("records")

    index_by_key: dict[tuple, list[int]] = {}
    for position, row in enumerate(result_rows):
        index_by_key.setdefault((row["date"], row["home"], row["away"]), []).append(position)

    claimed: dict[int, dict] = {}  # result position -> odds row, oriented to the result

    def claim(key: tuple, row: dict) -> bool:
        for position in index_by_key.get(key, []):
            if position not in claimed:
                claimed[position] = row
                return True
        return False

    remaining = [
        row for row in odds_rows if not claim((row["date"], row["home"], row["away"]), row)
    ]

    still_remaining = []
    for row in remaining:
        flipped = _flip(row)
        key = (flipped["date"], flipped["home"], flipped["away"])
        if not (row["neutral"] and claim(key, flipped)):
            still_remaining.append(row)

    unused = []
    for row in still_remaining:
        candidates = _nearby_candidates(row, index_by_key, result_rows, claimed)
        if len(candidates) == 1:
            claimed[candidates[0][0]] = candidates[0][1]
        else:
            unused.append(row)

    games = _build_games(results, claimed)
    covered = set(odds["season"])
    matched_mask = games.index.isin(list(claimed))
    in_covered = games["season"].isin(covered)
    mismatches = [
        {
            **{key: result_rows[position][key] for key in ("game_id", "date", "home", "away")},
            "home_pts": result_rows[position]["home_pts"],
            "away_pts": result_rows[position]["away_pts"],
            "odds_home_final": row["home_final"],
            "odds_away_final": row["away_final"],
        }
        for position, row in claimed.items()
        if not _scores_agree(result_rows[position], row)
    ]
    return JoinResult(
        games=games,
        unmatched_games=games.loc[in_covered & ~matched_mask, GAMES_COLUMNS[:7]].reset_index(
            drop=True
        ),
        unused_odds=pd.DataFrame(unused, columns=odds.columns),
        score_mismatches=pd.DataFrame(mismatches),
        rate_by_season=_rates(games, matched_mask, covered),
    )


def _nearby_candidates(
    row: dict, index_by_key: dict, result_rows: list[dict], claimed: dict
) -> list[tuple[int, dict]]:
    """Find unclaimed results a day either side with the same teams and final score."""
    orientations = [row, _flip(row)] if row["neutral"] else [row]
    candidates = []
    for oriented in orientations:
        for offset in (-1, 1):
            day = oriented["date"] + timedelta(days=offset)
            for position in index_by_key.get((day, oriented["home"], oriented["away"]), []):
                if position not in claimed and _scores_agree(result_rows[position], oriented):
                    candidates.append((position, oriented))
    return candidates


def _build_games(results: pd.DataFrame, claimed: dict[int, dict]) -> pd.DataFrame:
    columns = ["close_spread_home", "close_total", "ml_home", "ml_away"]
    games = results.copy()
    for column in columns:
        values = {position: row[column] for position, row in claimed.items()}
        games[column] = pd.Series(values, index=games.index, dtype="float64")
    neutral = {position: bool(row["neutral"]) for position, row in claimed.items()}
    games["is_neutral"] = games["is_bubble"] | pd.Series(neutral, index=games.index).eq(True)
    return games[GAMES_COLUMNS]


def _rates(games: pd.DataFrame, matched_mask, covered: set[str]) -> pd.DataFrame:
    frame = games[["season", "close_spread_home", "close_total"]].copy()
    frame["matched"] = matched_mask
    frame = frame[frame["season"].isin(covered)]
    rates = (
        frame.groupby("season")
        .agg(
            games=("matched", "size"),
            matched=("matched", "sum"),
            with_spread=("close_spread_home", "count"),
            with_total=("close_total", "count"),
        )
        .reset_index()
    )
    rates["join_rate"] = rates["matched"] / rates["games"]
    return rates
