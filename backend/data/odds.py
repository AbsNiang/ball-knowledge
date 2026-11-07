"""Parse the Sportsbook Reviews Online odds archive into one row per game.

Each season file is expected to have two consecutive rows per game, away then home, with columns
Date, Rot, VH, Team, 1st-4th, Final, Open, Close, ML, 2H. The format comes from the archive's
known layout and hasn't been checked against a real download yet.

Quirks handled here:
- `Date` is MMDD with no year, so the year comes from file order. 2019-20 ran from October to
  October because of the bubble, so "month >= 7 means the first year" would be wrong.
- `Close` holds the spread on the favourite's row and the total on the other row, unsigned, so
  the two are told apart by size and the favourite is whichever row has the spread.
- Pick'ems appear as "pk", no line appears as "NL", and half points can appear as "7½".
- `VH` is "N" for neutral sites, where the away/home order isn't reliable. `join` handles that.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from backend.data.seasons import season_label, season_start
from backend.data.teams import franchise_from_odds_name

MAX_SPREAD = 40.0  # anything larger in Close can't be a spread
MIN_TOTAL = 120.0  # anything smaller in Close can't be a total
MAX_TOTAL = 320.0

_NUMBER = re.compile(r"^[+-]?\d*\.?\d+$")
_PICK_EM = {"pk", "pick", "pickem", "pick'em"}
_SEASON_IN_NAME = re.compile(r"(\d{4})[-_](\d{2,4})")
_NEW_YEAR_MONTH_DROP = 6  # a fall this large in month means the file crossed into January

ODDS_COLUMNS = [
    "date",
    "season",
    "home",
    "away",
    "neutral",
    "home_final",
    "away_final",
    "close_spread_home",
    "close_total",
    "ml_home",
    "ml_away",
]


@dataclass
class OddsParse:
    games: pd.DataFrame
    issues: list[dict] = field(default_factory=list)


def parse_line(value: object) -> float | None:
    """Return an unsigned spread or total, 0.0 for a pick'em, or None if there is no line."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower().replace("½", ".5")
    if text in _PICK_EM:
        return 0.0
    return abs(float(text)) if _NUMBER.match(text) else None


def parse_moneyline(value: object) -> float | None:
    """Return a signed American moneyline, or None if there isn't one."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not _NUMBER.match(text) or float(text) == 0:
        return None
    return float(text)


def resolve_close(
    away_close: float | None, home_close: float | None
) -> tuple[float | None, float | None]:
    """Turn the two Close cells into `(close_spread_home, close_total)`.

    The result is negative when home is favoured. A pair that can't be read as one spread and
    one total gives None for whichever value is unreadable, because a guessed line would silently
    corrupt every ATS number.
    """
    spread_home: float | None = None
    total: float | None = None
    if _is_total(away_close) and _is_total(home_close):
        return None, None
    if _is_spread(away_close) and _is_spread(home_close):
        return None, None
    for value, sign in ((away_close, 1.0), (home_close, -1.0)):
        if _is_total(value):
            total = value
        elif _is_spread(value):
            # A row with the spread is the favourite: +x when away is, -x when home is.
            spread_home = 0.0 if value == 0 else sign * value
    return spread_home, total


def _is_total(value: float | None) -> bool:
    return value is not None and MIN_TOTAL <= value <= MAX_TOTAL


def _is_spread(value: float | None) -> bool:
    return value is not None and value <= MAX_SPREAD


def assign_dates(mmdd_values: list[object], start_year: int) -> list[date | None]:
    """Turn file-ordered MMDD values into dates, starting in `start_year`."""
    dates: list[date | None] = []
    year = start_year
    previous_month: int | None = None
    for value in mmdd_values:
        try:
            mmdd = int(float(str(value)))
            month, day = mmdd // 100, mmdd % 100
            if previous_month is not None and month < previous_month - _NEW_YEAR_MONTH_DROP:
                year += 1
            dates.append(date(year, month, day))
            previous_month = month
        except ValueError:
            dates.append(None)
    return dates


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, dtype=object)
    else:
        raw = pd.read_csv(path, dtype=object)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    return raw


def parse_season_odds(raw: pd.DataFrame, season: str) -> OddsParse:
    """Parse one season's rows into one row per game, recording what couldn't be used."""
    rows = raw.dropna(subset=["date", "team"]).reset_index(drop=True)
    records = rows.to_dict("records")
    dates = assign_dates([row["date"] for row in records], season_start(season))

    games: list[dict] = []
    issues: list[dict] = []
    index = 0
    while index + 1 < len(records):
        first, second = records[index], records[index + 1]
        day, other_day = dates[index], dates[index + 1]
        if day is None or other_day is None or not _is_pair(first, second, day, other_day):
            issues.append({"season": season, "kind": "unpaired_row", "row": index})
            index += 1
            continue
        if day != other_day:
            # The archive sometimes dates the two rows of one game differently. The away row's
            # date is used, and the join's +-1 day score check settles which one is right.
            detail = f"{day} / {other_day} {first['team']} at {second['team']}"
            issues.append({"season": season, "kind": "dates_disagree", "detail": detail})
        game = _parse_game(first, second, day, season, issues)
        if game is not None:
            games.append(game)
        index += 2
    if index == len(records) - 1:
        issues.append({"season": season, "kind": "unpaired_row", "row": index})
    return OddsParse(pd.DataFrame(games, columns=ODDS_COLUMNS), issues)


def _side(row: dict) -> str:
    return str(row.get("vh")).strip().upper()


def _rotation(row: dict) -> int | None:
    try:
        return int(float(str(row.get("rot"))))
    except ValueError:
        return None


def _is_pair(first: dict, second: dict, day: date, other_day: date) -> bool:
    """Return True when two consecutive rows are the away and home rows of one game.

    The away row always comes first, with rotation numbers n and n + 1. When rotation numbers
    are present they decide, so a misdated row can't shift every later pairing. Without them the
    rows must share a date.
    """
    if (_side(first), _side(second)) not in (("V", "H"), ("N", "N")):
        return False
    rotations = _rotation(first), _rotation(second)
    if None in rotations:
        return day == other_day
    return rotations[1] == rotations[0] + 1 and abs((other_day - day).days) <= 1


def _parse_game(
    first: dict, second: dict, day: date, season: str, issues: list[dict]
) -> dict | None:
    away, home = (franchise_from_odds_name(row["team"]) for row in (first, second))
    if away is None or home is None:
        names = f"{first['team']} at {second['team']}"
        issues.append({"season": season, "kind": "unknown_team", "detail": names})
        return None
    spread_home, total = resolve_close(
        parse_line(first.get("close")), parse_line(second.get("close"))
    )
    if spread_home is None or total is None:
        issues.append(
            {"season": season, "kind": "unusable_close", "detail": f"{day} {away} at {home}"}
        )
    return {
        "date": pd.Timestamp(day),
        "season": season,
        "home": home,
        "away": away,
        "neutral": str(first.get("vh")).strip().upper() == "N",
        "home_final": pd.to_numeric(second.get("final"), errors="coerce"),
        "away_final": pd.to_numeric(first.get("final"), errors="coerce"),
        "close_spread_home": spread_home,
        "close_total": total,
        "ml_home": parse_moneyline(second.get("ml")),
        "ml_away": parse_moneyline(first.get("ml")),
    }


def season_from_filename(path: Path) -> str:
    """Read the season from a name such as `nba odds 2007-08.xlsx`."""
    match = _SEASON_IN_NAME.search(path.name)
    if match is None:
        raise ValueError(f"no season in odds file name: {path.name}")
    return season_label(int(match.group(1)))


def load_odds(directory: Path) -> OddsParse:
    """Parse every season file in a directory."""
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in {".xlsx", ".xls", ".csv"}
    )
    if not paths:
        raise FileNotFoundError(f"no odds files in {directory}")
    parsed = [parse_season_odds(_read_raw(path), season_from_filename(path)) for path in paths]
    return OddsParse(
        pd.concat([item.games for item in parsed], ignore_index=True),
        [issue for item in parsed for issue in item.issues],
    )
