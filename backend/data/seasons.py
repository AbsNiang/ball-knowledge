"""Season labels and the explore/validate/special/holdout split.

Boundaries depend on odds coverage (open question Q4), so they live here and nowhere else.
"""

from datetime import date
from typing import Literal

Split = Literal["explore", "validate", "special", "holdout"]

FIRST_SEASON_START = 2007
EXPLORE_STARTS = range(2007, 2016)  # 2007-08 to 2015-16
VALIDATE_STARTS = range(2016, 2019)  # 2016-17 to 2018-19
SPECIAL_STARTS = range(2019, 2021)  # 2019-20 and 2020-21
HOLDOUT_FIRST_START = 2021  # 2021-22 onward, sealed

BUBBLE_FIRST_DAY = date(2020, 7, 30)
BUBBLE_LAST_DAY = date(2020, 10, 11)


def season_label(start_year: int) -> str:
    """Return the label for a season, for example 2007 -> "2007-08"."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start(label: str) -> int:
    """Return the calendar year a season label starts in, for example "2007-08" -> 2007."""
    return int(label[:4])


def split_of(label: str) -> Split:
    """Return the split a season belongs to."""
    start = season_start(label)
    if start < FIRST_SEASON_START:
        raise ValueError(f"season {label} is before the study period")
    if start in EXPLORE_STARTS:
        return "explore"
    if start in VALIDATE_STARTS:
        return "validate"
    if start in SPECIAL_STARTS:
        return "special"
    return "holdout"


def is_bubble_day(day: date) -> bool:
    """Return True for dates inside the 2020 Orlando bubble."""
    return BUBBLE_FIRST_DAY <= day <= BUBBLE_LAST_DAY
