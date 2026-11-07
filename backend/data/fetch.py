"""Download the raw files the build needs, then run the build separately.

    .\\venv\\Scripts\\python -m pip install -e ".[data]"
    .\\venv\\Scripts\\python -m backend.data.fetch
    .\\venv\\Scripts\\python -m backend.data.build

Odds: the Sportsbook Reviews Online archive no longer serves its season files, so they come from
the Internet Archive's copies, taken in January 2023. Those copies are complete through 2021-22;
2022-23 stops in mid-January, which would fail the join check, so it isn't downloaded.

Results: `nba_api`, through `results.pull_nba_api_gamelog`.

Files that already exist are skipped, so an interrupted run can simply be rerun.
"""

import argparse
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from backend.data.build import RAW_DIR
from backend.data.results import pull_nba_api_gamelog
from backend.data.seasons import FIRST_SEASON_START, season_label

LAST_COMPLETE_ODDS_START = 2021  # 2021-22
ODDS_URL = (
    "https://web.archive.org/web/2023if_/https://www.sportsbookreviewsonline.com/"
    "scoresoddsarchives/nba/nba%20odds%20{season}.xlsx"
)
_XLSX_SIGNATURE = b"PK"  # .xlsx files are zip archives
_ATTEMPTS = 3


def odds_url(start_year: int) -> str:
    return ODDS_URL.format(season=season_label(start_year))


def odds_path(start_year: int, odds_dir: Path) -> Path:
    return odds_dir / f"nba odds {season_label(start_year)}.xlsx"


def _download(url: str, path: Path) -> None:
    """Save a URL to a file, retrying with growing waits."""
    request = urllib.request.Request(url, headers={"User-Agent": "whack-data-fetch"})
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                path.write_bytes(response.read())
            return
        except OSError:
            if attempt == _ATTEMPTS:
                raise
            time.sleep(5 * attempt)


def fetch_odds(
    start_years: range,
    odds_dir: Path,
    download: Callable[[str, Path], None] = _download,
) -> list[Path]:
    """Download each season's odds file that isn't already present."""
    odds_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for start_year in start_years:
        path = odds_path(start_year, odds_dir)
        if path.exists():
            continue
        download(odds_url(start_year), path)
        if path.read_bytes()[:2] != _XLSX_SIGNATURE:
            path.unlink()
            raise ValueError(f"{odds_url(start_year)} did not return an Excel file")
        fetched.append(path)
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw odds and results.")
    parser.add_argument("--first", type=int, default=FIRST_SEASON_START, help="first season start")
    parser.add_argument("--last", type=int, default=LAST_COMPLETE_ODDS_START, help="last start")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    seasons = range(args.first, args.last + 1)
    fetched = fetch_odds(seasons, args.raw_dir / "odds")
    print(f"odds: downloaded {len(fetched)}, already present {len(seasons) - len(fetched)}")
    pull_nba_api_gamelog(seasons, args.raw_dir / "results" / "gamelog.csv")
    print(f"results: saved to {args.raw_dir / 'results' / 'gamelog.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
