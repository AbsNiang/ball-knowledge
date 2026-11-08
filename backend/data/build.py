"""Build the dataset from raw downloads.

Download the raw files with `python -m backend.data.fetch`, or put them here by hand, then run
this from the repo root:

    backend/data/raw/results/games.csv        Kaggle dataset, or
    backend/data/raw/results/gamelog.csv      saved nba_api game log
    backend/data/raw/odds/*.xlsx              one file per season, named with the season

    .\\venv\\Scripts\\python -m backend.data.build

Tables go to `backend/data/processed/`, and everything that failed to line up goes to
`backend/data/processed/reports/` so it can be read and fixed. The build fails if the join rate
is under 98% unless `--allow-low-join` is passed, which is meant for early partial runs.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.data.join import MIN_JOIN_RATE, JoinResult, join_odds
from backend.data.manifest import write_manifest
from backend.data.odds import load_odds
from backend.data.results import load_lauga_games, load_nba_api_gamelog
from backend.data.store import PROCESSED_DIR, save_tables
from backend.data.tables import build_team_games

RAW_DIR = Path(__file__).parent / "raw"


@dataclass
class BuildReport:
    join: JoinResult
    odds_issues: pd.DataFrame
    team_games_rows: int

    @property
    def join_rate(self) -> float:
        return self.join.join_rate


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load whichever results file is present, preferring the nba_api log.

    The nba_api log wins because the Kaggle dataset may stop before the latest holdout seasons.
    """
    gamelog, kaggle = results_dir / "gamelog.csv", results_dir / "games.csv"
    if gamelog.exists():
        return load_nba_api_gamelog(gamelog)
    if kaggle.exists():
        return load_lauga_games(kaggle)
    raise FileNotFoundError(f"expected gamelog.csv or games.csv in {results_dir}")


def build(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> BuildReport:
    """Run the pipeline and write tables and reports.

    Reports can name holdout games, because join problems need fixing there too. They are for
    people debugging the build: no tool reads them, and they stay out of git.
    """
    results = load_results(raw_dir / "results")
    odds = load_odds(raw_dir / "odds")
    joined = join_odds(results, odds.games)
    team_games = build_team_games(joined.games)

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_tables(joined.games, team_games, processed_dir)

    issues = pd.DataFrame(odds.issues, columns=["season", "kind", "row", "detail"])
    reports = processed_dir / "reports"
    reports.mkdir(exist_ok=True)
    joined.rate_by_season.to_csv(reports / "join_rate_by_season.csv", index=False)
    joined.unmatched_games.to_csv(reports / "unmatched_games.csv", index=False)
    joined.unused_odds.to_csv(reports / "unused_odds.csv", index=False)
    joined.score_mismatches.to_csv(reports / "score_mismatches.csv", index=False)
    issues.to_csv(reports / "odds_issues.csv", index=False)
    write_manifest(processed_dir, joined.rate_by_season)
    return BuildReport(joined, issues, len(team_games))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the NBA results and odds dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--allow-low-join", action="store_true")
    args = parser.parse_args(argv)

    report = build(args.raw_dir, args.processed_dir)
    print(report.join.rate_by_season.to_string(index=False))
    print(f"\njoin rate: {report.join_rate:.2%}")
    print(f"games: {len(report.join.games)}, team_games rows: {report.team_games_rows}")
    print(f"unmatched games: {len(report.join.unmatched_games)}")
    print(f"unused odds rows: {len(report.join.unused_odds)}")
    print(f"score mismatches: {len(report.join.score_mismatches)}")
    print(f"odds issues: {len(report.odds_issues)}")

    if report.join_rate < MIN_JOIN_RATE and not args.allow_low_join:
        print(f"\nFAILED: join rate is below {MIN_JOIN_RATE:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
