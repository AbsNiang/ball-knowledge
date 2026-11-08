"""Generate a reviewable provenance manifest after building processed tables."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backend.data.readiness import FILES
from backend.data.store import PROCESSED_DIR


def write_manifest(directory: Path, join_rates: pd.DataFrame) -> dict:
    files = {
        name: {
            "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            "bytes": (directory / name).stat().st_size,
        }
        for name in FILES
    }
    manifest = {
        "format_version": 1,
        "coverage": {"first_season": "2007-08", "last_season": "2021-22"},
        "sources": [
            {
                "name": "NBA Stats LeagueGameLog via nba_api",
                "url": "https://www.nba.com/stats/teams/boxscores",
            },
            {
                "name": "Sportsbook Reviews Online closing odds archive",
                "url": "https://web.archive.org/web/2023/https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm",
            },
        ],
        "build": "python -m backend.data.build",
        "files": files,
        "join_rates": json.loads(join_rates.to_json(orient="records")),
        "dataset_version": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[
            :16
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    write_manifest(args.directory, pd.read_csv(args.directory / "reports/join_rate_by_season.csv"))
