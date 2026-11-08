"""Verify packaged bytes and public tables; never deserialize sealed outcomes."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.data.join import GAMES_COLUMNS
from backend.data.seasons import split_of
from backend.data.tables import TEAM_GAMES_COLUMNS

FILES = ("games.parquet", "team_games.parquet", "sealed/games.parquet", "sealed/team_games.parquet")


def validate_dataset(directory: Path) -> dict:
    paths = [directory / "manifest.json", *(directory / name for name in FILES)]
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    return _validate(signature)


@lru_cache(maxsize=4)
def _validate(signature: tuple) -> dict:
    directory = Path(signature[0][0]).parent
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["coverage"] != {"first_season": "2007-08", "last_season": "2021-22"}:
        raise ValueError("Unsupported historical coverage")
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported dataset manifest")
    version = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:16]
    if version != manifest["dataset_version"]:
        raise ValueError("Invalid dataset version")
    for name in FILES:
        path = directory / name
        expected = manifest["files"][name]
        if (
            path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Dataset checksum mismatch")
    tables = []
    for name, columns in ((FILES[0], GAMES_COLUMNS), (FILES[1], TEAM_GAMES_COLUMNS)):
        frame = pd.read_parquet(directory / name)
        if not set(columns) <= set(frame.columns) or frame.empty:
            raise ValueError("Invalid public schema")
        if not frame["season"].map(split_of).isin(["explore", "validate", "special"]).all():
            raise ValueError("Invalid public split boundaries")
        if set(frame["season"]) != {f"{y}-{(y + 1) % 100:02d}" for y in range(2007, 2021)}:
            raise ValueError("Incomplete public seasons")
        tables.append(frame)
    games, teams = tables
    if games.game_id.duplicated().any() or teams.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate game keys")
    if len(teams) != 2 * len(games) or set(games.game_id) != set(teams.game_id):
        raise ValueError("Invalid game join")
    if not teams.groupby("game_id").size().eq(2).all():
        raise ValueError("Invalid team pairs")
    return manifest
