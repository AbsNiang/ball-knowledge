"""Save the processed tables and control who can read them.

Holdout rows are written to a separate folder at build time, so the public files contain none and
no filter on them can ever reach a holdout game. `get_frame` is the only reader for general use.
`read_sealed_holdout` exists only for `holdout_confirm`, and a test fails if any other module
references it.
"""

from pathlib import Path

import pandas as pd

from backend.data.seasons import split_of

PROCESSED_DIR = Path(__file__).parent / "processed"
GAMES_FILE = "games.parquet"
TEAM_GAMES_FILE = "team_games.parquet"
SEALED_DIR = "sealed"
PUBLIC_SPLITS = ("explore", "validate", "special")


def _splits(frame: pd.DataFrame) -> pd.Series:
    """Return the split label of every row."""
    return frame["season"].map(split_of)


def save_tables(games: pd.DataFrame, team_games: pd.DataFrame, directory: Path) -> None:
    """Write public and sealed tables to separate locations."""
    sealed = directory / SEALED_DIR
    sealed.mkdir(parents=True, exist_ok=True)
    for frame, name in ((games, GAMES_FILE), (team_games, TEAM_GAMES_FILE)):
        is_holdout = _splits(frame) == "holdout"
        frame[~is_holdout].to_parquet(directory / name, index=False)
        frame[is_holdout].to_parquet(sealed / name, index=False)


def get_frame(split: str, directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Return `team_games` rows for a public split."""
    if split not in PUBLIC_SPLITS:
        raise ValueError(f"split must be one of {PUBLIC_SPLITS}, got {split!r}")
    frame = pd.read_parquet(directory / TEAM_GAMES_FILE)
    return frame[_splits(frame) == split].reset_index(drop=True)


def get_games(split: str, directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Return `games` rows for a public split."""
    if split not in PUBLIC_SPLITS:
        raise ValueError(f"split must be one of {PUBLIC_SPLITS}, got {split!r}")
    frame = pd.read_parquet(directory / GAMES_FILE)
    return frame[_splits(frame) == split].reset_index(drop=True)


def read_sealed_holdout(directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Return holdout `team_games` rows. Only `holdout_confirm` may call this."""
    return pd.read_parquet(directory / SEALED_DIR / TEAM_GAMES_FILE)
