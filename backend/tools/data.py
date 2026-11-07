"""Read-only access to the built dataset for the tools.

The public table is loaded once and cached until its file changes. `data_version` is a hash of
that file, so `slice_hash` changes whenever the data does, which keeps results reproducible from
`(args, data version, seed)`.

Holdout rows aren't reachable from here: only `backend/tools/holdout.py` reads them.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.data.seasons import split_of
from backend.data.store import PROCESSED_DIR, TEAM_GAMES_FILE

_data_dir: Path = PROCESSED_DIR


def data_dir() -> Path:
    return _data_dir


def use_data_dir(directory: Path) -> None:
    """Point the tools at another built dataset. Tests use this."""
    global _data_dir
    _data_dir = Path(directory)


def is_built() -> bool:
    return (_data_dir / TEAM_GAMES_FILE).exists()


def public_frame() -> pd.DataFrame:
    """Every public team-game row, with a `split` column."""
    path = _data_dir / TEAM_GAMES_FILE
    if not path.exists():
        raise FileNotFoundError(f"dataset not built: {path}")
    stat = path.stat()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)[0]


def data_version() -> str:
    path = _data_dir / TEAM_GAMES_FILE
    stat = path.stat()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)[1]


@lru_cache(maxsize=4)
def _load(path: str, _mtime: int, _size: int) -> tuple[pd.DataFrame, str]:
    raw = Path(path).read_bytes()
    frame = pd.read_parquet(path)
    frame["split"] = frame["season"].map(split_of)
    return frame, hashlib.sha256(raw).hexdigest()[:12]
