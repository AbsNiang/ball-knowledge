"""Turn slice specs into row masks and metrics into per-row values."""

import numpy as np
import pandas as pd

from backend.tools.schemas import Filter, SliceSpec

_BOOL_COLUMNS = {"is_home", "b2b", "opp_b2b", "is_bubble", "is_playoffs"}
_TEXT_COLUMNS = {"season", "team", "opp"}
_WIN_PAYOUT = 100 / 110  # profit on a winning -110 bet per unit staked


class SliceError(ValueError):
    """A filter that can't be applied, such as a text value on a numeric column."""


def _check_value(column: str, value: object) -> None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if column in _BOOL_COLUMNS:
            ok = isinstance(item, bool)
        elif column in _TEXT_COLUMNS:
            ok = isinstance(item, str)
        else:
            ok = isinstance(item, (int, float)) and not isinstance(item, bool)
        if not ok:
            raise SliceError(f"{column} can't be compared with {item!r}")


def _mask(frame: pd.DataFrame, item: Filter) -> pd.Series:
    _check_value(item.col, item.value)
    column = frame[item.col]
    if item.op == "eq":
        return column == item.value
    if item.op == "ne":
        return column != item.value
    if item.op == "in":
        return column.isin(item.value)
    if item.op == "gte":
        return column >= item.value
    if item.op == "lte":
        return column <= item.value
    low, high = item.value
    return column.between(low, high)


def select(frame: pd.DataFrame, spec: SliceSpec, *, use_split: bool = True) -> pd.Series:
    """Return a boolean mask of the rows in a slice.

    `use_split=False` ignores the spec's split, for running a frozen spec on the holdout frame.
    """
    mask = frame["split"] == spec.split if use_split else pd.Series(True, index=frame.index)
    for item in spec.filters:
        mask &= _mask(frame, item).fillna(False).astype(bool)
    return mask


def complement(frame: pd.DataFrame, spec: SliceSpec, group_a: pd.Series, *, use_split: bool = True):
    """Rows in the same split as group A that aren't in group A."""
    in_split = frame["split"] == spec.split if use_split else pd.Series(True, index=frame.index)
    return in_split & ~group_a


def metric_values(frame: pd.DataFrame, metric: str) -> np.ndarray:
    """Per-row values, with NaN where the metric is undefined (pushes or no line)."""
    if metric in ("ats_margin", "margin"):
        return frame[metric].to_numpy(dtype=float)
    if metric == "win":
        return frame["win"].to_numpy(dtype=float)
    if metric in ("ats_cover", "total_over"):
        return frame[metric].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    if metric == "roi":
        # Flat one-unit ATS bet at -110: a push returns the stake, so it counts as 0, not NaN.
        cover = frame["ats_cover"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        margin = frame["ats_margin"].to_numpy(dtype=float)
        roi = np.where(cover == 1, _WIN_PAYOUT, -1.0)
        roi = np.where(margin == 0, 0.0, roi)
        return np.where(np.isnan(margin), np.nan, roi)
    raise ValueError(f"unknown metric {metric}")
