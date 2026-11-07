"""Resampling core: whole games, blocked by season.

Rows are packed into a games x 2 table (home slot, away slot), so a game's two rows always move
together. Each slot carries a value, whether the value is usable, and a group label (0 neither,
1 group A, 2 group B).

- Permutation: within each season, the label pattern of each game is shuffled to another game,
  keeping the values in place. Group sizes per season are preserved.
- Bootstrap: within each season, games are resampled with replacement.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_95 = 1.959964  # two-sided 5%
Z_POWER_80 = 0.841621
_BATCH = 250


@dataclass(frozen=True)
class GameTable:
    values: np.ndarray  # (games, 2), 0 where not usable
    valid: np.ndarray  # (games, 2) bool
    labels: np.ndarray  # (games, 2) int8
    season_codes: np.ndarray  # (games,), ascending
    season_sizes: list[int]

    @property
    def n_games(self) -> int:
        return len(self.season_codes)

    def counts(self) -> tuple[int, int]:
        """Usable rows in group A and group B."""
        usable = self.valid
        return int((usable & (self.labels == 1)).sum()), int((usable & (self.labels == 2)).sum())


def build_table(rows: pd.DataFrame, values: np.ndarray, labels: np.ndarray) -> GameTable:
    """Pack labelled rows (needs game_id, season, is_home) into a game table."""
    keep = labels > 0
    rows, values, labels = rows[keep], values[keep], labels[keep]
    games = (
        rows[["game_id", "season"]]
        .drop_duplicates("game_id")
        .sort_values(["season", "game_id"])
        .reset_index(drop=True)
    )
    position = pd.Series(games.index.to_numpy(), index=games["game_id"].to_numpy())
    game = position.loc[rows["game_id"].to_numpy()].to_numpy()
    slot = np.where(rows["is_home"].to_numpy(dtype=bool), 0, 1)

    n_games = len(games)
    table_values = np.zeros((n_games, 2))
    table_valid = np.zeros((n_games, 2), dtype=bool)
    table_labels = np.zeros((n_games, 2), dtype=np.int8)
    usable = ~np.isnan(values)
    table_values[game, slot] = np.where(usable, values, 0.0)
    table_valid[game, slot] = usable
    table_labels[game, slot] = labels
    codes, _ = pd.factorize(games["season"], sort=True)
    sizes = np.bincount(codes).tolist() if n_games else []
    return GameTable(table_values, table_valid, table_labels, codes, sizes)


def _means(sum_a, count_a, sum_b, count_b, kind: str):
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = sum_a / count_a
        if kind == "mean":
            return mean_a
        return mean_a - sum_b / count_b


def _per_game(table: GameTable, labels: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    in_group = (labels == group) & table.valid
    return (table.values * in_group).sum(axis=-1), in_group.sum(axis=-1)


def observed(table: GameTable, kind: str = "difference") -> float:
    """Mean of A minus mean of B, or the mean of A alone when `kind="mean"`."""
    sum_a, count_a = _per_game(table, table.labels, 1)
    sum_b, count_b = _per_game(table, table.labels, 2)
    return float(_means(sum_a.sum(), count_a.sum(), sum_b.sum(), count_b.sum(), kind))


def _within_seasons(table: GameTable, size: int, rng: np.random.Generator, draw) -> np.ndarray:
    """Stack per-season game indices drawn by `draw(first, count)` into a (size, games) array."""
    blocks, first = [], 0
    for count in table.season_sizes:
        blocks.append(draw(first, count))
        first += count
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((size, 0), dtype=int)


def permutation_null(table: GameTable, n: int, rng: np.random.Generator) -> np.ndarray:
    """Differences in means with games' labels shuffled within seasons.

    Each game's label pattern goes to another game in the same season with a random home/away
    orientation, so the null matches the raw difference in means that `effect` reports. Keeping
    orientation would build any home/away gap into the null instead; adjusting for home/away is
    what `regress` with an `is_home` control is for.

    Moving values among games while labels stay put has the same distribution as moving labels,
    and lets both groups' sums come from one matrix product.
    """
    weighted = table.values * table.valid
    usable = table.valid.astype(float)
    # Contiguous 1-D columns gather far faster than `array[order, slot]`.
    # float32 halves the memory traffic; sums are still accumulated in float64.
    home_weighted, away_weighted = (weighted[:, slot].astype(np.float32) for slot in (0, 1))
    home_valid, away_valid = (usable[:, slot].astype(np.float32) for slot in (0, 1))
    # members[:, slot, group]: is this slot in group A (0) or group B (1)?
    members = np.stack([table.labels == 1, table.labels == 2], axis=-1).astype(np.float64)
    results = []
    for start in range(0, n, _BATCH):
        size = min(_BATCH, n - start)
        order = _within_seasons(
            table,
            size,
            rng,
            lambda first, count, size=size: rng.permuted(
                np.broadcast_to(
                    np.arange(first, first + count, dtype=np.int32), (size, count)
                ),
                axis=1,
            ),
        )
        flip = rng.random(order.shape) < 0.5
        home_value, away_value = home_weighted[order], away_weighted[order]
        home_usable, away_usable = home_valid[order], away_valid[order]
        first_value = np.where(flip, away_value, home_value)
        first_usable = np.where(flip, away_usable, home_usable)
        second_value = home_value + away_value - first_value
        second_usable = home_usable + away_usable - first_usable
        first_members, second_members = members[:, 0], members[:, 1]
        sums = first_value @ first_members + second_value @ second_members
        counts = first_usable @ first_members + second_usable @ second_members
        results.append(_means(sums[:, 0], counts[:, 0], sums[:, 1], counts[:, 1], "difference"))
    return np.concatenate(results) if results else np.array([])


def bootstrap(table: GameTable, n: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    """The statistic over game resamples drawn with replacement within each season."""
    sum_a, count_a = _per_game(table, table.labels, 1)
    sum_b, count_b = _per_game(table, table.labels, 2)
    results = []
    for start in range(0, n, _BATCH):
        size = min(_BATCH, n - start)
        drawn = _within_seasons(
            table,
            size,
            rng,
            lambda first, count, size=size: rng.integers(first, first + count, (size, count)),
        )
        results.append(
            _means(
                sum_a[drawn].sum(1),
                count_a[drawn].sum(1),
                sum_b[drawn].sum(1),
                count_b[drawn].sum(1),
                kind,
            )
        )
    return np.concatenate(results) if results else np.array([])


def permutation_p(null: np.ndarray, observed_value: float, direction: str) -> float:
    """P-value with the +1 correction, so it is never exactly 0."""
    null = null[~np.isnan(null)]
    tolerance = 1e-12
    if direction == "a_gt_b":
        extreme = null >= observed_value - tolerance
    elif direction == "a_lt_b":
        extreme = null <= observed_value + tolerance
    else:
        extreme = np.abs(null) >= abs(observed_value) - tolerance
    return float((extreme.sum() + 1) / (len(null) + 1))


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def min_detectable_effect(standard_error: float) -> float:
    """Smallest true effect detected 80% of the time at a two-sided 5% level."""
    return (Z_95 + Z_POWER_80) * standard_error


def histogram(samples: np.ndarray, bins: int = 30) -> dict:
    samples = samples[~np.isnan(samples)]
    counts, edges = np.histogram(samples, bins=bins)
    return {"edges": [round(float(edge), 5) for edge in edges], "counts": counts.tolist()}
