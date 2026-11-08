"""The global test ledger behind G4 (plan.md v2 section 7.3, ledger entry rule).

Every p-value from `compare_groups`, `regress`, `persistence` and `market_test` is an entry,
keyed by the hash of the tool and its arguments. Re-running an identical spec adds nothing;
changing anything adds an entry, so the forking paths are counted. Diagnostics
(`out_of_sample`, `placebo`, `power`, `holdout_confirm`) are not entries.
"""

import math

from backend.gate import config
from backend.tools.hashing import hash_args
from backend.tools.schemas import ToolResult

LEDGER_TOOLS = frozenset({"compare_groups", "regress", "persistence", "market_test"})


def entry_key(result: ToolResult) -> str:
    """The same key the real tools report as `extra["spec_hash"]`."""
    return hash_args({"tool": result.tool, "args": result.args})


class Ledger:
    def __init__(self, q: float = config.BH_Q) -> None:
        self.q = q
        self._p: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._p)

    def add(self, result: ToolResult) -> str | None:
        """Record a ledger tool's p-value. Returns its key, or None if it isn't an entry."""
        p = result.p_value
        if result.tool not in LEDGER_TOOLS or result.status != "ok" or p is None:
            return None
        if not math.isfinite(p):
            return None
        key = entry_key(result)
        self._p.setdefault(key, p)
        return key

    def p_value(self, key: str) -> float:
        return self._p[key]

    def rejected(self) -> set[str]:
        """Benjamini-Hochberg step-up at `q` over every entry so far."""
        ranked = sorted(self._p.items(), key=lambda item: item[1])
        m = len(ranked)
        cutoff = 0
        for rank, (_, p) in enumerate(ranked, 1):
            if p <= rank / m * self.q:
                cutoff = rank
        return {key for key, _ in ranked[:cutoff]}

    def rejects(self, key: str) -> bool:
        return key in self.rejected()
