"""Statistical gates, global test ledger, and verdict rules.

- `config`: thresholds.
- `gates`: `StatsGate`, the G1-G7 checks on tool results.
- `ledger`: the run's test ledger and Benjamini-Hochberg correction (G4).
- `verdict`: `compute_verdict` and the chance rule.
"""
