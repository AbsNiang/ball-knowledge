"""Statistical thresholds and constants, kept in one place (plan.md v2 section 9.1)."""

ALPHA = 0.05
BH_Q = 0.05
N_MIN = 200
RETENTION = 0.5
BREAKEVEN = 0.5238  # 110/210, the win rate needed to break even at -110

SESOI_POINTS = 1.0  # smallest effect of interest for margin metrics
SESOI_RATE = 0.03  # smallest effect of interest for binary and ROI metrics

SEED = 1729
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 5_000
