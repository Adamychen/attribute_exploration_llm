"""Power of the sign test used in the prompt-sensitivity analysis.

For each cell where a variant applies, the probability that an exploration
finishes under C0 is taken as the mean of the two executions of the main sweep
(0, 0.5 or 1); under the variant it is shifted by ``delta`` (clipped to [0, 1],
so the mean shift actually applied is smaller than ``delta`` and is printed).
Pairs are drawn independently and the exact two-sided sign test is applied to the
discordant pairs, at alpha = 0.05 and at alpha = 0.05 / 4 (the smallest Holm
threshold over the four variants).

Usage:
    python -m experiments.sensitivity_power
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import binomtest

from experiments.analyze_sensitivity import CELL, VARIANTS, cells_of, load_all


def cell_probabilities(runs, variant: str) -> np.ndarray:
    cells = cells_of(runs, variant)
    main = runs[runs.condition.isin(["main1", "main2"])]
    p = cells.merge(main, on=CELL).groupby(CELL).finished.mean()
    return p.to_numpy(dtype=float)


def power(p: np.ndarray, delta: float, alpha: float, n_sim: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    q = np.clip(p + delta, 0, 1)
    hits = 0
    for _ in range(n_sim):
        a = rng.random(len(p)) < p
        b = rng.random(len(p)) < q
        lost, gained = int((a & ~b).sum()), int((~a & b).sum())
        if lost + gained and binomtest(gained, lost + gained, 0.5).pvalue < alpha:
            hits += 1
    return hits / n_sim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-sim", type=int, default=10_000)
    args = ap.parse_args()
    runs = load_all()
    print("variant     pairs  delta  mean shift  power(a=0.05)  power(a=0.0125)")
    for v in VARIANTS:
        p = cell_probabilities(runs, v)
        for delta in (-0.20, -0.10, 0.10, 0.20):
            shift = float(np.mean(np.clip(p + delta, 0, 1) - p))
            print(f"{v:<11} {len(p):>5}  {delta:+.2f}  {shift:+10.3f}"
                  f"   {power(p, delta, 0.05, args.n_sim):>11.2f}"
                  f"   {power(p, delta, 0.0125, args.n_sim):>13.2f}")


if __name__ == "__main__":
    main()
