"""
Tests for the analysis_prompts script.

Verifies that the LaTeX generators are stable and produce valid tabular
output for a small synthetic DataFrame.
"""
import os
import sys

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from experiments.analysis_prompts import (
    generate_prompt_ablation_table,
    _wilcoxon_paired,
    _format_p,
)


def _synthetic_df() -> pd.DataFrame:
    rows = []
    models = ["m1", "m2", "m3", "m4", "m5"]
    rng = np.random.default_rng(42)
    for m in models:
        for ctx in ["integers", "graphs"]:
            for qm in ["implicational", "existential"]:
                for p in ["basic", "full_context", "stem_basis"]:
                    rows.append({
                        "model": m,
                        "context_name": ctx,
                        "prompt_mode": p,
                        "question_mode": qm,
                        "early_stop": False,
                        "timeout": False,
                        "S": float(rng.uniform(0.5, 0.9)),
                        "S_H": float(rng.uniform(0.6, 0.95)),
                    })
    return pd.DataFrame(rows)


def test_wilcoxon_paired_basic():
    """Wilcoxon returns (stat, p) for a paired sample with non-zero diffs."""
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    stat, p = _wilcoxon_paired(a, b)
    assert 0.0 <= p <= 1.0, f"p out of range: {p}"
    assert stat >= 0
    print(f"OK test_wilcoxon_paired_basic (p={p:.4f})")


def test_wilcoxon_paired_constant():
    """Wilcoxon with all-equal diffs returns NaN, NaN."""
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([1.0, 2.0, 3.0])
    stat, p = _wilcoxon_paired(a, b)
    assert np.isnan(stat) and np.isnan(p), f"Expected NaN, got ({stat}, {p})"
    print("OK test_wilcoxon_paired_constant")


def test_wilcoxon_paired_too_small():
    """Wilcoxon with N<2 returns NaN, NaN."""
    a = pd.Series([1.0])
    b = pd.Series([2.0])
    stat, p = _wilcoxon_paired(a, b)
    assert np.isnan(stat) and np.isnan(p)
    print("OK test_wilcoxon_paired_too_small")


def test_format_p_thresholds():
    """_format_p respects the <0.001 and <0.01 conventions."""
    assert _format_p(0.0001) == "<0.001"
    assert _format_p(0.005).startswith("0.0")
    assert _format_p(0.123) == "0.123"
    assert _format_p(float("nan")) == "n/a"
    print("OK test_format_p_thresholds")


def test_generate_table_smoke():
    """The full LaTeX generator produces a non-empty tabular for synthetic data."""
    df = _synthetic_df()
    tex = generate_prompt_ablation_table(df, metric="S")
    assert r"\begin{table}" in tex
    assert r"\end{table}" in tex
    assert "Integer Numbers" in tex
    assert r"\toprule" in tex
    print(f"OK test_generate_table_smoke ({len(tex)} chars)")


def test_generate_table_empty():
    """Empty DataFrame yields a comment, not a crash."""
    df = pd.DataFrame(columns=["model", "context_name", "prompt_mode", "question_mode",
                                "early_stop", "timeout", "S", "S_H"])
    tex = generate_prompt_ablation_table(df, metric="S")
    assert "No completed runs" in tex
    print("OK test_generate_table_empty")


if __name__ == "__main__":
    test_wilcoxon_paired_basic()
    test_wilcoxon_paired_constant()
    test_wilcoxon_paired_too_small()
    test_format_p_thresholds()
    test_generate_table_smoke()
    test_generate_table_empty()
    print("\nAll analysis_prompts tests passed.")
