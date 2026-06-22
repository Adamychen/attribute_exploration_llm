"""
Tests for the correlation script.

Verifies that the LaTeX generator returns a valid table and that the
correlation helpers behave correctly on synthetic data.
"""
import os
import sys

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from experiments.correlation import _safe_corr, _fmt_corr, generate_correlation_table


def _synthetic_fca_df() -> pd.DataFrame:
    rows = []
    models = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
    rng = np.random.default_rng(7)
    for m in models:
        for ctx in ["integers", "graphs", "rcc", "top"]:
            for qm in ["implicational", "existential"]:
                rows.append({
                    "model": m,
                    "context_name": ctx,
                    "prompt_mode": "basic",
                    "question_mode": qm,
                    "early_stop": False,
                    "timeout": False,
                    "S": float(rng.uniform(0.4, 0.9)),
                    "S_H": float(rng.uniform(0.5, 0.95)),
                })
    return pd.DataFrame(rows)


def _synthetic_bench_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "model": [f"m{i}" for i in range(1, 9)],
        "accuracy_mmlu": rng.uniform(0.3, 0.9, 8).round(3),
        "accuracy_gsm8k": rng.uniform(0.1, 0.7, 8).round(3),
    })


def test_safe_corr_positive():
    """A perfectly correlated pair has r=1, p=0."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    r, p, n = _safe_corr(x, y, "pearson")
    assert abs(r - 1.0) < 1e-9
    assert p < 0.001
    assert n == 5
    print(f"OK test_safe_corr_positive (r={r:.4f}, p={p:.4f}, n={n})")


def test_safe_corr_too_small():
    """N<3 returns NaN, NaN, N."""
    x = pd.Series([1.0, 2.0])
    y = pd.Series([1.0, 2.0])
    r, p, n = _safe_corr(x, y, "pearson")
    assert np.isnan(r) and np.isnan(p) and n == 2
    print("OK test_safe_corr_too_small")


def test_safe_corr_constant():
    """A constant series returns NaN, NaN, N (no variance)."""
    x = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    r, p, n = _safe_corr(x, y, "pearson")
    assert np.isnan(r) and np.isnan(p)
    print("OK test_safe_corr_constant")


def test_safe_corr_with_nan():
    """NaN entries are dropped before computing correlation."""
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    y = pd.Series([2.0, 4.0, 6.0, np.nan, 10.0])
    r, p, n = _safe_corr(x, y, "pearson")
    assert n == 3, f"Expected 3, got {n}"
    assert abs(r - 1.0) < 1e-9
    print(f"OK test_safe_corr_with_nan (r={r:.4f}, n={n})")


def test_fmt_corr_with_significance():
    """_fmt_corr appends significance asterisks for p < 0.05 / 0.01 / 0.001."""
    assert "0.85" in _fmt_corr(0.85, 0.01)
    assert "0.50" in _fmt_corr(0.50, 0.10)
    assert "---" == _fmt_corr(float("nan"), float("nan"))
    assert "^{***}" in _fmt_corr(0.95, 0.0001)
    print("OK test_fmt_corr_with_significance")


def test_generate_correlation_table_smoke():
    """The full LaTeX generator produces a valid tabular for synthetic data."""
    fca = _synthetic_fca_df()
    bench = _synthetic_bench_df()
    tex = generate_correlation_table(fca, bench)
    assert r"\begin{table}" in tex
    assert r"\end{table}" in tex
    assert "Pearson" in tex or "$r_p$" in tex
    assert "Spearman" in tex or "$r_s$" in tex
    print(f"OK test_generate_correlation_table_smoke ({len(tex)} chars)")


def test_generate_correlation_table_column_count():
    """Regression: the col spec must match the number of data columns
    (Context + N + 2 fca_cols × 2 benchmark_cols × 2 = 10 columns).
    Previous versions emitted 'l' + 'cc' * (ncols-1), doubling the col count
    and producing an unrenderable table."""
    import re
    fca = _synthetic_fca_df()
    bench = _synthetic_bench_df()
    tex = generate_correlation_table(fca, bench)
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", tex)
    assert m, "tabular not found"
    spec = m.group(1)
    ncols = len(spec)
    # 2 fca_cols × 2 benchmark_cols × 2 (r_p, r_s) + 2 (Context, N) = 10
    assert ncols == 10, f"Expected 10 columns, got {ncols} (spec={spec!r})"
    # And the data row should also have 10 &-separated values.
    data_rows = [
        line for line in tex.split("\n")
        if line.strip().startswith(("Integer", "Graph", "Region", "KR "))
    ]
    assert data_rows, "No data rows found"
    n_amp = data_rows[0].count("&")
    assert n_amp == 9, f"Expected 9 '&' separators, got {n_amp}"
    print(f"OK test_generate_correlation_table_column_count (spec={spec!r}, ncols={ncols})")


def test_generate_correlation_table_no_overlap():
    """Empty merge yields a comment, not a crash."""
    fca = pd.DataFrame(columns=["model", "context_name", "early_stop", "timeout", "S", "S_H"])
    bench = pd.DataFrame(columns=["model", "accuracy_mmlu", "accuracy_gsm8k"])
    tex = generate_correlation_table(fca, bench)
    assert "No overlap" in tex or "No completed" in tex
    print("OK test_generate_correlation_table_no_overlap")


if __name__ == "__main__":
    test_safe_corr_positive()
    test_safe_corr_too_small()
    test_safe_corr_constant()
    test_safe_corr_with_nan()
    test_fmt_corr_with_significance()
    test_generate_correlation_table_smoke()
    test_generate_correlation_table_column_count()
    test_generate_correlation_table_no_overlap()
    print("\nAll correlation tests passed.")
