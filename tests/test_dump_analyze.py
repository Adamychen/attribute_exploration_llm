"""
Tests for the dump_prompts and analyze_traces scripts.

Verifies the LaTeX generators and the trace loader are stable.
"""
import os
import sys
import json
import tempfile

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
sys.path.insert(0, ROOT)

import pandas as pd

from experiments.dump_prompts import build_appendix, _escape_tex
from experiments.analyze_traces import load_traces, build_per_context_table, build_per_size_bin_table


def test_escape_tex_curly_and_percent():
    """_escape_tex must escape {, }, and % but leave FOL backslash commands alone."""
    s = "{predicates_description} 50% \\land \\forall x"
    out = _escape_tex(s)
    assert r"\{" in out and r"\}" in out
    assert r"\%" in out
    assert r"\land" in out and r"\forall" in out
    print("OK test_escape_tex_curly_and_percent")


def test_build_appendix_includes_all_prompts():
    """The appendix contains one verbatim block per (context, strategy)."""
    tex = build_appendix()
    assert r"\begin{table}" not in tex
    assert r"\begin{verbatim}" in tex
    assert "Integer Numbers" in tex
    assert "Graph Theory" in tex
    assert "Region Connection Calculus" in tex
    assert "Top-Level Categories" in tex
    for strategy in ["Baseline", "CFC", "Stem basis"]:
        assert strategy in tex, f"Missing strategy: {strategy}"
    assert "EVALUATION_RULES_FOL_IMPLICATION" in tex or "UNIVERSAL QUANTIFIER" in tex
    print(f"OK test_build_appendix_includes_all_prompts ({len(tex)} chars)")


def test_load_traces_empty_dir():
    """An empty traces directory returns an empty DataFrame."""
    with tempfile.TemporaryDirectory() as td:
        df = load_traces(td)
    assert df.empty
    print("OK test_load_traces_empty_dir")


def test_load_traces_with_files():
    """A directory with two valid traces loads them both."""
    with tempfile.TemporaryDirectory() as td:
        for i in range(2):
            payload = {
                "model": f"m{i}",
                "context_name": "integers",
                "prompt_mode": "basic",
                "question_mode": "implicational",
                "finished": True,
                "early_stop": False,
                "timeout": False,
                "wallclock_seconds": 12.3 + i,
                "n_iterations": 5,
                "n_retries_total": 1,
                "n_retry_exhausted": 0,
                "joint_entropy": 1.5,
                "mean_entropy": 0.5,
                "iteration_log": [],
            }
            with open(os.path.join(td, f"m{i}__integers__basic__implicational.json"), "w") as f:
                json.dump(payload, f)
        df = load_traces(td)
    assert len(df) == 2
    assert set(df["model"]) == {"m0", "m1"}
    assert abs(df["wallclock_seconds"].sum() - 25.6) < 1e-6
    print("OK test_load_traces_with_files")


def test_per_context_table_smoke():
    """A two-context synthetic DataFrame produces a non-empty tabular."""
    df = pd.DataFrame({
        "context_name": ["integers", "integers", "graphs", "graphs"],
        "n_iterations": [3, 5, 4, 6],
        "n_retries_total": [1, 0, 2, 1],
        "early_stop": [False, True, False, False],
        "joint_entropy": [1.2, 0.0, 2.0, 1.5],
    })
    tex = build_per_context_table(df)
    assert r"\begin{table}" in tex
    assert "Integer Numbers" in tex
    assert "Graph Theory" in tex
    print(f"OK test_per_context_table_smoke ({len(tex)} chars)")


def test_per_size_bin_table_without_size():
    """If the size column is missing, the per-bin table is the empty string."""
    df = pd.DataFrame({"context_name": ["integers"], "n_iterations": [1]})
    tex = build_per_size_bin_table(df)
    assert tex == ""
    print("OK test_per_size_bin_table_without_size")


if __name__ == "__main__":
    test_escape_tex_curly_and_percent()
    test_build_appendix_includes_all_prompts()
    test_load_traces_empty_dir()
    test_load_traces_with_files()
    test_per_context_table_smoke()
    test_per_size_bin_table_without_size()
    print("\nAll dump/analyze tests passed.")
