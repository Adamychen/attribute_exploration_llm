"""Tests for the prompt-sensitivity analysis."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from experiments.analyze_sensitivity import (CELL, holm, load_all, paired, sign_test,
                                             signed_rank, analyse, table_finishing,
                                             table_similarity)


def test_sign_test_counts_discordant_pairs():
    ref = pd.Series([True, True, False, False, True])
    cmp = pd.Series([True, False, True, True, True])
    r = sign_test(ref, cmp)
    assert (r["lost"], r["gained"], r["k_ref"], r["k_cmp"]) == (1, 2, 3, 4)
    assert r["agree"] == pytest.approx(0.4)
    assert r["delta"] == pytest.approx(0.2)
    assert r["p"] == pytest.approx(1.0)


def test_sign_test_without_discordant_pairs():
    s = pd.Series([True, False])
    assert sign_test(s, s)["p"] == 1.0


def test_signed_rank_drops_zero_differences():
    r = signed_rank(pd.Series([0.5, 0.5, 0.75, 0.5]), pd.Series([0.5, 1.0, 0.5, 0.75]))
    assert r["n"] == 4 and r["n_equal"] == 1
    assert r["mean_abs_delta"] == pytest.approx(0.25)
    # d = 0, 0.5, -0.25, 0.25; ranks of |d| = 3, 1.5, 1.5; W+ = 4.5, W- = 1.5
    assert r["r_rb"] == pytest.approx(0.5)


def test_signed_rank_all_equal():
    r = signed_rank(pd.Series([1.0, 0.5]), pd.Series([1.0, 0.5]))
    assert math.isnan(r["p"]) and r["mean_delta"] == 0


def test_holm():
    assert holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    out = holm([0.02, float("nan")])
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1])


def test_paired_rejects_missing_cells():
    runs = pd.DataFrame([
        {"model": "m", "context_name": "c", "strategy": "basic", "question_mode": "q",
         "condition": "C0", "finished": True, "nontrivial": True, "S": 1.0, "S_H": 1.0},
    ])
    cells = pd.DataFrame([{"model": "m", "context_name": "c", "strategy": "basic",
                           "question_mode": "q"}])
    with pytest.raises(ValueError):
        paired(runs, "C0", "V3_wording", cells)


@pytest.fixture(scope="module")
def runs():
    return load_all()


def test_committed_runs_are_complete(runs):
    counts = runs.condition.value_counts().to_dict()
    assert counts == {"main1": 576, "main2": 576, "C0": 118, "V1_order": 56,
                      "V2_format": 56, "V3_wording": 84, "V4_attrs": 84}
    assert not runs.duplicated(CELL + ["condition"]).any()


def test_first_execution_scores_match_run_summary(runs):
    from experiments.analyze_sensitivity import DATA
    import os
    summary = pd.read_csv(os.path.join(DATA, "derived", "run_summary.csv")).rename(
        columns={"prompt_mode": "strategy"})
    m = runs[runs.condition == "main1"].merge(summary, on=CELL, suffixes=("", "_rs"))
    assert len(m) == 576
    assert (m.S.fillna(-1) - m.S_rs.fillna(-1)).abs().max() < 1e-9
    assert (m.S_H.fillna(-1) - m.S_H_rs.fillna(-1)).abs().max() < 1e-9


def test_tables(runs):
    res = analyse(runs)
    assert res["variants"]["V1_order"]["finished"]["n"] == 56
    assert res["variants"]["V3_wording"]["finished"]["n"] == 84
    assert res["repair"][("main1", "C0")]["finished"]["n"] == 48
    finish, sim = table_finishing(res), table_similarity(res)
    assert r"\label{tab:sensitivity_finish}" in finish
    assert r"\label{tab:sensitivity_similarity}" in sim
    assert finish.count(r" \\") == sim.count(r" \\")
