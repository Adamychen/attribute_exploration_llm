"""Tests for the Results/Discussion tables generated from the main sweep."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.discussion_tables import (fig_tradeoff_scatter, load_executions, summary,
                                           table_cfc_gains, table_dg_graphs,
                                           table_query_asymmetry, table_replication)


def test_summary_counts():
    runs = pd.DataFrame({"finished": [True, True, False, True],
                         "nontrivial": [True, False, False, True],
                         "S": [0.9, 0.125, None, 0.5], "S_H": [0.8, 0.0, None, 0.4]})
    r = summary(runs)
    assert (r["n"], r["k"], r["accepted_first"]) == (4, 3, 1)
    assert r["rate"] == pytest.approx(0.75)
    assert r["S"] == pytest.approx((0.9 + 0.125 + 0.5) / 3)
    assert r["S_nt"] == pytest.approx(0.7)


@pytest.fixture(scope="module")
def ex():
    return load_executions()


def test_rates_match_aggregated_table(ex):
    # Graphs, existential: 8, 6 and 16 of 24 models finish (Table tab:aggregated_stats).
    first = ex["first"]
    g = first[(first.context_name == "graphs") & (first.question_mode == "existential")]
    assert [int(g[g.strategy == s].finished.sum()) for s in ("basic", "full_context", "stem_basis")] == [8, 6, 16]
    assert "0.33 & 0.52 & 0.58" in table_dg_graphs(first)


def test_kr_baseline_finishes_all_accept_first_implication(ex):
    first = ex["first"]
    kr = first[(first.context_name == "top") & (first.strategy == "basic") & first.finished]
    assert len(kr) == 7 and not kr.nontrivial.any()


def test_outputs_have_labels(ex):
    first = ex["first"]
    for text, label in ((table_replication(ex), "tab:replication"),
                        (table_query_asymmetry(first), "tab:query_asymmetry"),
                        (table_cfc_gains(first), "tab:cfc_gains"),
                        (table_dg_graphs(first), "tab:stem_graphs"),
                        (fig_tradeoff_scatter(first), "fig:tradeoff_scatter")):
        assert f"\\label{{{label}}}" in text
    assert "(control)" in table_cfc_gains(first)
