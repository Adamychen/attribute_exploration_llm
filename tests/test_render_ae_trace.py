"""Tests for the worked Attribute Exploration trace (Table 1 of the paper)."""
from __future__ import annotations

from experiments.render_ae_trace import integer_attributes, run_trace, to_latex

EXPECTED = [
    # premise, conclusion, accepted, counterexample, support (of 7)
    ([], ["even", "odd", "prime", "compound", "quadratic"], False, 25, 0),
    ([], ["odd", "compound", "quadratic"], False, 2, 1),
    (["even"], ["prime"], False, 6, 1),
    (["odd"], ["compound", "quadratic"], False, 3, 1),
    (["quadratic"], ["odd", "compound"], False, 1, 1),
    (["quadratic"], ["odd"], False, 4, 2),
    (["even", "odd"], ["prime", "compound", "quadratic"], True, None, 0),
    (["even", "quadratic"], ["compound"], True, None, 1),
    (["odd", "compound"], ["quadratic"], False, 15, 1),
    (["prime", "compound"], ["even", "odd", "quadratic"], True, None, 0),
    (["prime", "quadratic"], ["even", "odd", "compound"], True, None, 0),
]


def test_integer_attributes():
    assert integer_attributes(1) == {"even": False, "odd": True, "prime": False,
                                     "compound": False, "quadratic": True}
    assert integer_attributes(4)["compound"] and integer_attributes(4)["quadratic"]
    assert integer_attributes(2)["prime"] and not integer_attributes(9)["prime"]


def test_trace_matches_worked_example():
    steps, objects = run_trace()
    assert objects == [25, 2, 6, 3, 1, 4, 15]
    got = [(s.premise, s.conclusion, s.accepted, s.counterexample, s.support) for s in steps]
    assert got == EXPECTED


def test_accepted_implications_are_valid_in_final_context():
    steps, objects = run_trace()
    for s in steps:
        if s.accepted:
            holders = [o for o in objects if all(integer_attributes(o)[m] for m in s.premise)]
            assert all(all(integer_attributes(o)[m] for m in s.conclusion) for o in holders)


def test_latex_has_support_column_and_label():
    steps, objects = run_trace()
    tex = to_latex(steps, objects)
    assert r"\label{tab:ae_trace_example}" in tex
    assert "$2/7$" in tex
    assert tex.count(r"\textsc{yes}") == 4
