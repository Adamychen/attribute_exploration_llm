"""Tests for experiments.build_derived_data."""
import glob
import json
import os

import pytest

from experiments.build_derived_data import (
    RefContext,
    duquenne_guigues_basis,
    extract_conjectures,
    parse_implication,
    reference_bases_payload,
    run_id,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_DIR = os.path.join(ROOT, "experiments", "reference_contexts")
TRACES = os.path.join(ROOT, "experiments", "data", "traces")


def small_context() -> RefContext:
    # g1 {a,b}   g2 {a,b,c}   g3 {c}
    return RefContext.from_matrix(
        ["g1", "g2", "g3"], ["a", "b", "c"],
        [[1, 1, 0], [1, 1, 1], [0, 0, 1]],
    )


def test_support_holds_and_counterexamples():
    ctx = small_context()
    a, b, c = ctx.mask(["a"]), ctx.mask(["b"]), ctx.mask(["c"])
    assert ctx.extent_count(a) == 2
    assert ctx.extent_count(0) == 3                 # empty premise: every object
    assert ctx.extent_count(a | c) == 1
    assert ctx.holds(a, b) and not ctx.holds(a, c)
    assert ctx.n_counterexamples(a, c) == 1
    assert ctx.closure(a) == a | b


def test_duquenne_guigues_basis_small():
    ctx = small_context()
    basis = {(frozenset(ctx.names(p)), frozenset(ctx.names(c))) for p, c in duquenne_guigues_basis(ctx)}
    # pseudo-intents: {a} -> {a,b}, {b} -> {a,b}; {a,c},{b,c} close to the full set
    assert (frozenset("a"), frozenset("ab")) in basis
    assert (frozenset("b"), frozenset("ab")) in basis
    for premise, conclusion in basis:
        assert premise < conclusion


@pytest.mark.parametrize("name,n_impl", [("integers", 4), ("graphs", 8), ("rcc", 23)])
def test_reference_basis_sizes_match_exhaustive_search(name, n_impl):
    """The NextClosure basis has the same premises as the explorer's exhaustive search."""
    pytest.importorskip("concepts")
    from concepts import Context
    from llmfca.attribute_explorer import AttributeExplorerGraph

    path = os.path.join(REF_DIR, f"{name}.json")
    ref = RefContext.from_json(path)
    mine = {frozenset(ref.names(p)) for p, _ in duquenne_guigues_basis(ref)}
    assert len(mine) == n_impl

    ctx = Context.fromjson(path)
    explorer = AttributeExplorerGraph(None, None, "", list(ctx.properties), None)
    basis = []
    while True:
        imp = explorer.next_implication(ctx, basis)
        if imp is None:
            break
        basis.append(imp)
    assert mine == {frozenset(b.antecedent) for b in basis}


def test_integers_basis_has_paper_supports():
    payload = reference_bases_payload({"integers": RefContext.from_json(os.path.join(REF_DIR, "integers.json"))})
    by_premise = {frozenset(i["premise"]): i for i in payload["integers"]["implications"]}
    assert by_premise[frozenset({"even", "quadratic"})]["support_count"] == 1
    assert by_premise[frozenset({"even", "odd"})]["support_count"] == 0
    assert by_premise[frozenset({"prime", "compound"})]["support_count"] == 0
    assert by_premise[frozenset({"quadratic", "prime"})]["support_count"] == 0
    assert payload["integers"]["n_objects"] == 7


def test_parse_implication():
    assert parse_implication("[] => ['a', 'b']") == ([], ["a", "b"])
    assert parse_implication("['even', 'odd'] => ['prime']") == (["even", "odd"], ["prime"])
    assert parse_implication("[\"2'\"] => ['x']") == (["2'"], ["x"])
    with pytest.raises(ValueError):
        parse_implication("not an implication")


def test_run_id_is_filesystem_safe():
    assert run_id("gemma3:1b", "graphs", "basic", "existential") == "gemma3_1b__graphs__basic__existential"


def test_extract_conjectures_three_outcomes():
    log = [
        {"step": 1, "stage": "initialization", "implication": "[] => ['a', 'b']",
         "event": "find_next_implication", "finished": False},
        {"step": 1, "event": "ask_llm", "joint_entropy": 0.1, "mean_entropy": 0.2, "retry_attempt": 0},
        {"step": 1, "event": "validate_logic", "decision": "accept"},
        {"step": 1, "event": "update_state", "decision": "add_counterexample", "object": "o1",
         "attribute_vector": [True, False]},
        {"step": 2, "stage": "exploration", "implication": "['a'] => ['b']",
         "event": "find_next_implication", "finished": False},
        {"step": 2, "event": "ask_llm", "retry_attempt": 0},
        {"step": 2, "event": "update_state", "decision": "accept", "implication": "['a'] => ['b']"},
        {"step": 3, "stage": "exploration", "implication": "['b'] => ['a']",
         "event": "find_next_implication", "finished": False},
        {"step": 3, "event": "ask_llm", "retry_attempt": 0},
        {"step": 3, "event": "validate_logic", "active_flags": ["invalid_format_answer"],
         "retries_used": 1, "max_retries": 2},
        {"step": 3, "event": "ask_llm", "retry_attempt": 1},
        {"step": 3, "event": "validate_logic", "active_flags": ["invalid_format_answer"],
         "retries_used": 3, "max_retries": 2},
        {"step": 4, "stage": "exploration", "implication": None,
         "event": "find_next_implication", "finished": True},
    ]
    recs = extract_conjectures(log)
    assert [r["step"] for r in recs] == [1, 2, 3]            # terminal step is not a conjecture
    assert [r["outcome"] for r in recs] == ["refuted", "accepted", "unresolved"]
    assert recs[0]["counterexample_object"] == "o1"
    assert recs[2]["retry_exhausted"] is True and recs[2]["n_retries"] == 3
    assert recs[2]["n_llm_calls"] == 2


@pytest.mark.skipif(not glob.glob(os.path.join(TRACES, "*.json")), reason="committed traces not present")
def test_accepted_conjectures_match_stem_basis_in_every_trace():
    mismatches = []
    for path in glob.glob(os.path.join(TRACES, "*.json")):
        with open(path, encoding="utf-8") as f:
            trace = json.load(f)
        recs = extract_conjectures(trace["iteration_log"])
        accepted = sum(r["outcome"] == "accepted" and r["stage"] != "initialization" for r in recs)
        if accepted != len(trace["stem_basis"]):
            mismatches.append(os.path.basename(path))
    assert not mismatches, mismatches[:5]
