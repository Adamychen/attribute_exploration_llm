"""
Tests unitarios para el módulo llmfca.analysis.metrics.

Cubre:
- S(ctx, ctx) == 1.0 (identidad reflexiva)
- S(ctx, ctx_swap_attrs) < 1.0 (diferencia detectable)
- S_H acotado en [0, 1]
- Edge cases: contextos vacíos, atributos disjuntos
"""
import sys
import os

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
sys.path.insert(0, ROOT)

from concepts import Context
from llmfca.analysis.metrics import (
    concept_intents,
    symmetric_difference,
    similarity,
    hausdorff_distance,
    normalized_hausdorff_similarity,
    compute_similarities,
)


def _ctx(objs, attrs, rels):
    return Context(objs, attrs, rels)


def test_similarity_identity():
    """S(ctx, ctx) debe ser 1.0 por identidad reflexiva."""
    ctx = _ctx(["a", "b", "c"], ["x", "y"], [[True, False], [True, True], [False, True]])
    S, S_H = compute_similarities(ctx, ctx)
    assert S == 1.0, f"Expected 1.0, got {S}"
    assert S_H == 1.0, f"Expected 1.0, got {S_H}"
    print("OK test_similarity_identity")


def test_similarity_disjoint_attrs():
    """Dos contextos con atributos disjuntos no pueden tener S=1."""
    c1 = _ctx(["a"], ["x"], [[True]])
    c2 = _ctx(["a"], ["y"], [[True]])
    S, S_H = compute_similarities(c1, c2)
    assert 0.0 <= S < 1.0
    assert 0.0 <= S_H <= 1.0
    print("OK test_similarity_disjoint_attrs")


def test_similarity_swapped_attr():
    """Dos contextos con objetos diferentes producen S < 1."""
    c1 = _ctx(["a", "b", "c"], ["x", "y"], [
        [True, False],
        [True, True],
        [False, True],
    ])
    c2 = _ctx(["a", "b", "d"], ["x", "y"], [
        [True, False],
        [True, True],
        [True, True],
    ])
    S, _ = compute_similarities(c1, c2)
    assert 0.0 < S < 1.0, f"Expected 0 < S < 1, got {S}"
    print("OK test_similarity_swapped_attr")


def test_similarity_empty_contexts():
    """Contextos vacíos: S=1 (definición del módulo)."""
    c1 = _ctx(["'"], ["None"], [["True"]])
    c2 = _ctx(["'"], ["None"], [["True"]])
    S, S_H = compute_similarities(c1, c2)
    assert S == 1.0
    print("OK test_similarity_empty_contexts")


def test_hausdorff_bounded():
    """S_H siempre en [0, 1]."""
    c1 = _ctx(["a", "b", "c", "d"], ["p", "q", "r"], [
        [True, False, False],
        [True, True, False],
        [True, True, True],
        [False, False, True],
    ])
    c2 = _ctx(["x", "y"], ["p", "q", "r"], [
        [True, False, True],
        [False, True, False],
    ])
    _, S_H = compute_similarities(c1, c2)
    assert 0.0 <= S_H <= 1.0, f"S_H out of bounds: {S_H}"
    print("OK test_hausdorff_bounded")


def test_concept_intents_unique():
    """concept_intents devuelve un set (intents únicos)."""
    c = _ctx(["a", "b"], ["x", "y"], [[True, False], [True, True]])
    intents = concept_intents(c)
    assert len(intents) == len(c.lattice)
    print("OK test_concept_intents_unique")


def test_symmetric_difference_sanity():
    """symmetric_difference es 0 para el mismo set."""
    c = _ctx(["a", "b"], ["x"], [[True], [False]])
    intents = concept_intents(c)
    assert symmetric_difference(intents, intents) == 0
    print("OK test_symmetric_difference_sanity")


if __name__ == "__main__":
    test_similarity_identity()
    test_similarity_disjoint_attrs()
    test_similarity_swapped_attr()
    test_similarity_empty_contexts()
    test_hausdorff_bounded()
    test_concept_intents_unique()
    test_symmetric_difference_sanity()
    print("\nAll metrics tests passed.")


def test_learned_context_replaces_placeholder():
    from llmfca.analysis.metrics import (learned_context, accepted_first_implication,
                                         ALL_ATTRIBUTES_OBJECT)
    ref = _ctx(["1", "2", "4"], ["even", "odd", "quadratic"],
               [[False, True, True], [True, False, False], [True, False, True]])
    ctx = learned_context(["'"], ["None"], [[True]], ref.properties)
    assert list(ctx.objects) == [ALL_ATTRIBUTES_OBJECT]
    assert concept_intents(ctx) == {frozenset(ref.properties)}
    assert accepted_first_implication(["'"], ["None"])
    assert accepted_first_implication(list(ctx.objects), list(ctx.properties))
    assert not accepted_first_implication(["1"], ["even", "odd", "quadratic"])
    # M is always an intent of the reference, so S = 2 / (|C_ref| + 1); the top
    # concept of the reference has an empty intent here, at distance |M| from M.
    n_ref = len(concept_intents(ref))
    S, S_H = compute_similarities(ctx, ref)
    assert abs(S - 2 / (n_ref + 1)) < 1e-12
    assert S_H == 0.0


def test_learned_context_keeps_regular_contexts():
    from llmfca.analysis.metrics import learned_context
    ctx = learned_context(["1"], ["even", "odd"], [[False, True]])
    assert list(ctx.objects) == ["1"]
