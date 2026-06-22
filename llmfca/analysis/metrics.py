from typing import Set, FrozenSet
from concepts import Context


def concept_intents(ctx: Context) -> Set[FrozenSet[int]]:
    return {frozenset(c.intent) for c in ctx.lattice}


def symmetric_difference(A: Set[FrozenSet[int]], B: Set[FrozenSet[int]]) -> int:
    return len((A - B) | (B - A))


def similarity(A: Set[FrozenSet[int]], B: Set[FrozenSet[int]]) -> float:
    if not A and not B:
        return 1.0
    diff = symmetric_difference(A, B)
    return 1 - diff / (len(A) + len(B))


def hausdorff_distance(A: Set[FrozenSet[int]], B: Set[FrozenSet[int]]) -> float:
    if not A or not B:
        return float('inf')
    max_min_dist = 0.0
    for a in A:
        min_dist = min(len(a ^ b) for b in B)
        if min_dist > max_min_dist:
            max_min_dist = min_dist
    for b in B:
        min_dist = min(len(a ^ b) for a in A)
        if min_dist > max_min_dist:
            max_min_dist = min_dist
    return max_min_dist


def normalized_hausdorff_similarity(A: Set[FrozenSet[int]], B: Set[FrozenSet[int]], num_attributes: int) -> float:
    if not A and not B:
        return 1.0
    d = hausdorff_distance(A, B)
    if d == float('inf'):
        return 0.0
    return 1 - d / num_attributes if num_attributes > 0 else 0.0


def compute_similarities(ctx_llm, ctx_ref):
    intents_llm = concept_intents(ctx_llm)
    intents_ref = concept_intents(ctx_ref)
    S = similarity(intents_llm, intents_ref)
    all_attrs = set(ctx_llm.properties) | set(ctx_ref.properties)
    S_H = normalized_hausdorff_similarity(intents_llm, intents_ref, len(all_attrs))
    return S, S_H
