from typing import FrozenSet, Optional, Sequence, Set
from concepts import Context

# Context recorded by runs made before the fix in ``AttributeExplorerGraph.update_state``
# when the oracle accepted the first implication [] -> M.
PLACEHOLDER_ATTRIBUTES = ["None"]
ALL_ATTRIBUTES_OBJECT = "all_attributes"


def all_attributes_context(attributes: Sequence[str]) -> Context:
    """Context with one object that has every attribute; its lattice has the single intent M."""
    attributes = list(attributes)
    return Context([ALL_ATTRIBUTES_OBJECT], attributes, [[True] * len(attributes)])


def is_placeholder(attributes: Sequence[str]) -> bool:
    return list(attributes) == PLACEHOLDER_ATTRIBUTES


def accepted_first_implication(objects: Sequence[str], attributes: Sequence[str]) -> bool:
    """Whether a finished run ended by accepting [] -> M (old placeholder or fixed context)."""
    return is_placeholder(attributes) or list(objects) == [ALL_ATTRIBUTES_OBJECT]


def learned_context(objects, attributes, matrix,
                    domain_attributes: Optional[Sequence[str]] = None) -> Context:
    """Context learned by a finished run, as stored in ``results.csv`` or a trace.

    Runs that ended by accepting [] -> M were stored with a placeholder context
    (attribute ``None``). For those runs the context defined by the oracle's
    answers is returned instead: one object with every attribute of the domain.
    """
    if is_placeholder(attributes):
        if domain_attributes is None:
            raise ValueError("placeholder context: the domain attributes are needed")
        return all_attributes_context(domain_attributes)
    return Context(objects, attributes, matrix)


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
