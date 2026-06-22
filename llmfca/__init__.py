"""
llmfca — Library for Attribute Exploration over LLMs.

Public API:
    AttributeExplorerGraph  — LangGraph state machine for AE (llmfca.attribute_explorer)
    Implication             — Candidate attribute implication (llmfca.attribute_explorer)
    EntropyAnalyzer         — Token-level entropy over LLM logprobs (llmfca.entropy_analyzer)
    make_model              — OfficialOpenAIClient factory for OpenAI-compatible endpoints (llmfca.llm.client)
    context_to_dict         — Serialize a concepts.Context to a dict (llmfca.fca_helpers)
    get_message_function    — FOL prompt formatter for an Implication (llmfca.fca_helpers)
    compute_similarities    — S, S_H metrics vs a reference context (llmfca.analysis.metrics)
    Prompts / parsers       — Per-domain prompts and Pydantic parsers (llmfca.prompting.prompts)
"""
from llmfca.attribute_explorer import AttributeExplorerGraph, Implication, ExplorationState
from llmfca.entropy_analyzer import EntropyAnalyzer
from llmfca.fca_helpers import context_to_dict, get_message_function
from llmfca.llm.client import make_model
from llmfca.analysis.metrics import (
    concept_intents,
    symmetric_difference,
    similarity,
    hausdorff_distance,
    normalized_hausdorff_similarity,
    compute_similarities,
)

__all__ = [
    "AttributeExplorerGraph",
    "Implication",
    "ExplorationState",
    "EntropyAnalyzer",
    "make_model",
    "context_to_dict",
    "get_message_function",
    "concept_intents",
    "symmetric_difference",
    "similarity",
    "hausdorff_distance",
    "normalized_hausdorff_similarity",
    "compute_similarities",
]

__version__ = "0.1.0"
