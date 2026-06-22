"""
llmfca.llm — OpenAI-compatible chat model factory.

The default factory `make_model` builds an `OfficialOpenAIClient` (a thin
wrapper around the official openai Python SDK) pointed at the LlamUS
endpoint (or any other OpenAI-compatible provider) with logprobs enabled,
deterministic temperature, and a hard cap on tokens per call.
"""
from llmfca.llm.client import make_model
from llmfca.llm.official import OfficialOpenAIClient, SimpleAIMessage

__all__ = ["make_model", "OfficialOpenAIClient", "SimpleAIMessage"]
