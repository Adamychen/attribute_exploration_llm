"""
llmfca.llm.client — Factory for OpenAI-compatible chat models.

Default: OfficialOpenAIClient (a thin wrapper around the official openai
Python SDK). Compatible with Ollama, vLLM, LlamUS (the openai-compatible
endpoint at https://llamus.cs.us.es/ollama/v1) and any vendor exposing the
OpenAI chat completions API.

The wrapper exposes the minimal surface used by AttributeExplorerGraph:
  - .invoke(messages)  (sync)
  - .ainvoke(messages) (async, returns a SimpleAIMessage with
                        .content and .response_metadata['logprobs'])
"""
from typing import Optional

from llmfca.llm.official import OfficialOpenAIClient


def make_model(
    model_name: str,
    base_url: str,
    api_key: str,
    *,
    temperature: float = 0.0,
    top_logprobs: int = 20,
    max_tokens: int = 2000,
    logprobs: bool = True,
    request_timeout: Optional[float] = None,
) -> OfficialOpenAIClient:
    """Build an OfficialOpenAIClient with logprobs enabled by default.

    Parameters
    ----------
    model_name : str
        The model id as exposed by the endpoint (e.g. "llama3.2:3b").
    base_url : str
        OpenAI-compatible base URL (omit the trailing /chat/completions).
    api_key : str
        Bearer token. Many local endpoints accept any non-empty string.
    temperature : float
        Sampling temperature. Default 0.0 for determinism in AE.
    top_logprobs : int
        Number of top alternatives to return per token. 20 is the LlamUS cap.
        Ignored when ``logprobs=False``.
    max_tokens : int
        Hard cap on tokens per call. Empirically validated to be respected
        by LlamUS/Ollama (finish_reason="length" when the cap is hit).
        Default 2000 — gives 4x headroom over the ~500-token typical
        response and 3.5x faster worst-case wallclock than 7000.
    logprobs : bool
        If True (default), request token-level logprobs from the endpoint and
        populate ``response.response_metadata['logprobs']['content']`` — the
        signal consumed by ``EntropyAnalyzer``. Set to False for non-AE
        consumers (e.g. MMLU/GSM8K benchmarks) that don't need logprobs
        and may hit endpoints that don't support them.
    request_timeout : float
        Per-request timeout in seconds. None disables (httpx default ~5min
        connect, ~10min read). Recommended: 180 for production sweeps.
    """
    return OfficialOpenAIClient(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens,
        logprobs=logprobs,
        request_timeout=request_timeout,
    )
