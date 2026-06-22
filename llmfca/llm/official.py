"""
llmfca.llm.official — Thin wrapper around the official openai SDK.

Why this exists: the openai-python SDK is the canonical OpenAI-compatible
client. Wrapping it directly (instead of going through langchain-openai)
avoids the extra layer that historically mangles fields like max_tokens
(rename to max_completion_tokens) and extra_body. The openai SDK flattens
extra_body to the wire payload top-level, which is what OpenAI-compatible
proxies (LlamUS, vLLM, LM Studio) expect.

The wrapper exposes the minimal surface used by AttributeExplorerGraph:
  - .invoke(messages)  (sync, runs an event loop internally)
  - .ainvoke(messages) (async, returns a duck-typed AIMessage)
  - .response_metadata  (dict with 'logprobs.content', 'token_usage', 'finish_reason')

Validated empirically against LlamUS with max_tokens=2000 (Part 3 of
tests/test_openai_client_direct.py shows finish_reason="length" when the
cap is hit, and finish_reason="stop" when the model emits EOS naturally).
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


@dataclass
class SimpleAIMessage:
    """Duck-typed AIMessage.

    Mimics the langchain_core.messages.AIMessage interface that
    AttributeExplorerGraph depends on. We do NOT subclass AIMessage to
    avoid pulling in pydantic v1 compatibility shims.
    """

    content: str
    response_metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SimpleAIMessage(content={self.content[:60]!r}...)"

    def __bool__(self) -> bool:
        return bool(self.content)


def _to_chat_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Convert langchain BaseMessage (or dict) to OpenAI chat format."""
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            continue
        # Detect role by class name (avoid importing langchain_core here)
        cls = type(m).__name__
        if "System" in cls:
            role = "system"
        elif "Human" in cls or "User" in cls:
            role = "user"
        elif "AI" in cls or "Assistant" in cls:
            role = "assistant"
        else:
            role = "user"
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


def _extract_logprobs(choice) -> List[Dict[str, Any]]:
    """Convert openai Choice.logprobs.content → list[dict] for entropy_analyzer.

    Output schema (what analyze_uncertainty expects):
        [{"token": str, "top_logprobs": [{"token": str, "logprob": float}, ...]}]
    """
    out: List[Dict[str, Any]] = []
    if not getattr(choice, "logprobs", None):
        return out
    content = getattr(choice.logprobs, "content", None) or []
    for tok in content:
        top_k: List[Dict[str, Any]] = []
        for alt in (getattr(tok, "top_logprobs", None) or []):
            top_k.append(
                {"token": getattr(alt, "token", ""), "logprob": float(getattr(alt, "logprob", 0.0))}
            )
        out.append(
            {
                "token": getattr(tok, "token", ""),
                "logprob": float(getattr(tok, "logprob", 0.0)),
                "top_logprobs": top_k,
            }
        )
    return out


class OfficialOpenAIClient:
    """Async-first OpenAI-compatible chat client.

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
    max_tokens : int
        Hard cap on tokens per call. Empirically validated to be respected
        by LlamUS/Ollama (finish_reason="length" when the cap is hit).
        Default 2000 — gives 4x headroom over the ~500-token typical
        response and 3.5x faster worst-case wallclock than 7000.
    request_timeout : float
        Per-request timeout in seconds. None disables.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        top_logprobs: int = 20,
        max_tokens: int = 2000,
        logprobs: bool = True,
        request_timeout: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if request_timeout is not None:
            kwargs["timeout"] = request_timeout
        self._client = AsyncOpenAI(**kwargs)

    async def ainvoke(self, messages: List[Any]) -> SimpleAIMessage:
        chat_messages = _to_chat_messages(messages)
        create_kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            messages=chat_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.logprobs:
            create_kwargs["logprobs"] = True
            create_kwargs["top_logprobs"] = self.top_logprobs
        resp = await self._client.chat.completions.create(**create_kwargs)
        choice = resp.choices[0]
        usage: Dict[str, Any] = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
        return SimpleAIMessage(
            content=choice.message.content or "",
            response_metadata={
                "logprobs": {"content": _extract_logprobs(choice)},
                "token_usage": usage,
                "finish_reason": choice.finish_reason,
            },
        )

    def invoke(self, messages: List[Any]) -> SimpleAIMessage:
        """Sync wrapper: runs an event loop internally.

        The AttributeExplorerGraph is async-first (it lives inside a
        LangGraph state machine), so this is only used for diagnostics.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(
                self.ainvoke(messages), loop
            ).result()
        return loop.run_until_complete(self.ainvoke(messages))
