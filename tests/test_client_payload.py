"""Verify OfficialOpenAIClient produces responses in the schema the AE expects.

Uses a fake AsyncOpenAI to capture the request kwargs and return a controlled
ChatCompletion response. Checks:
  - max_tokens appears in the request kwargs (this is what the AE bound
    depends on — verified empirically with tests/test_openai_client_direct.py
    that the endpoint respects it)
  - logprobs=True and top_logprobs=20 in the request
  - The SimpleAIMessage exposes response_metadata['logprobs']['content']
    in the dict shape EntropyAnalyzer.analyze_uncertainty expects
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv

from llmfca.entropy_analyzer import EntropyAnalyzer
from llmfca.llm.official import OfficialOpenAIClient, _extract_logprobs

load_dotenv()


FAKE_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "test:1b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"response": "yes"}'},
            "logprobs": {
                "content": [
                    {
                        "token": '{"',
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": '{"', "logprob": -0.1},
                            {"token": "{", "logprob": -1.5},
                        ],
                    },
                    {
                        "token": '"response"',
                        "logprob": -0.2,
                        "top_logprobs": [
                            {"token": '"response"', "logprob": -0.2},
                            {"token": '"object"', "logprob": -2.0},
                        ],
                    },
                ]
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


def _make_mock_client(monkeypatch_target):
    """Build a mock openai.AsyncOpenAI whose chat.completions.create
    records kwargs and returns FAKE_COMPLETION."""
    mock_create = AsyncMock(return_value=_to_response(FAKE_COMPLETION))
    mock_chat = MagicMock()
    mock_chat.completions.create = mock_create
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client, mock_create


def _to_response(d):
    """Convert a dict to a ChatCompletion-shaped object via attribute access.

    The openai SDK returns Pydantic models; for our purposes we only read
    attributes. A simple namespace with __dict__ works.
    """
    from types import SimpleNamespace

    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [dict_to_ns(x) for x in d]
        return d

    return dict_to_ns(d)


def test_official_client_request_kwargs():
    """The official client must pass max_tokens, logprobs, top_logprobs."""
    mock_client, mock_create = _make_mock_client(None)
    with patch("llmfca.llm.official.AsyncOpenAI", return_value=mock_client):
        c = OfficialOpenAIClient("test:1b", "https://e/v1", "sk-test", max_tokens=2000)
        asyncio.run(c.ainvoke([{"role": "user", "content": "hi"}]))

    assert mock_create.await_count == 1
    kwargs = mock_create.await_args.kwargs
    print("=== Request kwargs sent to endpoint ===")
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if k == "messages":
            print(f"  {k}: [{len(v)} messages]")
        else:
            print(f"  {k}: {v}")
    print()

    assert kwargs["max_tokens"] == 2000
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 20
    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == "test:1b"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    print("OK request kwargs are correct")


def test_official_client_response_shape():
    """The SimpleAIMessage must expose response_metadata['logprobs']['content']
    in the dict shape EntropyAnalyzer.analyze_uncertainty expects."""
    mock_client, mock_create = _make_mock_client(None)
    with patch("llmfca.llm.official.AsyncOpenAI", return_value=mock_client):
        c = OfficialOpenAIClient("test:1b", "https://e/v1", "sk-test")
        msg = asyncio.run(c.ainvoke([{"role": "user", "content": "hi"}]))

    print("=== Response shape ===")
    print(f"  content: {msg.content!r}")
    print(f"  finish_reason: {msg.response_metadata.get('finish_reason')}")
    print(f"  usage: {msg.response_metadata.get('token_usage')}")
    print(f"  logprobs.content[0]: {msg.response_metadata['logprobs']['content'][0]}")
    print()

    assert msg.content == '{"response": "yes"}'
    assert msg.response_metadata["finish_reason"] == "stop"
    assert msg.response_metadata["token_usage"]["completion_tokens"] == 2

    logprobs_content = msg.response_metadata["logprobs"]["content"]
    assert isinstance(logprobs_content, list)
    assert len(logprobs_content) == 2
    for tok in logprobs_content:
        assert "token" in tok
        assert "top_logprobs" in tok
        for alt in tok["top_logprobs"]:
            assert "token" in alt
            assert "logprob" in alt

    # End-to-end: feed the logprobs through EntropyAnalyzer
    analyzer = EntropyAnalyzer(n_op=10)
    joint, mean = analyzer.analyze_uncertainty(logprobs_content)
    print(f"  EntropyAnalyzer → joint={joint:.3f}, mean={mean:.3f}")
    assert joint >= 0.0
    assert mean >= 0.0
    print("OK response shape is compatible with EntropyAnalyzer")


def test_extract_logprobs_empty():
    """No logprobs in the response → empty list (not error)."""
    resp = _to_response(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "{}"},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ]
        }
    )
    out = _extract_logprobs(resp.choices[0])
    assert out == []
    print("OK empty logprobs handled gracefully")


def test_official_client_logprobs_disabled():
    """When logprobs=False, the request omits logprobs/top_logprobs keys.

    Needed by run_benchmarks.py and any non-AE consumer.
    """
    mock_client, mock_create = _make_mock_client(None)
    with patch("llmfca.llm.official.AsyncOpenAI", return_value=mock_client):
        c = OfficialOpenAIClient(
            "test:1b", "https://e/v1", "sk-test",
            logprobs=False, top_logprobs=20, max_tokens=512,
        )
        asyncio.run(c.ainvoke([{"role": "user", "content": "hi"}]))

    kwargs = mock_create.await_args.kwargs
    print("=== Request kwargs with logprobs=False ===")
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if k == "messages":
            print(f"  {k}: [{len(v)} messages]")
        else:
            print(f"  {k}: {v}")
    print()

    assert "logprobs" not in kwargs, (
        f"logprobs must not appear when logprobs=False, got: {kwargs.get('logprobs')!r}"
    )
    assert "top_logprobs" not in kwargs, (
        f"top_logprobs must not appear when logprobs=False, got: {kwargs.get('top_logprobs')!r}"
    )
    assert kwargs["max_tokens"] == 512
    print("OK logprobs disabled → request omits logprobs/top_logprobs")


def test_make_model_forwards_logprobs_false():
    """make_model must forward logprobs=False to OfficialOpenAIClient
    (and run_benchmarks.py relies on this)."""
    from llmfca.llm.client import make_model
    m = make_model("test:1b", "https://e/v1", "sk-test", logprobs=False)
    assert m.logprobs is False
    print("OK make_model forwards logprobs=False")


if __name__ == "__main__":
    test_official_client_request_kwargs()
    test_official_client_response_shape()
    test_extract_logprobs_empty()
    test_official_client_logprobs_disabled()
    test_make_model_forwards_logprobs_false()
    print("All official-client tests passed.")
