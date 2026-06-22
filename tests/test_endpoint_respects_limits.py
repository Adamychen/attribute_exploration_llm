"""Empirically verify the LlamUS endpoint actually respects max_tokens and stop.

This test hits the REAL endpoint (not a mock) and measures the response
length under different control configurations. Compares:

  A. max_tokens=2000, stop=("```",)        — current factory defaults
  B. max_tokens=10,  stop=("```",)        — hard cap of 10 tokens
  C. max_tokens=2000, stop=()              — no stop, default cap
  D. max_tokens=7000, stop=()              — old factory default (no stop)

If the endpoint respects max_tokens, the response token count in (B) should
be <= 10. If the endpoint also respects stop, the response in (A) should
end with the stop token (or just before it, depending on OpenAI semantics).

NOTE: requires LLAMUS_API_KEY in .env. Skipped if not present.
"""
import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from llmfca.llm.client import make_model

load_dotenv()

API_KEY = os.getenv("LLAMUS_API_KEY", "")
BASE_URL = "https://llamus.cs.us.es/ollama/v1"
MODEL = "llama3.2:3b"  # fastest available, lowest variance


def skip_if_no_key():
    if not API_KEY or API_KEY == "sk-your-token-here":
        print("SKIP: LLAMUS_API_KEY not set in .env")
        sys.exit(0)


def count_tokens_rough(text: str) -> int:
    """Cheap approximation: split on whitespace.

    Real tokenizers differ, but for this diagnostic we just need an order
    of magnitude. A real tokenizer would be 1-1.3x this number.
    """
    return len(text.split())


async def measure(name: str, *, max_tokens: int, stop: tuple[str, ...], prompt: str):
    """Run one experiment, return (elapsed, response_text, finish_reason)."""
    m = ChatOpenAI(
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
        extra_body={"max_tokens": max_tokens, "stop": list(stop)},
    )
    t0 = time.time()
    resp = await m.ainvoke(prompt)
    elapsed = time.time() - t0
    text = resp.content if hasattr(resp, "content") else str(resp)
    finish = resp.response_metadata.get("finish_reason", "?")
    usage = resp.response_metadata.get("token_usage", {})
    completion_tokens = usage.get("completion_tokens", "?")
    return {
        "name": name,
        "max_tokens": max_tokens,
        "stop": list(stop),
        "elapsed_s": round(elapsed, 2),
        "rough_tokens": count_tokens_rough(text),
        "completion_tokens": completion_tokens,
        "finish_reason": finish,
        "ends_with_stop": text.rstrip().endswith(list(stop)[0]) if stop else None,
        "preview": text[:120].replace("\n", " "),
    }


PROMPT_WITH_FENCE = (
    "Output a JSON object that lists the integers from 1 to 100, with one "
    "key per integer. Use the format ```json { ... } ```. Do not include any "
    "explanation or commentary outside the code block."
)

PROMPT_NO_FENCE = (
    "Output a JSON object that lists the integers from 1 to 100, with one "
    "key per integer. Do not use markdown code fences. Do not include any "
    "explanation or commentary before or after the JSON."
)


async def main():
    skip_if_no_key()
    print(f"Endpoint: {BASE_URL}")
    print(f"Model:    {MODEL}")
    print()

    print("#" * 60)
    print("# PART 1: prompt WITH ``` fence (model emits ``` first)")
    print("#" * 60)
    print(f"Prompt: {PROMPT_WITH_FENCE[:80]}...")
    print()

    cases = [
        ("A: defaults (max=2000, stop=```)", 2000, ("```",)),
        ("B: hard cap (max=10,  stop=```)",    10, ("```",)),
        ("C: no stop  (max=2000, stop=())",  2000, ()),
        ("D: old cfg  (max=7000, stop=())",  7000, ()),
    ]

    for name, max_t, stop in cases:
        r = await measure(name, max_tokens=max_t, stop=stop, prompt=PROMPT_WITH_FENCE)
        print(f"--- {r['name']} ---")
        print(f"  elapsed:           {r['elapsed_s']}s")
        print(f"  completion_tokens: {r['completion_tokens']}")
        print(f"  finish_reason:     {r['finish_reason']}")
        print(f"  preview:           {r['preview']!r}")
        print()

    print("#" * 60)
    print("# PART 2: prompt WITHOUT ``` (forces pathological long gen)")
    print("#" * 60)
    print(f"Prompt: {PROMPT_NO_FENCE[:80]}...")
    print()

    for name, max_t, stop in cases:
        r = await measure(name, max_tokens=max_t, stop=stop, prompt=PROMPT_NO_FENCE)
        print(f"--- {r['name']} ---")
        print(f"  elapsed:           {r['elapsed_s']}s")
        print(f"  completion_tokens: {r['completion_tokens']}")
        print(f"  finish_reason:     {r['finish_reason']}")
        print(f"  preview:           {r['preview']!r}")
        print()

    print("=" * 60)
    print("DIAGNOSTIC")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
