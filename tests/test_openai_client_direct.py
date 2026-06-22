"""Verify max_tokens is respected using the official openai SDK directly.

Bypasses langchain-openai entirely. The openai-python SDK is the canonical
client; if max_tokens works here, langchain's wrappers were the issue.

We test BOTH a fence and a non-fence prompt to expose the pathological case
where a model emits ``` first and gets cut off by stop tokens.
"""
import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

API_KEY = os.getenv("LLAMUS_API_KEY", "")
BASE_URL = "https://llamus.cs.us.es/ollama/v1"
MODEL = "llama3.2:3b"


def skip_if_no_key():
    if not API_KEY or API_KEY == "sk-your-token-here":
        print("SKIP: LLAMUS_API_KEY not set in .env")
        sys.exit(0)


async def ask(client: AsyncOpenAI, *, max_tokens: int, prompt: str, stop=None):
    """Single completion call, return (elapsed, content, finish_reason, usage)."""
    kwargs = dict(model=MODEL, max_tokens=max_tokens, temperature=0.0)
    if stop:
        kwargs["stop"] = list(stop)
    t0 = time.time()
    resp = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    elapsed = time.time() - t0
    choice = resp.choices[0]
    return {
        "elapsed_s": round(elapsed, 2),
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
    }


PROMPT_FENCE = (
    "Output a JSON object that lists the integers from 1 to 100, with one "
    "key per integer. Use the format ```json { ... } ```. Do not include any "
    "explanation or commentary outside the code block."
)

PROMPT_NO_FENCE = (
    "Output a JSON object that lists the integers from 1 to 100, with one "
    "key per integer. Do not use markdown code fences. Do not include any "
    "explanation or commentary before or after the JSON. The response should "
    "be ONLY the JSON object, nothing else, no prose."
)

PROMPT_LONGER = (
    "Write a single JSON object with 200 keys named k1, k2, ..., k200. Each "
    "key maps to a 50-word description of what the key might mean in a "
    "fictional inventory system. Do not use markdown code fences. The JSON "
    "must be valid and parseable. No commentary outside the JSON."
)


async def main():
    skip_if_no_key()
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"Endpoint: {BASE_URL}")
    print(f"Model:    {MODEL}")
    print(f"SDK:      openai==2.15.0 (official, not langchain)")
    print()

    print("#" * 60)
    print("# PART 1: PROMPT WITH ``` FENCE")
    print("#" * 60)
    cases = [
        ("max=2000, stop=```", 2000, ["```"]),
        ("max=10,   stop=```", 10, ["```"]),
        ("max=2000, stop=()", 2000, []),
        ("max=7000, stop=()", 7000, []),
    ]
    for label, max_t, stop in cases:
        r = await ask(client, max_tokens=max_t, prompt=PROMPT_FENCE, stop=stop)
        print(f"  {label}")
        print(f"    elapsed={r['elapsed_s']}s  completion={r['completion_tokens']}  "
              f"finish={r['finish_reason']}")
        print(f"    preview={r['content'][:80]!r}")
        print()

    print("#" * 60)
    print("# PART 2: PROMPT WITHOUT ``` (forces longer generation)")
    print("#" * 60)
    for label, max_t, stop in cases:
        r = await ask(client, max_tokens=max_t, prompt=PROMPT_NO_FENCE, stop=stop)
        print(f"  {label}")
        print(f"    elapsed={r['elapsed_s']}s  completion={r['completion_tokens']}  "
              f"finish={r['finish_reason']}")
        print(f"    preview={r['content'][:80]!r}")
        print()

    print("#" * 60)
    print("# PART 3: PATHOLOGICAL LONG (200 keys × 50 words)")
    print("#" * 60)
    for label, max_t, stop in cases:
        r = await ask(client, max_tokens=max_t, prompt=PROMPT_LONGER, stop=stop)
        print(f"  {label}")
        print(f"    elapsed={r['elapsed_s']}s  completion={r['completion_tokens']}  "
              f"finish={r['finish_reason']}")
        print(f"    preview={r['content'][:80]!r}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
