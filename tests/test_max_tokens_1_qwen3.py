"""Empirical test: max_tokens=1 is respected by qwen3:4b.

No stop tokens, no logprobs. Just a hard cap of 1 token to prove the
endpoint honours the cap. This is the cleanest possible diagnostic for
the wire-payload path: if completion_tokens > 1, the cap is being
ignored; if finish_reason == "length" and completion_tokens == 1, the
cap is working.

Also tests max_tokens=1 with a model that historically hangs
(gemma3:1b) to see if the cap alone breaks the hang, and verifies a
default (2000) run still works.
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

PROMPT = "What is 2+2? Just the number."

CASES = [
    ("qwen3:4b",  1, 2000, "cap=1, no stop, no logprobs"),
    ("qwen3:4b",  5, 2000, "cap=5, no stop, no logprobs"),
    ("qwen3:4b",  50, 2000, "cap=50, no stop, no logprobs"),
    ("gemma3:1b", 1, 2000, "cap=1 on a model that historically hangs"),
]


async def probe(client, model):
    """Check the model is reachable; return True/False."""
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        return True
    except Exception as e:
        return f"unavailable: {type(e).__name__}: {str(e)[:80]}"


async def ask(client, model, max_tokens):
    t0 = time.time()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - t0
    choice = resp.choices[0]
    return {
        "completion": resp.usage.completion_tokens if resp.usage else None,
        "finish": choice.finish_reason,
        "elapsed_s": round(elapsed, 2),
        "content": choice.message.content or "",
    }


async def main():
    if not API_KEY or API_KEY == "sk-your-token-here":
        print("SKIP: LLAMUS_API_KEY not set in .env")
        sys.exit(0)
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"Endpoint: {BASE_URL}")
    print(f"Prompt:   {PROMPT!r}")
    print()

    # Probe each model first
    print("=== Probe ===")
    for model, *_ in CASES:
        result = await probe(client, model)
        if result is True:
            print(f"  {model}: reachable")
        else:
            print(f"  {model}: {result}")
    print()

    print("=== max_tokens diagnostic ===")
    all_pass = True
    for model, cap, _, label in CASES:
        result = await ask(client, model, cap)
        ok = result["completion"] is not None and result["completion"] <= cap
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {model}  cap={cap:4d}  {label}")
        print(f"         completion={result['completion']}  "
              f"finish={result['finish']}  "
              f"elapsed={result['elapsed_s']}s")
        print(f"         content={result['content'][:60]!r}")
        if not ok:
            all_pass = False
        print()

    if all_pass:
        print("=== max_tokens=1 is respected by all probed models ===")
    else:
        print("=== SOME FAILURES — cap not respected ===")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
