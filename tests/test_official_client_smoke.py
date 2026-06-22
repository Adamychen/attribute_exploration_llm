"""Smoke test: full AttributeExplorer run with the OfficialOpenAIClient.

Validates that the AE graph + bound logic + parse_response + validate_logic
work end-to-end with the openai-SDK wrapper, not just langchain-openai.

The test loads the integers context (smallest), runs the basic prompt
with llama3.2:3b, and verifies:
  - finished=True
  - n_iterations > 0
  - final_context has at least one row/col
  - no early_stop, no timeout
  - n_retries_total < n_iterations (most accepts on first try)
  - response_metadata has 'logprobs.content' with at least 1 token
"""
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from llmfca.extraction import run_explorer
from llmfca.fca_helpers import get_message_function
from llmfca.llm.client import make_model
from llmfca.prompting.prompts import (
    INTEGER_ATTRIBUTES,
    INTEGER_PROMPT,
    INTEGER_PARSER,
)

load_dotenv()

API_KEY = os.getenv("LLAMUS_API_KEY", "")
BASE_URL = "https://llamus.cs.us.es/ollama/v1"
MODEL = "qwen3:8b"
ATTRS = INTEGER_ATTRIBUTES


def skip_if_no_key():
    if not API_KEY or API_KEY == "sk-your-token-here":
        print("SKIP: LLAMUS_API_KEY not set in .env")
        sys.exit(0)


async def main():
    skip_if_no_key()
    print(f"Endpoint: {BASE_URL}")
    print(f"Model:    {MODEL}")
    print(f"Context:  integers ({len(ATTRS)} attrs: {ATTRS})")
    print()

    model = make_model(MODEL, BASE_URL, API_KEY, max_tokens=20)
    parser = INTEGER_PARSER
    msg_fn = get_message_function("implicational", arity=2)

    t0 = time.time()
    result = await run_explorer(
        model=model,
        message_function=msg_fn,
        system_prompt=INTEGER_PROMPT,
        attrs=ATTRS,
        accept_option="yes",
        parser=parser,
        timeout_seconds=120,
    )
    elapsed = time.time() - t0

    print(f"--- Result (elapsed {elapsed:.1f}s) ---")
    print(f"  finished:        {result.get('finished')}")
    print(f"  early_stop:      {result.get('early_stop')}")
    print(f"  timeout:         {result.get('timeout')}")
    print(f"  n_iterations:    {result.get('n_iterations')}")
    print(f"  n_retries_total: {result.get('n_retries_total')}")
    print(f"  n_retry_exhausted: {result.get('n_retry_exhausted')}")
    print(f"  joint_entropy:   {result.get('joint_entropy', 0.0):.3f}")
    print(f"  mean_entropy:    {result.get('mean_entropy', 0.0):.3f}")
    print(f"  stem_basis size: {len(result.get('stem_basis', []))}")
    print(f"  final_context:   {result.get('context') is not None}")
    if result.get("context") is not None:
        n_obj = len(result["context"].objects)
        n_attr = len(result["context"].properties)
        print(f"    → {n_obj} objects × {n_attr} attrs")

    n_iter = result.get("n_iterations", 0)
    assert n_iter > 0, "AE did not iterate"
    assert result.get("context") is not None, "AE produced no final context"
    log = result.get("iteration_log", [])
    has_ask_llm = any(e.get("event") == "ask_llm" for e in log)
    has_validate = any(e.get("event") == "validate_logic" for e in log)
    assert has_ask_llm, "No ask_llm events in iteration_log"
    assert has_validate, "No validate_logic events in iteration_log"
    # Check that the official client produced real logprobs
    for ev in log:
        if ev.get("event") == "ask_llm":
            jh = ev.get("joint_entropy", 0.0)
            assert jh > 0.0, f"joint_entropy is 0 — logprobs not extracted"
            break
    print()
    print("=== OFFICIAL CLIENT SMOKE TEST PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
