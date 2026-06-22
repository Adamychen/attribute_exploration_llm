"""
llmfca.extraction — High-level orchestration of an Attribute Exploration run.

Public surface:
    run_explorer(...)     — async; run the AE graph once and return final state.
    build_result_row(...) — flatten the AE result state into a single dict.
    load_processed(...)   — read a results.csv and return the set of completed keys.
    Contexts / prompts    — registries of (PROMPTS, ATTRIBUTES, PARSERS, ...).
    ARITY                 — predicate arity per context.
    ACCEPT_OPTION         — the option that does not need a counterexample per question mode.

Constants (PROMPTS, ATTRIBUTES, PARSERS, etc.) are exposed so external scripts
(such as experiments/run_extraction.py) can iterate over the experimental grid
without re-importing the full prompts module.
"""
import os
import json
import time
import asyncio
from typing import Any, Dict, List, Tuple

import pandas as pd
from concepts import Context

from llmfca.attribute_explorer import AttributeExplorerGraph
from llmfca.fca_helpers import context_to_dict
from llmfca.llm.client import make_model
from llmfca.prompting.prompts import (
    INTEGER_PROMPT, INTEGER_PROMPT_FULL_CONTEXT, INTEGER_PROMPT_STEM_BASIS,
    GRAPHS_PROMPT, GRAPHS_PROMPT_FULL_CONTEXT, GRAPHS_PROMPT_STEM_BASIS,
    TOP_PROMPT, TOP_PROMPT_FULL_CONTEXT, TOP_PROMPT_STEM_BASIS,
    RCC_PROMPT, RCC_PROMPT_FULL_CONTEXT, RCC_PROMPT_STEM_BASIS,
    INTEGER_ATTRIBUTES, ATTRIBUTES_GRAPHS, ATTRIBUTES_TOP, RCC_ATTRIBUTES,
    INTEGER_PARSER, GRAPHS_PARSER, TOP_PARSER, RCC_PARSER,
)

DEFAULT_TIMEOUT_SECONDS = 3600 * 6

PROMPT_STRATEGIES = ("basic", "full_context", "stem_basis")
QUESTION_MODES = ("implicational", "existential")
CONTEXT_NAMES = ("integers", "graphs", "rcc", "top")
ARITY = {"integers": 1, "graphs": 1, "top": 1, "rcc": 2}
ACCEPT_OPTION = {"implicational": "yes", "existential": "no"}

PROMPTS = {
    ("integers", "basic"):         INTEGER_PROMPT,
    ("integers", "full_context"):  INTEGER_PROMPT_FULL_CONTEXT,
    ("integers", "stem_basis"):    INTEGER_PROMPT_STEM_BASIS,
    ("graphs",   "basic"):         GRAPHS_PROMPT,
    ("graphs",   "full_context"):  GRAPHS_PROMPT_FULL_CONTEXT,
    ("graphs",   "stem_basis"):    GRAPHS_PROMPT_STEM_BASIS,
    ("top",      "basic"):         TOP_PROMPT,
    ("top",      "full_context"):  TOP_PROMPT_FULL_CONTEXT,
    ("top",      "stem_basis"):    TOP_PROMPT_STEM_BASIS,
    ("rcc",      "basic"):         RCC_PROMPT,
    ("rcc",      "full_context"):  RCC_PROMPT_FULL_CONTEXT,
    ("rcc",      "stem_basis"):    RCC_PROMPT_STEM_BASIS,
}

ATTRIBUTES = {
    "integers": INTEGER_ATTRIBUTES,
    "graphs":   ATTRIBUTES_GRAPHS,
    "top":      ATTRIBUTES_TOP,
    "rcc":      RCC_ATTRIBUTES,
}

PARSERS = {
    "integers": INTEGER_PARSER,
    "graphs":   GRAPHS_PARSER,
    "top":      TOP_PARSER,
    "rcc":      RCC_PARSER,
}


def _initial_state(attrs: List[str], accept_option: str, parser) -> Dict[str, Any]:
    """Build the typed initial state for the AE graph."""
    return {
        "context": None,
        "stage": "initialization",
        "stem_basis": [],
        "attributes": attrs,
        "current_implication": None,
        "llm_response": "",
        "parsed_response": {},
        "flags": {k: False for k in [
            "wrong_len", "not_enough_attrs", "invalid_format_answer",
            "useless_counterexample", "no_counterexample",
            "no_attributes",
        ]},
        "retries": 0,
        "max_retries": -1,
        "accept_option": accept_option,
        "early_stop": False,
        "finished": False,
        "timeout": False,
        "current_messages": [],
        "joint_entropy": 0.0,
        "mean_entropy": 0.0,
        "parser": parser,
        "parsing_error_msg": None,
        "wallclock_start": time.perf_counter(),
        "n_iterations": 0,
        "n_retries_total": 0,
        "n_retry_exhausted": 0,
        "iteration_log": [],
    }


async def run_explorer(
    model,
    message_function,
    attrs: List[str],
    system_prompt: str,
    accept_option: str,
    parser,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    recursion_limit: int = 100_000,
) -> Dict[str, Any]:
    """Run the AE graph once; return the final state.

    On `asyncio.TimeoutError` the returned state has `timeout=True` and
    `early_stop=True` is *not* forced (callers decide).
    """
    explorer = AttributeExplorerGraph(model, message_function, system_prompt, attrs, parser)
    app = explorer.build()
    state = _initial_state(attrs, accept_option, parser)
    try:
        return await asyncio.wait_for(
            app.ainvoke(state, config={"recursion_limit": recursion_limit}),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        print(f"Timeout {timeout_seconds}s reached; forcing early_stop.")
        state["timeout"] = True
        return state


def _dummy_context() -> Context:
    """A sentinel Context used when exploration aborted."""
    return Context(["'"], ["None"], [["True"]])


def build_result_row(
    model_name: str,
    context_name: str,
    prompt_mode: str,
    question_mode: str,
    result: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Flatten the AE result state into one CSV row."""
    early_stop = bool(result.get("early_stop", False)) or bool(result.get("timeout", False))
    if early_stop:
        ctx = _dummy_context()
    else:
        ctx = result["context"]
    try:
        ctx_data = context_to_dict(ctx)
    except Exception as e:
        print(f"  Error serialising context for {model_name}-{context_name}: {e}")
        ctx = _dummy_context()
        ctx_data = context_to_dict(ctx)
        early_stop = True

    wallclock = time.perf_counter() - result.get("wallclock_start", time.perf_counter())
    if result.get("timeout", False):
        wallclock = float(timeout_seconds)

    return {
        "model": model_name,
        "context_name": context_name,
        "prompt_mode": prompt_mode,
        "question_mode": question_mode,
        "ctx_objects": json.dumps(ctx_data["objects"]),
        "ctx_attributes": json.dumps(ctx_data["properties"]),
        "ctx_relations": json.dumps(ctx_data["context"]),
        "early_stop": early_stop,
        "finished": result.get("finished", False),
        "timeout": result.get("timeout", False),
        "wallclock_seconds": round(wallclock, 3),
        "n_iterations": result.get("n_iterations", 0),
        "n_retries_total": result.get("n_retries_total", 0),
        "n_retry_exhausted": result.get("n_retry_exhausted", 0),
        "error_msg": str(result.get("error_msg", "") or "")[:500],
        "error_type": str(result.get("error_type", "") or ""),
    }


def load_processed(path: str) -> set:
    """Return the set of (model, mode, ctx, prompt) tuples already in `path`."""
    processed: set = set()
    if not os.path.exists(path):
        return processed
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            processed.add((row["model"], row["question_mode"], row["context_name"], row["prompt_mode"]))
        print(f"  Resuming: {len(processed)} experiments already done.")
    except Exception as e:
        print(f"  Warning reading {path} (probably empty): {e}")
    return processed


def build_trace_payload(
    model_name: str,
    context_name: str,
    prompt_mode: str,
    question_mode: str,
    result: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Build a JSON-serialisable trace of one AE run for post-mortem analysis.

    Includes the per-iteration `iteration_log`, the final `stem_basis`, the
    final context (objects, properties, bool matrix), and the cumulative
    metrics (retries, entropy, wallclock).
    """
    log = result.get("iteration_log", []) or []
    stem_basis = [
        {"antecedent": list(imp.antecedent), "consequent": list(imp.consequent)}
        for imp in result.get("stem_basis", [])
    ]
    ctx_data = None
    if result.get("context") is not None and not (result.get("early_stop") or result.get("timeout")):
        try:
            ctx_data = context_to_dict(result["context"])
        except Exception:
            ctx_data = None

    wallclock = time.perf_counter() - result.get("wallclock_start", time.perf_counter())
    if result.get("timeout", False):
        wallclock = float(timeout_seconds)

    return {
        "model": model_name,
        "context_name": context_name,
        "prompt_mode": prompt_mode,
        "question_mode": question_mode,
        "finished": bool(result.get("finished", False)),
        "early_stop": bool(result.get("early_stop", False)),
        "timeout": bool(result.get("timeout", False)),
        "wallclock_seconds": round(wallclock, 3),
        "n_iterations": int(result.get("n_iterations", 0)),
        "n_retries_total": int(result.get("n_retries_total", 0)),
        "n_retry_exhausted": int(result.get("n_retry_exhausted", 0)),
        "joint_entropy": float(result.get("joint_entropy", 0.0)),
        "mean_entropy": float(result.get("mean_entropy", 0.0)),
        "stem_basis": stem_basis,
        "final_context": ctx_data,
        "iteration_log": log,
    }
