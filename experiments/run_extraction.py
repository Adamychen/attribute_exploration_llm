"""
experiments/run_extraction.py — CLI entry point for the Attribute Exploration sweep.

Usage:
    python -m experiments.run_extraction --output results.csv --traces-dir traces
    python -m experiments.run_extraction --contexts integers graphs --prompts basic
    python -m experiments.run_extraction --timeout 3600
    python -m experiments.run_extraction --concurrency 4

Reads .env (LLAMUS_API_KEY) and experiments/models.csv. Writes one row per
(model, question_mode, context, prompt) combination to --output and a JSON
trace per experiment to --traces-dir/<model>__<ctx>__<prompt>__<qmode>.json.
Resumes from the same CSV. Models that are not available at the endpoint
(HTTP 400/404) are probed once and skipped for the rest of the run.

Relative paths in --output and --traces-dir are resolved against the
parent of the `code/` directory (i.e. one level up from this file), so the
tool works the same whether you run it from a standalone checkout or
alongside a paper repository. Use absolute paths to point elsewhere.

Experiments are dispatched to a pool of N asyncio workers, capped by a
Semaphore so the endpoint is not overwhelmed. CSV writes are serialised by
a Lock to avoid interleaved rows; trace JSON filenames are unique per
experiment so they need no lock.
"""
import os
import sys
import json
import asyncio
import argparse
import itertools
import traceback
import time
from typing import Tuple, Set, Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from llmfca.extraction import (
    run_explorer, build_result_row, build_trace_payload, load_processed,
    PROMPTS, ATTRIBUTES, PARSERS, ARITY, ACCEPT_OPTION,
    PROMPT_STRATEGIES, QUESTION_MODES, CONTEXT_NAMES,
    DEFAULT_TIMEOUT_SECONDS,
)
from llmfca.fca_helpers import get_message_function
from llmfca.llm.client import make_model


def _resolve_output_path(arg: str) -> str:
    if os.path.isabs(arg):
        return arg
    return os.path.normpath(os.path.join(REPO_ROOT, arg))


def _resolve_traces_dir(arg: str) -> str:
    if os.path.isabs(arg):
        return arg
    return os.path.normpath(os.path.join(REPO_ROOT, arg))


def _trace_filename(model_name: str, context_name: str, prompt_mode: str, question_mode: str) -> str:
    safe = lambda s: s.replace("/", "_").replace(":", "_").replace(" ", "_")
    return f"{safe(model_name)}__{safe(context_name)}__{safe(prompt_mode)}__{safe(question_mode)}.json"


def probe_model_availability(
    model_name: str,
    base_url: str,
    api_key: str,
    timeout_seconds: int = 30,
) -> bool:
    """Cheap ping: returns False if the model is not available at the endpoint.

    Sends a single minimal 'ok' request. Treats 400 ('not found'), 404, and
    similar 4xx errors as 'unavailable'. Real LLM errors (timeouts, 5xx)
    propagate so the caller can decide.
    """
    try:
        llm = make_model(model_name, base_url, api_key, max_tokens=8)
        llm.invoke("ok")
        return True
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "404" in msg or "400" in msg or "model" in msg and "exist" in msg:
            print(f"  ⚠ {model_name} not available at the endpoint, skipping all (model, ctx, prompt) for it.")
            return False
        print(f"  ⚠ {model_name} probe failed with non-fatal error: {e}; will attempt runs anyway.")
        return True


async def _run_one(
    model_name: str,
    mode: str,
    context_name: str,
    prompt_mode: str,
    url: str,
    api_key: str,
    processed: set,
    unavailable: Set[str],
    probed: Set[str],
    unavailable_lock: asyncio.Lock,
    csv_lock: asyncio.Lock,
    sem: asyncio.Semaphore,
    timeout_seconds: int,
    output_path: str,
    traces_dir: str,
    probe: bool,
    worker_id: int,
) -> None:
    """Run a single experiment, write its CSV row and trace JSON.

    Concurrency invariants:
        - The semaphore caps the number of in-flight LLM calls.
        - `processed` is read-only here (populated once at the start).
        - The probe is done under `unavailable_lock` so each model is pinged
          exactly once even when N workers race for it.
        - `unavailable` and `probed` are read+write under `unavailable_lock`.
        - CSV writes are serialised by `csv_lock`. Trace JSON filenames are
          unique per experiment so they need no lock.
    """
    if (model_name, mode, context_name, prompt_mode) in processed:
        return

    async with sem:
        async with unavailable_lock:
            if model_name in unavailable:
                return
            if probe and model_name not in probed:
                available = await asyncio.to_thread(
                    probe_model_availability, model_name, url, api_key
                )
                probed.add(model_name)
                if not available:
                    unavailable.add(model_name)
                    return

        print(f"\n[w{worker_id}] >>> {model_name} | {mode} | {context_name} | {prompt_mode}")

        model = make_model(model_name, url, api_key, max_tokens=3000)
        prompt = PROMPTS[(context_name, prompt_mode)]
        msg_fn = get_message_function(mode, ARITY[context_name])

        wallclock_start = time.perf_counter()
        try:
            result = await run_explorer(
                model=model,
                message_function=msg_fn,
                attrs=ATTRIBUTES[context_name],
                system_prompt=prompt,
                accept_option=ACCEPT_OPTION[mode],
                parser=PARSERS[context_name],
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[w{worker_id}] ✗ FAILED: {model_name} | {mode} | {context_name} | {prompt_mode}")
            print(f"      Type: {type(e).__name__}")
            print(f"      Error: {e}")
            print(f"      Traceback:\n{tb}")

            failed_result = {
                "context": None,
                "early_stop": True,
                "finished": False,
                "timeout": False,
                "wallclock_start": wallclock_start,
                "n_iterations": 0,
                "n_retries_total": 0,
                "n_retry_exhausted": 0,
                "error_msg": str(e),
                "error_type": type(e).__name__,
                "iteration_log": [],
            }
            try:
                row = build_result_row(model_name, context_name, prompt_mode, mode, failed_result, timeout_seconds)
                df_row = pd.DataFrame([row])
                async with csv_lock:
                    write_header = not os.path.exists(output_path)
                    df_row.to_csv(output_path, mode="a", header=write_header, index=False)
                print(f"  [w{worker_id}] Saved FAILED row to CSV.")
            except Exception as write_err:
                print(f"  [w{worker_id}] CRITICAL: could not even write failed row: {write_err}")
                print(f"  [w{worker_id}] Traceback:\n{traceback.format_exc()}")

            try:
                trace = {
                    "model": model_name,
                    "context_name": context_name,
                    "prompt_mode": prompt_mode,
                    "question_mode": mode,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": tb,
                }
                trace_path = os.path.join(traces_dir, _trace_filename(model_name, context_name, prompt_mode, mode))
                with open(trace_path, "w") as f:
                    json.dump(trace, f, indent=2, default=str)
            except Exception as trace_err:
                print(f"  [w{worker_id}] Could not write error trace: {trace_err}")
            return

        row = build_result_row(model_name, context_name, prompt_mode, mode, result, timeout_seconds)
        df_row = pd.DataFrame([row])
        async with csv_lock:
            write_header = not os.path.exists(output_path)
            df_row.to_csv(output_path, mode="a", header=write_header, index=False)
        print(f"  [w{worker_id}] Saved: early_stop={row['early_stop']}, finished={row['finished']}, "
              f"iter={row['n_iterations']}, retries={row['n_retries_total']}, "
              f"wallclock={row['wallclock_seconds']}s")

        trace = build_trace_payload(model_name, context_name, prompt_mode, mode, result, timeout_seconds)
        trace_path = os.path.join(traces_dir, _trace_filename(model_name, context_name, prompt_mode, mode))
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2, default=str)


async def main_experiment(
    contexts: Tuple[str, ...] = CONTEXT_NAMES,
    prompt_modes: Tuple[str, ...] = PROMPT_STRATEGIES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    output_path: str = "results.csv",
    traces_dir: str = "traces",
    probe: bool = True,
    concurrency: int = 4,
) -> None:
    """Run the full sweep. Resumes from output_path if present.

    When `probe` is True, each model is pinged once before its first
    experiment; unavailable models are skipped wholesale.

    `concurrency` caps the number of in-flight LLM calls. The CSV is written
    under a lock; trace JSONs are written lock-free (unique filenames).
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    load_dotenv()
    api_key = os.getenv("LLAMUS_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMUS_API_KEY not set. Create a .env file at the repo root.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    models_csv = os.path.join(CURRENT, "models.csv")
    df_models = pd.read_csv(models_csv)
    models = df_models.sort_values(by="size", ascending=True)["model"]
    plan = list(itertools.product(models, QUESTION_MODES, contexts, prompt_modes))

    processed = load_processed(output_path)
    unavailable: Set[str] = set()
    probed: Set[str] = set()
    unavailable_lock = asyncio.Lock()
    csv_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    print(f"Plan: {len(plan)} experiments, concurrency={concurrency}, "
          f"already done: {len(processed)}, probe={probe}")

    tasks: List[asyncio.Task] = []
    for i, (model_name, mode, context_name, prompt_mode) in enumerate(plan):
        if (model_name, mode, context_name, prompt_mode) in processed:
            continue
        url = df_models[df_models["model"] == model_name]["url"].iloc[0]
        worker_id = (i % concurrency) + 1
        tasks.append(asyncio.create_task(_run_one(
            model_name=model_name,
            mode=mode,
            context_name=context_name,
            prompt_mode=prompt_mode,
            url=url,
            api_key=api_key,
            processed=processed,
            unavailable=unavailable,
            probed=probed,
            unavailable_lock=unavailable_lock,
            csv_lock=csv_lock,
            sem=sem,
            timeout_seconds=timeout_seconds,
            output_path=output_path,
            traces_dir=traces_dir,
            probe=probe,
            worker_id=worker_id,
        )))

    if not tasks:
        print("Nothing to do (all experiments already processed).")
        return

    print(f"Dispatched {len(tasks)} tasks.")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        print(f"\n⚠ {len(errors)} task(s) raised exceptions:")
        for e in errors[:5]:
            print(f"  - {type(e).__name__}: {e}")
    else:
        print("\n✓ All tasks completed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Attribute Exploration sweep over the configured models."
    )
    parser.add_argument("--contexts", nargs="+", default=None, choices=CONTEXT_NAMES,
                        help="Which formal contexts to explore. Defaults to all four.")
    parser.add_argument("--prompts", nargs="+", default=None, choices=PROMPT_STRATEGIES,
                        help="Which prompt strategies to use. Defaults to all three.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Per-experiment wallclock timeout in seconds.")
    parser.add_argument("--output", default=None,
                        help="Path to the results CSV. Relative paths resolve against "
                             "the parent of the `code/` directory. Required if you "
                             "want to control the destination; otherwise a default of "
                             "`results.csv` (in the current working directory) is used.")
    parser.add_argument("--traces-dir", default=None,
                        help="Directory to write per-experiment JSON traces. Same "
                             "resolution rules as --output. Defaults to `traces`.")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip the model availability probe (not recommended).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max number of in-flight LLM calls (default 4). Lower if you see 429s.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_path = _resolve_output_path(args.output or "results.csv")
    traces_dir = _resolve_traces_dir(args.traces_dir or "traces")
    asyncio.run(main_experiment(
        contexts=tuple(args.contexts) if args.contexts else CONTEXT_NAMES,
        prompt_modes=tuple(args.prompts) if args.prompts else PROMPT_STRATEGIES,
        timeout_seconds=args.timeout,
        output_path=output_path,
        traces_dir=traces_dir,
        probe=not args.no_probe,
        concurrency=args.concurrency,
    ))
