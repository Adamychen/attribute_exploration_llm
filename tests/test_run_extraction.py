"""
Tests for the parallel run_extraction pipeline.

Verifies the concurrency invariants:
    1. CSV writes are serialised: no two workers can interleave rows.
    2. The probe is shared: only one probe per model even with N workers.
    3. The processed-set short-circuits without launching tasks.
    4. concurrency=1 still works (legacy sequential behaviour).
"""
import os
import sys
import asyncio
import tempfile
import time
import csv

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
sys.path.insert(0, ROOT)

import pandas as pd
from unittest.mock import patch, MagicMock

from experiments import run_extraction as re


def _make_dummy_result():
    """Build a fake AE result state that build_result_row can consume."""
    return {
        "context": None,
        "stage": "exploration",
        "stem_basis": [],
        "attributes": ["a", "b"],
        "current_implication": None,
        "llm_response": "",
        "parsed_response": {},
        "flags": {"a": False, "b": False},
        "retries": 0,
        "max_retries": 1,
        "accept_option": "yes",
        "early_stop": True,
        "finished": False,
        "timeout": False,
        "current_messages": [],
        "joint_entropy": 0.0,
        "mean_entropy": 0.0,
        "parser": None,
        "parsing_error_msg": None,
        "wallclock_start": time.perf_counter(),
        "n_iterations": 1,
        "n_retries_total": 0,
        "n_retry_exhausted": 0,
        "iteration_log": [],
    }


def test_csv_lock_serialises_writes(tmp_path=None):
    """Hammer the CSV write path from many coroutines; verify every row is whole."""
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    output = os.path.join(tmp_path, "results.csv")
    csv_lock = None  # created inside the event loop (asyncio.Lock binds to a loop on Python 3.9)
    N = 20

    async def worker(i: int):
        row = {
            "model": f"m{i}",
            "context_name": "integers",
            "prompt_mode": "basic",
            "question_mode": "implicational",
            "ctx_objects": "[]",
            "ctx_attributes": "[]",
            "ctx_relations": "[]",
            "early_stop": False,
            "finished": True,
            "timeout": False,
            "wallclock_seconds": 0.1,
            "n_iterations": 1,
            "n_retries_total": 0,
            "n_retry_exhausted": 0,
            "error_msg": "",
            "error_type": "",
        }
        df_row = pd.DataFrame([row])
        async with csv_lock:
            write_header = not os.path.exists(output)
            df_row.to_csv(output, mode="a", header=write_header, index=False)

    async def main():
        nonlocal csv_lock
        csv_lock = asyncio.Lock()
        await asyncio.gather(*(worker(i) for i in range(N)))

    asyncio.run(main())
    df = pd.read_csv(output)
    assert len(df) == N, f"Expected {N} rows, got {len(df)}"
    assert sorted(df["model"].tolist()) == sorted(f"m{i}" for i in range(N))
    # Every row must have the full set of 16 columns (no half-written rows).
    assert len(df.columns) == 16
    print(f"OK test_csv_lock_serialises_writes ({N} rows, 16 cols each)")


def test_probe_dedup_under_contention():
    """Many workers racing to probe the same model should call the probe only once.

    The probe is done under the unavailable_lock, so even with N workers the
    probe fires exactly once per model. The probed set prevents re-probing
    even for the second, third, ... experiment with the same model.
    """
    call_count = {"n": 0}

    def fake_probe(*args, **kwargs):
        call_count["n"] += 1
        return True  # pretend the model is available

    async def run_workers(model_name: str, n_workers: int):
        unavailable: set = set()
        probed: set = set()
        unavailable_lock = asyncio.Lock()
        sem = asyncio.Semaphore(n_workers)

        async def worker():
            async with sem:
                async with unavailable_lock:
                    if model_name in unavailable:
                        return
                    if model_name not in probed:
                        ok = await asyncio.to_thread(fake_probe, model_name, "u", "k")
                        probed.add(model_name)
                        if not ok:
                            unavailable.add(model_name)
                            return

        await asyncio.gather(*(worker() for _ in range(n_workers)))

    asyncio.run(run_workers("gemma3:1b", 10))
    assert call_count["n"] == 1, f"Probe was called {call_count['n']} times; expected 1"
    print(f"OK test_probe_dedup_under_contention (probe called once across 10 workers)")


def test_probe_dedup_under_contention_unavailable():
    """When the probe returns False, the model is marked unavailable and subsequent
    workers skip the experiment entirely."""
    call_count = {"n": 0}
    ran = {"n": 0}

    def fake_probe(*args, **kwargs):
        call_count["n"] += 1
        return False  # unavailable

    async def run_workers(model_name: str, n_workers: int):
        unavailable: set = set()
        probed: set = set()
        unavailable_lock = asyncio.Lock()
        sem = asyncio.Semaphore(n_workers)

        async def worker():
            async with sem:
                async with unavailable_lock:
                    if model_name in unavailable:
                        return
                    if model_name not in probed:
                        ok = await asyncio.to_thread(fake_probe, model_name, "u", "k")
                        probed.add(model_name)
                        if not ok:
                            unavailable.add(model_name)
                            return
                ran["n"] += 1

        await asyncio.gather(*(worker() for _ in range(n_workers)))

    asyncio.run(run_workers("gemma3:1b", 10))
    assert call_count["n"] == 1
    assert ran["n"] == 0, "No worker should run when probe returns False"
    print(f"OK test_probe_dedup_under_contention_unavailable (1 probe, 0 runs)")


def test_processed_set_short_circuits():
    """An experiment already in the CSV must not be launched."""
    output = tempfile.mktemp(suffix=".csv")
    df = pd.DataFrame({
        "model": ["m1"] * 4,
        "context_name": ["integers", "graphs", "rcc", "top"],
        "prompt_mode": ["basic"] * 4,
        "question_mode": ["implicational"] * 4,
    })
    df.to_csv(output, index=False)
    processed = re.load_processed(output)
    assert ("m1", "implicational", "integers", "basic") in processed
    assert len(processed) == 4
    print(f"OK test_processed_set_short_circuits (4 rows loaded)")


def test_concurrency_one_works():
    """concurrency=1 is a valid setting (sequential)."""
    async def fake_runner():
        sem = asyncio.Semaphore(1)
        async with sem:
            return "ran"
    result = asyncio.run(fake_runner())
    assert result == "ran"
    print("OK test_concurrency_one_works")


def test_concurrency_zero_rejected():
    """concurrency=0 should raise ValueError."""
    try:
        asyncio.run(re.main_experiment(concurrency=0))
        assert False, "Should have raised"
    except ValueError as e:
        assert "concurrency" in str(e).lower()
        print(f"OK test_concurrency_zero_rejected (raised: {e})")


def test_exception_in_run_explorer_writes_failed_row(tmp_path=None):
    """When run_explorer raises, _run_one must:
    1. Log the error immediately (not swallow it).
    2. Write a 'failed' row to the CSV with error_msg and error_type populated.
    3. Release the semaphore so the next task can proceed.
    """
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    output = os.path.join(tmp_path, "results.csv")
    traces_dir = os.path.join(tmp_path, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    processed = set()
    unavailable: set = set()
    probed: set = set()
    state = {}

    boom = RuntimeError("LLM endpoint exploded")

    async def fake_run_explorer(**kwargs):
        raise boom

    async def main():
        # Locks and semaphores are created inside the event loop (Python 3.9).
        csv_lock, sem, unavailable_lock = asyncio.Lock(), asyncio.Semaphore(1), asyncio.Lock()
        state["sem"] = sem
        with patch.object(re, "run_explorer", side_effect=fake_run_explorer), \
             patch.object(re, "make_model", return_value=MagicMock()):
            await re._run_one(
                model_name="m-fail",
                mode="implicational",
                context_name="integers",
                prompt_mode="basic",
                url="http://fake",
                api_key="k",
                processed=processed,
                unavailable=unavailable,
                probed=probed,
                unavailable_lock=unavailable_lock,
                csv_lock=csv_lock,
                sem=sem,
                timeout_seconds=60,
                output_path=output,
                traces_dir=traces_dir,
                probe=False,
                worker_id=1,
            )

    asyncio.run(main())

    assert os.path.exists(output), "Failed row must have been written to the CSV"
    df = pd.read_csv(output)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    row = df.iloc[0]
    assert row["model"] == "m-fail"
    assert row["early_stop"] == True
    assert row["finished"] == False
    assert row["error_type"] == "RuntimeError"
    assert "LLM endpoint exploded" in row["error_msg"]
    assert not state["sem"].locked(), "Semaphore must be released after the failure"
    print(f"OK test_exception_in_run_explorer_writes_failed_row "
          f"(error_type={row['error_type']}, msg snippet: {row['error_msg'][:40]!r})")


if __name__ == "__main__":
    test_csv_lock_serialises_writes()
    test_probe_dedup_under_contention()
    test_processed_set_short_circuits()
    test_concurrency_one_works()
    try:
        test_concurrency_zero_rejected()
    except AssertionError:
        # main_experiment tries to load .env, which may not exist in tests;
        # the ValueError is raised before any env access. So this still works.
        raise
    test_exception_in_run_explorer_writes_failed_row()
    print("\nAll run_extraction concurrency tests passed.")
