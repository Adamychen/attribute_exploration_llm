"""
experiments/run_benchmarks.py — External LLM benchmarks (R1.6).

Loads real MMLU and GSM8K subsets from HuggingFace and scores each model in
experiments/models.csv on them. Writes a per-model accuracy CSV to
--output (default `benchmark_results.csv` in the current working directory).

Datasets
--------
* MMLU:  cais/mmlu, iterating per academic subject (config = subject name,
  split='test'). Stratified sample of --n-mmlu-per-subject questions
  across the 57 academic subjects (humanities, STEM, social sciences,
  other). We deliberately avoid the 'all' config because it would also
  download the 99k-row auxiliary_train split (~160 MB) that is unused
  here.
* GSM8K: openai/gsm8k (config='main', split='test'). Random sample of
  --n-gsm8k grade-school math problems.

The first run downloads each MMLU test split into --cache-dir
(~30 MB total) and the GSM8K test split (~3 MB). Subsequent runs are
fully offline against the local cache.

Usage:
    python -m experiments.run_benchmarks
    python -m experiments.run_benchmarks --n-mmlu-per-subject 5 --n-gsm8k 100
    python -m experiments.run_benchmarks --n-mmlu-per-subject 1 --n-gsm8k 5  # smoke
    python -m experiments.run_benchmarks --no-hf            # offline fallback
"""
import os
import sys
import json
import asyncio
import argparse
import re

import pandas as pd
from dotenv import load_dotenv

CURRENT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from llmfca.llm.client import make_model


MMLU_DATASET = "cais/mmlu"
GSM8K_DATASET = "openai/gsm8k"


def _resolve(arg: str) -> str:
    """Resolve `arg` against the appropriate base directory.

    - Absolute paths are returned as-is.
    - Paths starting with `experiments/` or `llmfca/` resolve against the
      `code/` directory.
    - Everything else resolves against the current working directory.
    """
    if os.path.isabs(arg):
        return arg
    if arg.startswith("experiments/") or arg.startswith("llmfca/"):
        return os.path.normpath(os.path.join(CODE_DIR, arg))
    return os.path.abspath(arg)


def _build_prompt_mmlu(q: dict) -> str:
    body = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(q["choices"]))
    return (
        f"Answer the following multiple-choice question with a single letter (A, B, C, or D).\n"
        f"Question: {q['question']}\n{body}\n\n"
        f"Respond with ONLY the letter of the correct answer. No explanations."
    )


def _build_prompt_gsm8k(q: str) -> str:
    return (
        f"Solve the following math problem. End your answer with 'Answer: <number>'.\n"
        f"Problem: {q}"
    )


def _extract_letter(text: str) -> str:
    """Return the first capital A-D found in the response, or '' if none."""
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else ""


def _extract_number(text: str) -> float:
    """Extract the final numeric answer, preferring the 'Answer: X' marker."""
    m = re.search(r"answer\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text.strip().lower())
    if m:
        return float(m.group(1))
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text.strip())
    if m:
        return float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])
    return float("nan")


def _list_mmlu_subjects(cache_dir: str) -> list:
    """Return the 57 MMLU academic subject names (excluding 'all' and 'auxiliary_train')."""
    from datasets import get_dataset_config_names
    return sorted(c for c in get_dataset_config_names(MMLU_DATASET)
                  if c not in {"all", "auxiliary_train"})


def _load_mmlu_stratified(
    n_per_subject: int,
    cache_dir: str,
    seed: int,
    max_subjects: int | None = None,
) -> list:
    """Stratified sample of MMLU test questions, ``n_per_subject`` per subject.

    Iterates per subject config (avoids downloading the 99k-row auxiliary_train
    embedded in the 'all' config). Caches each subject's test split on disk
    for fast re-runs.

    Returns a list of dicts with keys: question, choices (list of 4), answer
    (A/B/C/D), subject.
    """
    from datasets import load_dataset

    subjects = _list_mmlu_subjects(cache_dir)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    rng = np_random(seed)
    sampled: list = []
    for subj in subjects:
        try:
            ds = load_dataset(
                MMLU_DATASET, subj, split="test",
                cache_dir=cache_dir,
                verification_mode="no_checks",
            )
        except Exception as e:
            print(f"  [mmlu] skipping subject {subj}: {e}")
            continue
        k = min(n_per_subject, len(ds))
        idxs = rng.choice(len(ds), size=k, replace=False) if k > 0 else []
        for i in idxs:
            row = ds[int(i)]
            sampled.append({
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": chr(65 + int(row["answer"])),
                "subject": row.get("subject", subj),
            })
    return sampled


def _load_gsm8k(n: int, cache_dir: str, seed: int) -> list:
    """Random sample of n GSM8K test questions, deterministically.

    Returns a list of dicts: question, answer (the numeric ground truth).
    """
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, "main", split="test", cache_dir=cache_dir)
    rng = np_random(seed)
    k = min(n, len(ds))
    idxs = rng.choice(len(ds), size=k, replace=False)
    sampled = []
    for i in idxs:
        row = ds[int(i)]
        gold = _extract_number(row["answer"])
        sampled.append({"question": row["question"], "answer": float(gold)})
    return sampled


def _offline_mmlu_subset() -> list:
    """Last-resort offline fallback: the original 10-question probe.

    Used only when --no-hf is set (e.g. inside CI without internet).
    """
    return [
        {"question": "What is the capital of France?", "choices": ["London", "Paris", "Berlin", "Madrid"], "answer": "B", "subject": "geography"},
        {"question": "Which gas is most abundant in Earth's atmosphere?", "choices": ["Oxygen", "Hydrogen", "Nitrogen", "Carbon dioxide"], "answer": "C", "subject": "chemistry"},
        {"question": "What is 7 multiplied by 8?", "choices": ["54", "56", "64", "49"], "answer": "B", "subject": "elementary_mathematics"},
        {"question": "Who wrote 'One Hundred Years of Solitude'?", "choices": ["Jorge Luis Borges", "Gabriel García Márquez", "Mario Vargas Llosa", "Pablo Neruda"], "answer": "B", "subject": "world_literature"},
        {"question": "Which planet has the most moons (as of 2024)?", "choices": ["Jupiter", "Saturn", "Uranus", "Neptune"], "answer": "B", "subject": "astronomy"},
        {"question": "What is the chemical symbol for gold?", "choices": ["Go", "Gd", "Au", "Ag"], "answer": "C", "subject": "high_school_chemistry"},
        {"question": "In which year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": "C", "subject": "world_history"},
        {"question": "What is the largest ocean on Earth?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "D", "subject": "geography"},
        {"question": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "answer": "C", "subject": "elementary_mathematics"},
        {"question": "Which element has the atomic number 1?", "choices": ["Helium", "Hydrogen", "Lithium", "Carbon"], "answer": "B", "subject": "high_school_chemistry"},
    ]


def _offline_gsm8k_subset() -> list:
    """Last-resort offline fallback: the original 5-question probe."""
    return [
        {"question": "Janet has 3 apples. She buys 5 more. How many apples does she have now?", "answer": 8.0},
        {"question": "A train travels 60 miles per hour for 2 hours. How far does it go?", "answer": 120.0},
        {"question": "If 5 boxes contain 4 pens each, how many pens are there in total?", "answer": 20.0},
        {"question": "Tom is 8 years old. His sister is twice his age. How old is his sister?", "answer": 16.0},
        {"question": "A shirt costs $25. It is on sale for 20% off. What is the sale price?", "answer": 20.0},
    ]


def np_random(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


async def _ask(llm, prompt: str, timeout: int) -> str:
    return await asyncio.wait_for(
        asyncio.to_thread(llm.invoke, prompt),
        timeout=timeout,
    )


async def _score_model(
    model_name: str,
    base_url: str,
    api_key: str,
    mmlu_questions: list,
    gsm8k_questions: list,
    timeout: int,
) -> dict:
    """Score one model against the MMLU/GSM8K subsets.

    Distinguishes three failure modes:
        - `unavailable`: model not found at the endpoint (HTTP 400/404).
        - `error`: at least one call raised an exception other than 'not found'.
        - `ok`: all calls succeeded.
    Failures propagate as NaN accuracies (not 0) so that downstream correlation
    analyses do not treat them as "model scored zero".
    """
    try:
        llm = make_model(model_name, base_url, api_key, max_tokens=512, top_logprobs=0, logprobs=False)
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "404" in msg or "400" in msg:
            return {
                "model": model_name,
                "n_mmlu": len(mmlu_questions),
                "correct_mmlu": 0,
                "n_gsm8k": len(gsm8k_questions),
                "correct_gsm8k": 0,
                "accuracy_mmlu": float("nan"),
                "accuracy_gsm8k": float("nan"),
                "status": "unavailable",
            }
        raise

    mmlu_correct = 0
    mmlu_failures = 0
    for q in mmlu_questions:
        try:
            resp = await _ask(llm, _build_prompt_mmlu(q), timeout=timeout)
            text = resp.content if hasattr(resp, "content") else str(resp)
            pred = _extract_letter(text)
            if pred == q["answer"]:
                mmlu_correct += 1
        except Exception as e:
            mmlu_failures += 1
            msg = str(e).lower()
            if "not found" in msg or "404" in msg or "400" in msg:
                return {
                    "model": model_name,
                    "n_mmlu": len(mmlu_questions),
                    "correct_mmlu": mmlu_correct,
                    "n_gsm8k": len(gsm8k_questions),
                    "correct_gsm8k": 0,
                    "accuracy_mmlu": mmlu_correct / len(mmlu_questions) if mmlu_questions else float("nan"),
                    "accuracy_gsm8k": float("nan"),
                    "status": "unavailable",
                }
            print(f"    [mmlu] {model_name} error: {e}")

    gsm8k_correct = 0
    gsm8k_failures = 0
    for q in gsm8k_questions:
        try:
            resp = await _ask(llm, _build_prompt_gsm8k(q["question"]), timeout=timeout)
            text = resp.content if hasattr(resp, "content") else str(resp)
            pred = _extract_number(text)
            if pred == q["answer"]:
                gsm8k_correct += 1
        except Exception as e:
            gsm8k_failures += 1
            msg = str(e).lower()
            if "not found" in msg or "404" in msg or "400" in msg:
                return {
                    "model": model_name,
                    "n_mmlu": len(mmlu_questions),
                    "correct_mmlu": mmlu_correct,
                    "n_gsm8k": len(gsm8k_questions),
                    "correct_gsm8k": gsm8k_correct,
                    "accuracy_mmlu": mmlu_correct / len(mmlu_questions) if mmlu_questions else float("nan"),
                    "accuracy_gsm8k": gsm8k_correct / len(gsm8k_questions) if gsm8k_questions else float("nan"),
                    "status": "unavailable",
                }
            print(f"    [gsm8k] {model_name} error: {e}")

    if mmlu_failures > 0 or gsm8k_failures > 0:
        status = "error"
    else:
        status = "ok"
    return {
        "model": model_name,
        "n_mmlu": len(mmlu_questions),
        "correct_mmlu": mmlu_correct,
        "n_gsm8k": len(gsm8k_questions),
        "correct_gsm8k": gsm8k_correct,
        "accuracy_mmlu": mmlu_correct / len(mmlu_questions) if mmlu_questions else float("nan"),
        "accuracy_gsm8k": gsm8k_correct / len(gsm8k_questions) if gsm8k_questions else float("nan"),
        "status": status,
    }


def _load_benchmarks(
    n_mmlu_per_subject: int,
    n_gsm8k: int,
    cache_dir: str,
    seed: int,
    use_hf: bool,
) -> tuple:
    """Return (mmlu_questions, gsm8k_questions) following the configured policy.

    If use_hf is True, attempt to load real data from HuggingFace and fall
    back to the offline probe on any network/dataset error.
    """
    if not use_hf:
        print(">>> [benchmarks] using offline probe (--no-hf)")
        return _offline_mmlu_subset(), _offline_gsm8k_subset()

    print(">>> [benchmarks] loading MMLU (cais/mmlu, per-subject, split='test')")
    try:
        mmlu = _load_mmlu_stratified(
            n_per_subject=n_mmlu_per_subject,
            cache_dir=cache_dir,
            seed=seed,
        )
    except Exception as e:
        print(f"  [mmlu] HF load failed: {e}. Falling back to offline probe.")
        mmlu = _offline_mmlu_subset()

    print(f">>> [benchmarks] loading GSM8K (openai/gsm8k, 'main', split='test')")
    try:
        gsm8k = _load_gsm8k(n=n_gsm8k, cache_dir=cache_dir, seed=seed)
    except Exception as e:
        print(f"  [gsm8k] HF load failed: {e}. Falling back to offline probe.")
        gsm8k = _offline_gsm8k_subset()

    return mmlu, gsm8k


async def main_async(args) -> None:
    load_dotenv()
    api_key = os.getenv("LLAMUS_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMUS_API_KEY not set. Create a .env file at the repo root.")

    output_path = _resolve(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    mmlu_questions, gsm8k_questions = _load_benchmarks(
        n_mmlu_per_subject=args.n_mmlu_per_subject,
        n_gsm8k=args.n_gsm8k,
        cache_dir=args.cache_dir,
        seed=args.seed,
        use_hf=args.use_hf,
    )
    print(f">>> [benchmarks] MMLU questions: {len(mmlu_questions)}; GSM8K questions: {len(gsm8k_questions)}")

    models_csv = os.path.join(CURRENT, "models.csv")
    df_models = pd.read_csv(models_csv)
    models = df_models.sort_values(by="size", ascending=True)["model"]

    results: list = []
    for model_name in models:
        url = df_models[df_models["model"] == model_name]["url"].iloc[0]
        print(f">>> {model_name}")
        try:
            row = await _score_model(
                model_name, url, api_key,
                mmlu_questions=mmlu_questions,
                gsm8k_questions=gsm8k_questions,
                timeout=args.timeout,
            )
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            row = {
                "model": model_name,
                "n_mmlu": len(mmlu_questions),
                "correct_mmlu": 0,
                "n_gsm8k": len(gsm8k_questions),
                "correct_gsm8k": 0,
                "accuracy_mmlu": float("nan"),
                "accuracy_gsm8k": float("nan"),
                "status": "error",
            }
        results.append(row)
        print(f"  status={row.get('status', 'ok')}  "
              f"mmlu={row['correct_mmlu']}/{row['n_mmlu']}  "
              f"gsm8k={row['correct_gsm8k']}/{row['n_gsm8k']}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU/GSM8K subset for each model.")
    parser.add_argument("--n-mmlu-per-subject", type=int, default=5,
                        help="Number of MMLU questions sampled per academic subject (default 5 → ~285 questions).")
    parser.add_argument("--n-gsm8k", type=int, default=100,
                        help="Number of GSM8K test questions to sample (default 100).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible stratified sampling (default 42).")
    parser.add_argument("--cache-dir", default="_hf_cache",
                        help="Local HuggingFace cache directory. Relative paths "
                             "resolve against the current working directory.")
    parser.add_argument("--no-hf", action="store_true",
                        help="Use the hardcoded 10+5 offline probe instead of loading from HuggingFace.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Path to the per-model accuracy CSV.")
    args = parser.parse_args()

    args.use_hf = not args.no_hf
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
