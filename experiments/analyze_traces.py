"""
experiments/analyze_traces.py — Quantitative post-mortem analysis of AE traces (R1.8).

Reads all JSON traces in --traces-dir and produces a per-context and
per-model-size-bin breakdown of:
    - distribution of `n_iterations` (pseudo-intent steps)
    - distribution of `n_retries_total`
    - rate of `n_retry_exhausted > 0` (the entropy-derived bound was hit)
    - mean/percentile of `joint_entropy` and `mean_entropy` at the
      bound-setting step

Output: a single LaTeX table written to --output (default
`table_traces_summary.tex` in the current working directory).
"""
import os
import sys
import json
import glob
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

CURRENT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from experiments.tables import SHORT_LABELS


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


def _size_bin(size: float) -> str:
    if size <= 4:
        return "Small ($\\leq$4B)"
    if size <= 14:
        return "Medium (5-14B)"
    return "Large ($\\geq$15B)"


# --- Flag classification ---
SYNTACTIC_FLAGS = {"invalid_format_answer", "wrong_len", "no_attributes", "not_enough_attrs"}
SEMANTIC_FLAGS = {"no_counterexample", "useless_counterexample"}
IGNORED_FLAGS = {"repeated_object"}


def _classify_flags(flags: list) -> Tuple[int, int]:
    """Return (syntactic_count, semantic_count) for a list of active_flags."""
    syn = sum(1 for f in flags if f in SYNTACTIC_FLAGS)
    sem = sum(1 for f in flags if f in SEMANTIC_FLAGS)
    return syn, sem


def _analyze_log(log: list) -> dict:
    """Walk iteration_log and return per-trace analysis of flag types."""
    syn_total = 0
    sem_total = 0
    last_syn = 0
    last_sem = 0
    last_mixed = 0

    for entry in log:
        flags = entry.get("active_flags")
        if not flags:
            continue
        syn, sem = _classify_flags(flags)
        syn_total += syn
        sem_total += sem

    # Determine what caused the ES (last entry with flags, for ES traces)
    for entry in reversed(log):
        flags = entry.get("active_flags")
        if not flags:
            continue
        syn, sem = _classify_flags(flags)
        last_syn = 1 if syn > 0 and sem == 0 else 0
        last_sem = 1 if sem > 0 and syn == 0 else 0
        last_mixed = 1 if syn > 0 and sem > 0 else 0
        break

    return {
        "syn_flags": syn_total,
        "sem_flags": sem_total,
        "last_syn": last_syn,
        "last_sem": last_sem,
        "last_mixed": last_mixed,
    }


def load_traces(traces_dir: str, df_models: pd.DataFrame = None) -> pd.DataFrame:
    rows: List[dict] = []
    for path in glob.glob(os.path.join(traces_dir, "*.json")):
        try:
            with open(path) as f:
                t = json.load(f)
        except Exception:
            continue
        log = t.get("iteration_log", [])
        flag_analysis = _analyze_log(log)
        row = {
            "model": t.get("model"),
            "context_name": t.get("context_name"),
            "prompt_mode": t.get("prompt_mode"),
            "question_mode": t.get("question_mode"),
            "finished": bool(t.get("finished", False)),
            "early_stop": bool(t.get("early_stop", False)),
            "timeout": bool(t.get("timeout", False)),
            "wallclock_seconds": float(t.get("wallclock_seconds", 0.0)),
            "n_iterations": int(t.get("n_iterations", 0)),
            "n_retries_total": int(t.get("n_retries_total", 0)),
            "n_retry_exhausted": int(t.get("n_retry_exhausted", 0)),
            "joint_entropy": float(t.get("joint_entropy", 0.0)),
            "mean_entropy": float(t.get("mean_entropy", 0.0)),
            **flag_analysis,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df_models is not None and "size" in df_models.columns and not df.empty:
        df = df.merge(df_models[["model", "size"]], on="model", how="left")
        df["size_bin"] = df["size"].apply(_size_bin) if "size" in df.columns else "Unknown"
    return df


def _pct(values: pd.Series, q: float) -> float:
    arr = values.dropna().to_numpy()
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def build_per_context_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Context} & \textbf{N} & \textbf{$\mu_{iter}$} & "
                 r"\textbf{med$_{iter}$} & \textbf{$\mu_{retries}$} & "
                 r"\textbf{Abort rate} & \textbf{$\mu_{H^*}$} & \textbf{p95 $H^*$} \\ \midrule")
    for ctx in ["integers", "graphs", "rcc", "top"]:
        sub = df[df["context_name"] == ctx]
        if sub.empty:
            continue
        n = len(sub)
        mu_iter = sub["n_iterations"].mean()
        med_iter = sub["n_iterations"].median()
        mu_ret = sub["n_retries_total"].mean()
        es_rate = sub["early_stop"].mean()
        mu_H = sub["joint_entropy"].mean()
        p95_H = _pct(sub["joint_entropy"], 95)
        label = SHORT_LABELS.get(ctx, ctx)
        lines.append(f"{label} & {n} & {mu_iter:.1f} & {med_iter:.0f} & {mu_ret:.2f} & "
                     f"{es_rate:.2f} & {mu_H:.2f} & {p95_H:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Per-context statistics computed from the JSON traces. "
                 r"$\mu_{iter}$: mean pseudo-intent steps. med$_{iter}$: median. "
                 r"$\mu_{retries}$: mean retries across runs. Abort rate: fraction of "
                 r"runs that aborted. $\mu_{H^*}$: mean joint entropy at the "
                 r"bound-setting step. p95 $H^*$: 95th percentile.}")
    lines.append(r"\label{tab:traces_context}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_per_size_bin_table(df: pd.DataFrame) -> str:
    if "size_bin" not in df.columns or df.empty:
        return ""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model size bin} & \textbf{N runs} & "
                 r"\textbf{$\mu_t$ (s)} & \textbf{$\mu_{iter}$} & "
                 r"\textbf{$\mu_{retries}$} & \textbf{ES-rate} \\ \midrule")
    for bin_name in ["Small ($\\leq$4B)", "Medium (5-14B)", "Large ($\\geq$15B)"]:
        sub = df[df["size_bin"] == bin_name]
        if sub.empty:
            continue
        n = len(sub)
        mu_t = sub["wallclock_seconds"].mean()
        mu_iter = sub["n_iterations"].mean()
        mu_ret = sub["n_retries_total"].mean()
        es_rate = sub["early_stop"].mean()
        lines.append(f"{bin_name} & {n} & {mu_t:.2f} & {mu_iter:.1f} & {mu_ret:.2f} & {es_rate:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Per-model-size-bin statistics from JSON traces. "
                 r"Large models ($\\geq$15B) include \\texttt{{qwen2.5:72b}}, "
                 r"\\texttt{{llama3.3:70b}}, \\texttt{{llama4:16x17b}}, "
                 r"\\texttt{{gpt-oss:120b}} and similar.}")
    lines.append(r"\label{tab:traces_size}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_syntax_vs_semantics_table(df: pd.DataFrame) -> str:
    """Build a table separating syntactic vs semantic validation errors."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Context} & \textbf{N} & "
                 r"\textbf{Syn$_{flags}$} & \textbf{Sem$_{flags}$} & "
                 r"\textbf{Syn-ES} & \textbf{Sem-ES} & "
                 r"\textbf{$\mu_{ret}$} & \textbf{ES-rate} \\ \midrule")

    for ctx in ["integers", "graphs", "rcc", "top"]:
        sub = df[df["context_name"] == ctx]
        if sub.empty:
            continue
        n = len(sub)
        mu_syn = sub["syn_flags"].mean()
        mu_sem = sub["sem_flags"].mean()
        es_syn = sub[sub["last_syn"] == 1]["early_stop"].sum() / n if n else 0
        es_sem = sub[sub["last_sem"] == 1]["early_stop"].sum() / n if n else 0
        mu_ret = sub["n_retries_total"].mean()
        es_rate = sub["early_stop"].mean()
        label = SHORT_LABELS.get(ctx, ctx)
        lines.append(
            f"{label} & {n} & {mu_syn:.2f} & {mu_sem:.2f} & "
            f"{es_syn:.2f} & {es_sem:.2f} & {mu_ret:.2f} & {es_rate:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Validation-error breakdown by category. "
                 r"\textbf{Syn$_{flags}$}: mean syntactic errors per run "
                 r"(JSON format: \texttt{wrong\_len, invalid\_format\_answer, no\_attributes}). "
                 r"\textbf{Sem$_{flags}$}: mean semantic errors per run "
                 r"(\texttt{no\_counterexample, useless\_counterexample}). "
                 r"\textbf{Syn-ES}: fraction of runs where the last error before early stop "
                 r"was syntactic. "
                 r"\textbf{Sem-ES}: fraction where the last error was semantic. "
                 r"Repeated-object flags are excluded.}")
    lines.append(r"\label{tab:syntax_vs_semantics}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative post-mortem analysis of AE traces.")
    parser.add_argument("--traces-dir", default="traces",
                        help="Directory of per-experiment JSON traces.")
    parser.add_argument("--models", default="experiments/models.csv",
                        help="Path to the models.csv (used for the per-size-bin breakdown).")
    parser.add_argument("--output", default="table_traces_summary.tex",
                        help="Path to the output LaTeX table.")
    args = parser.parse_args()

    traces_dir = _resolve(args.traces_dir)
    models_path = _resolve(args.models)
    output_path = _resolve(args.output)

    df_models = pd.read_csv(models_path) if os.path.exists(models_path) else None
    print(f"Loading traces from {traces_dir}...")
    df = load_traces(traces_dir, df_models)
    if df.empty:
        print("No traces found; nothing to analyse.")
        return
    print(f"Loaded {len(df)} traces.")

    ctx_tex = build_per_context_table(df)
    size_tex = build_per_size_bin_table(df)
    syn_tex = build_syntax_vs_semantics_table(df)
    tex = ctx_tex + "\n" + size_tex + "\n" + syn_tex

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"Saved: {output_path}")
    # Also print a quick summary
    print()
    print("=== Syntax vs Semantics summary ===")
    for ctx in ["integers", "graphs", "rcc", "top"]:
        sub = df[df["context_name"] == ctx]
        print(f"  {ctx}: syn_flags={sub['syn_flags'].mean():.2f}, sem_flags={sub['sem_flags'].mean():.2f}, "
              f"syn_ES={sub[sub['last_syn']==1]['early_stop'].sum()/len(sub):.2%}, "
              f"sem_ES={sub[sub['last_sem']==1]['early_stop'].sum()/len(sub):.2%}, "
              f"ES={sub['early_stop'].mean():.2%}")


if __name__ == "__main__":
    main()
