"""
experiments/analyze_traces.py — Quantitative post-mortem analysis of AE traces (R1.8).

Reads all JSON traces in --traces-dir and produces a per-context and
per-model-size-bin breakdown of:
    - distribution of `n_iterations` (pseudo-intent steps)
    - distribution of `n_retries_total`
    - rate of `n_retry_exhausted > 0` (the entropy-derived bound was hit)

Outputs (the files included by the paper):
    - --output             `table_traces_summary.tex` (per context, per size bin)
    - --syntax-output      `table_syntax_vs_semantics.tex`
    - --complexity-output  `table_complexity_outcome.tex` (per context and per
                           size bin, split into completed and aborted runs)
    - --figure-output      `complexity_iter.png` (mean iterations per domain,
                           completed vs aborted)
The last three default to files next to --output (the figure goes to
`../figuras_rendimiento/`, as in the paper).
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

CONTEXTS = ["integers", "graphs", "rcc", "top"]
# Labels used in the trace tables and figure of the paper's appendix.
SHORT_LABELS = {"integers": "Integers", "graphs": "Graphs", "rcc": "RCC", "top": "KR"}
SIZE_BINS = ["Small ($\\leq$4B)", "Medium (5-14B)", "Large ($\\geq$15B)"]


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
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Context} & \textbf{N} & \textbf{$\mu_{iter}$} & "
                 r"\textbf{med$_{iter}$} & \textbf{$\mu_{retries}$} & "
                 r"\textbf{Abort rate} \\ \midrule")
    for ctx in CONTEXTS:
        sub = df[df["context_name"] == ctx]
        if sub.empty:
            continue
        n = len(sub)
        mu_iter = sub["n_iterations"].mean()
        med_iter = sub["n_iterations"].median()
        mu_ret = sub["n_retries_total"].mean()
        es_rate = sub["early_stop"].mean()
        label = SHORT_LABELS.get(ctx, ctx)
        lines.append(f"{label} & {n} & {mu_iter:.1f} & {med_iter:.0f} & {mu_ret:.2f} & "
                     f"{es_rate:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Per-context statistics computed from the JSON traces "
                 r"(second execution). $\mu_{iter}$: mean number of exploration steps; "
                 r"med$_{iter}$: median; $\mu_{retries}$: mean number of retries per "
                 r"exploration; Abort rate: fraction of explorations that ended with an "
                 r"Early Stop or a time-out.}")
    lines.append(r"\label{tab:traces_context}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_per_size_bin_table(df: pd.DataFrame) -> str:
    if "size_bin" not in df.columns or df.empty:
        return ""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model size bin} & \textbf{N runs} & "
                 r"\textbf{$\mu_t$ (s)} & \textbf{$\mu_{iter}$} & "
                 r"\textbf{$\mu_{retries}$} & \textbf{ES-rate} \\ \midrule")
    for bin_name in SIZE_BINS:
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
                 r"$\mu_t$: mean wallclock time (s); $\mu_{iter}$: mean AE iterations; "
                 r"$\mu_{retries}$: mean number of retries per exploration; ES-rate: fraction of runs "
                 r"that resulted in early stop. Large models ($\geq$15B) include "
                 r"\texttt{qwen2.5:72b}, \texttt{llama3.3:70b}, \texttt{llama4:16x17b}, "
                 r"\texttt{gpt-oss:120b} and similar.}")
    lines.append(r"\label{tab:traces_size}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_syntax_vs_semantics_table(df: pd.DataFrame) -> str:
    """Build a table separating syntactic vs semantic validation errors."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Context} & \textbf{N} & "
                 r"\textbf{Syn$_{flags}$} & \textbf{Sem$_{flags}$} & "
                 r"\textbf{$\mu_{ret}$} & \textbf{ES-rate} \\ \midrule")
    for ctx in CONTEXTS:
        sub = df[df["context_name"] == ctx]
        if sub.empty:
            continue
        lines.append(
            f"{SHORT_LABELS.get(ctx, ctx)} & {len(sub)} & {sub['syn_flags'].mean():.2f} & "
            f"{sub['sem_flags'].mean():.2f} & {sub['n_retries_total'].mean():.2f} & "
            f"{sub['early_stop'].mean():.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Validation-error breakdown by category. "
                 r"\textbf{Syn$_{flags}$}: mean syntactic errors per run "
                 r"(JSON format: \texttt{wrong\_len, invalid\_format\_answer, no\_attributes}). "
                 r"\textbf{Sem$_{flags}$}: mean semantic errors per run "
                 r"(\texttt{no\_counterexample, useless\_counterexample}). "
                 r"$\mu_{ret}$: mean number of retries per exploration. "
                 r"ES-rate: fraction of runs that ended in early stop. "
                 r"Repeated-object flags are excluded from both counts.}")
    lines.append(r"\label{tab:syntax_vs_semantics}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _with_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Completed = the exploration finished; Aborted = early stop or timeout."""
    out = df.copy()
    out["outcome"] = np.where(out["finished"], "Completed", "Aborted")
    return out


def _complexity_block(df: pd.DataFrame, key: str, groups: List[str], labels: dict,
                      caption: str, label: str) -> str:
    df = _with_outcome(df)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & \textbf{Outcome} & \textbf{$\mu_t$ (s)} & \textbf{$\tilde{t}$ (s)} & "
             r"\textbf{$\mu_{iter}$} & \textbf{$\mu_{retries}$} \\ \midrule"]
    for g in groups:
        first = True
        for outcome in ("Completed", "Aborted"):
            sub = df[(df[key] == g) & (df["outcome"] == outcome)]
            if sub.empty:
                continue
            head = rf"\multirow{{2}}{{*}}{{{labels.get(g, g)}}}" if first else ""
            first = False
            lines.append(
                f"{head} & {outcome} & {sub['wallclock_seconds'].mean():.2f} & "
                f"{sub['wallclock_seconds'].median():.2f} & {sub['n_iterations'].mean():.1f} & "
                f"{sub['n_retries_total'].mean():.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", caption, rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_complexity_outcome_tables(df: pd.DataFrame) -> str:
    """Per context and per size bin, split into completed and aborted runs."""
    ctx = _complexity_block(
        df, "context_name", CONTEXTS, SHORT_LABELS,
        r"\caption{Complexity and scalability metrics by formal context, split by terminal "
        r"outcome. $\mu_t$: mean wallclock time (s); $\tilde{t}$: median wallclock time; "
        r"$\mu_{iter}$: mean number of exploration steps; $\mu_{retries}$: mean number of "
        r"retries per exploration (second execution).}",
        "tab:complexity_outcome")
    if "size_bin" not in df.columns:
        return ctx
    size = _complexity_block(
        df, "size_bin", SIZE_BINS, {},
        r"\caption{Complexity and scalability metrics by model size bin, split by terminal "
        r"outcome. $\mu_t$: mean wallclock time (s); $\mu_{iter}$: mean AE iterations; "
        r"$\mu_{retries}$: mean number of retries per exploration (second execution).}",
        "tab:complexity_outcome_size")
    return ctx + "\n" + size


def plot_complexity_iter(df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of mean AE iterations per domain, completed vs aborted runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = _with_outcome(df)
    means = df.groupby(["context_name", "outcome"])["n_iterations"].mean()
    x = np.arange(len(CONTEXTS))
    width = 0.35
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i, (outcome, color) in enumerate((("Completed", "#1f77b4"), ("Aborted", "#d62728"))):
        vals = [means.get((c, outcome), np.nan) for c in CONTEXTS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=outcome,
                      color=color, edgecolor="black")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in CONTEXTS], fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Domain", fontsize=14)
    ax.set_ylabel(r"Mean pseudo-intent steps ($\mu_{iter}$)", fontsize=14)
    ax.legend(fontsize=13, loc="upper left", frameon=True, edgecolor="#cccccc")
    sns.despine()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative post-mortem analysis of AE traces.")
    parser.add_argument("--traces-dir", default="traces",
                        help="Directory of per-experiment JSON traces.")
    parser.add_argument("--models", default="experiments/models.csv",
                        help="Path to the models.csv (used for the per-size-bin breakdown).")
    parser.add_argument("--output", default="table_traces_summary.tex",
                        help="Path to the per-context / per-size-bin LaTeX table.")
    parser.add_argument("--syntax-output", default=None,
                        help="Default: table_syntax_vs_semantics.tex next to --output.")
    parser.add_argument("--complexity-output", default=None,
                        help="Default: table_complexity_outcome.tex next to --output.")
    parser.add_argument("--figure-output", default=None,
                        help="Default: ../figuras_rendimiento/complexity_iter.png from --output.")
    args = parser.parse_args()

    traces_dir = _resolve(args.traces_dir)
    models_path = _resolve(args.models)
    output_path = _resolve(args.output)
    out_dir = os.path.dirname(output_path)
    syntax_path = _resolve(args.syntax_output) if args.syntax_output else \
        os.path.join(out_dir, "table_syntax_vs_semantics.tex")
    complexity_path = _resolve(args.complexity_output) if args.complexity_output else \
        os.path.join(out_dir, "table_complexity_outcome.tex")
    figure_path = _resolve(args.figure_output) if args.figure_output else \
        os.path.join(os.path.dirname(out_dir), "figuras_rendimiento", "complexity_iter.png")

    df_models = pd.read_csv(models_path) if os.path.exists(models_path) else None
    print(f"Loading traces from {traces_dir}...")
    df = load_traces(traces_dir, df_models)
    if df.empty:
        print("No traces found; nothing to analyse.")
        return
    print(f"Loaded {len(df)} traces.")

    _write(output_path, build_per_context_table(df) + "\n" + build_per_size_bin_table(df))
    _write(syntax_path, build_syntax_vs_semantics_table(df))
    _write(complexity_path, build_complexity_outcome_tables(df))
    plot_complexity_iter(df, figure_path)
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
