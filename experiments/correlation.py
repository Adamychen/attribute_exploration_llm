"""
experiments/correlation.py — Correlation between FCA similarity and benchmark accuracy.

Reads the results CSV (--results) and computes S, S_H on the fly from the
per-row columns ctx_objects / ctx_attributes / ctx_relations, which are the
JSON serialisation of the LLM-extracted formal context. No auxiliary file
(results_with_s.csv) is required.

Joins per-model mean S and S_H with the benchmark accuracy CSV
(--benchmarks; columns accuracy_mmlu, accuracy_gsm8k per model) and reports
Pearson and Spearman correlations with p-values, per formal context.

Output: a single LaTeX table written to --output (default
`table_correlation.tex` in the current working directory).
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from concepts import Context

CURRENT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from llmfca.analysis.metrics import compute_similarities
from experiments.tables import DOMAIN_LABELS


REPLACEMENTS = {
    "qwen3-coder:latest": "qwen3-coder:30b",
    "phi4-mini:latest": "phi4-mini:3.8b",
}


def _resolve(arg: str) -> str:
    """Resolve `arg` against the appropriate base directory.

    - Absolute paths are returned as-is.
    - Paths starting with `experiments/` or `llmfca/` resolve against the
      `code/` directory (i.e. they live inside this package).
    - Everything else is resolved against the current working directory,
      so a relative path like `results.csv` opens a file in the directory
      from which the script was launched.
    """
    if os.path.isabs(arg):
        return arg
    if arg.startswith("experiments/") or arg.startswith("llmfca/"):
        return os.path.normpath(os.path.join(CODE_DIR, arg))
    return os.path.abspath(arg)


def _build_context_from_row(row) -> Context:
    objs = json.loads(row["ctx_objects"]) if isinstance(row["ctx_objects"], str) else row["ctx_objects"]
    attrs = json.loads(row["ctx_attributes"]) if isinstance(row["ctx_attributes"], str) else row["ctx_attributes"]
    rels = json.loads(row["ctx_relations"]) if isinstance(row["ctx_relations"], str) else row["ctx_relations"]
    return Context(objs, attrs, rels)


def _load_reference_contexts(ref_dir: str):
    contexts = {}
    for fname in os.listdir(ref_dir):
        if fname.endswith(".json"):
            name = os.path.splitext(fname)[0]
            try:
                contexts[name] = Context.fromjson(os.path.join(ref_dir, fname))
            except Exception:
                continue
    return contexts


def _compute_s_sh(df: pd.DataFrame, ref_contexts: dict) -> pd.DataFrame:
    """Compute S and S_H per row from ctx_objects/ctx_attributes/ctx_relations."""
    S_list, SH_list = [], []
    n_ok, n_skip = 0, 0
    for _, row in df.iterrows():
        ctx_name = row.get("context_name")
        ref = ref_contexts.get(ctx_name) if ctx_name else None
        if ref is None or row.get("early_stop", False) or row.get("timeout", False):
            S_list.append(None)
            SH_list.append(None)
            n_skip += 1
            continue
        try:
            ctx_llm = _build_context_from_row(row)
            S, SH = compute_similarities(ctx_llm, ref)
            S_list.append(S)
            SH_list.append(SH)
            n_ok += 1
        except Exception:
            S_list.append(None)
            SH_list.append(None)
            n_skip += 1
    df = df.copy()
    df["S"] = S_list
    df["S_H"] = SH_list
    print(f"  S/S_H computed for {n_ok} rows ({n_skip} skipped: early-stop/timeout/no-ref).")
    return df


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return (float("nan"), float("nan"), int(mask.sum()))
    if x.nunique() < 2 or y.nunique() < 2:
        return (float("nan"), float("nan"), int(mask.sum()))
    try:
        if method == "pearson":
            r, p = pearsonr(x, y)
        else:
            r, p = spearmanr(x, y)
        return (float(r), float(p), int(mask.sum()))
    except Exception:
        return (float("nan"), float("nan"), int(mask.sum()))


def _fmt_corr(r: float, p: float) -> str:
    if pd.isna(r):
        return "---"
    sig = ""
    if p < 0.05:
        sig = "^{*}"
    if p < 0.01:
        sig = "^{**}"
    if p < 0.001:
        sig = "^{***}"
    return f"${r:.2f}{sig}$"


def generate_correlation_table(fca_df: pd.DataFrame, bench_df: pd.DataFrame) -> str:
    """Per-context correlation between (S, S_H) and (accuracy_mmlu, accuracy_gsm8k)."""
    valid = fca_df[(~fca_df["early_stop"]) & (~fca_df["timeout"]) & fca_df["S"].notna()].copy()
    if valid.empty:
        return "% No completed runs; cannot compute correlations.\n"

    contexts = ["integers", "graphs", "rcc", "top"]
    benchmark_cols = [("accuracy_mmlu", "MMLU"), ("accuracy_gsm8k", "GSM8K")]
    fca_cols = [("S", "$\\mu_S$"), ("S_H", "$\\mu_{S_H}$")]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    ncols = 2 + 2 * len(fca_cols) * len(benchmark_cols)
    col_spec = "l" + "c" * (ncols - 1)
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = r"\textbf{Context} & \textbf{N}"
    for fca_label in [c[1] for c in fca_cols]:
        for bench_label in [b[1] for b in benchmark_cols]:
            header += f" & \\multicolumn{{2}}{{c}}{{{fca_label} $\\times$ {bench_label}}}"
    header += r" \\"
    lines.append(header)
    sub = r" & "
    for _ in fca_cols:
        for _ in benchmark_cols:
            sub += " & $r_p$ & $r_s$"
    lines.append(sub + r" \\")
    lines.append(r"\midrule")

    for ctx in contexts:
        sub_ctx = valid[valid["context_name"] == ctx]
        if sub_ctx.empty:
            continue
        ctx_per_model = (
            sub_ctx.groupby("model")
            .agg(S=("S", "mean"), S_H=("S_H", "mean"))
            .reset_index()
        )
        m = ctx_per_model.merge(bench_df, on="model", how="inner")
        if m.empty:
            continue
        ctx_label = DOMAIN_LABELS.get(ctx, ctx)
        n_models = len(m)
        row = f"{ctx_label} & {n_models}"
        for fca_col, _ in fca_cols:
            for bench_col, _ in benchmark_cols:
                rp, pp, _ = _safe_corr(m[fca_col], m[bench_col], "pearson")
                rs, ps, _ = _safe_corr(m[fca_col], m[bench_col], "spearman")
                row += f" & {_fmt_corr(rp, pp)} & {_fmt_corr(rs, ps)}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\caption{Correlation between FCA similarity scores and external benchmark "
                 r"accuracy per formal context. $r_p$: Pearson. $r_s$: Spearman. "
                 r"Asterisks: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}")
    lines.append(r"\label{tab:correlation_benchmarks}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.csv",
                        help="FCA results CSV. S and S_H are computed on the fly from "
                             "the per-row ctx_objects/ctx_attributes/ctx_relations columns.")
    parser.add_argument("--benchmarks", default="experiments/data/benchmark_results.csv",
                        help="Path to the benchmark accuracy CSV (MMLU, GSM8K per model).")
    parser.add_argument("--ref-dir", default="experiments/reference_contexts",
                        help="Directory of reference context JSONs for similarity computation.")
    parser.add_argument("--output", default="table_correlation.tex",
                        help="Path to the output LaTeX table.")
    args = parser.parse_args()

    fca_path = _resolve(args.results)
    bench_path = _resolve(args.benchmarks)
    ref_dir = _resolve(args.ref_dir)
    output_path = _resolve(args.output)

    if not os.path.exists(fca_path):
        raise FileNotFoundError(f"FCA results not found: {fca_path}. Run `python -m experiments.run_extraction` first.")
    if not os.path.exists(bench_path):
        raise FileNotFoundError(f"Benchmark results not found: {bench_path}. Run `python -m experiments.run_benchmarks` first.")

    print(f"Loading FCA results from: {fca_path}")
    fca_df = pd.read_csv(fca_path)
    fca_df["model"] = fca_df["model"].replace(REPLACEMENTS)

    print(f"Loading reference contexts from: {ref_dir}")
    ref_contexts = _load_reference_contexts(ref_dir)

    print("Computing S and S_H on the fly from ctx_objects/ctx_attributes/ctx_relations...")
    fca_df = _compute_s_sh(fca_df, ref_contexts)

    print(f"Loading benchmark results from: {bench_path}")
    bench_df = pd.read_csv(bench_path)

    tex = generate_correlation_table(fca_df, bench_df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
