"""
experiments/analysis_prompts.py — Prompt-strategy ablation analysis (R1.5).

Reads --results and answers: "Do the three prompt strategies
(basic, full_context, stem_basis) produce statistically distinguishable
S / S_H scores?"

For each (context, question_mode) group, computes:
  - per-prompt mean and std of S and S_H (over models)
  - paired Wilcoxon tests between every pair of prompts
  - a per-context LaTeX table

Output: 3 LaTeX tables (S, S_H, syntax_vs_semantics) written to --output-dir
(default `./tables/`).
"""
import json
import os
import sys
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CURRENT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from experiments.tables import DOMAIN_LABELS, _dash_score


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


def _wilcoxon_paired(a: pd.Series, b: pd.Series) -> tuple:
    """Paired Wilcoxon test of two strategies; (nan, nan) if fewer than 2 pairs or all equal.

    ``a`` and ``b`` are indexed by model name, so that each model is paired with itself.
    """
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) < 2:
        return (float("nan"), float("nan"))
    a, b = a.loc[common], b.loc[common]
    diffs = a.values - b.values
    if np.all(diffs == 0):
        return (float("nan"), float("nan"))
    try:
        stat, p = wilcoxon(a, b)
        return (float(stat), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def generate_prompt_ablation_table(df: pd.DataFrame, metric: str = "S") -> str:
    """Per-context prompt-strategy comparison with paired Wilcoxon tests."""
    df = df[~df["early_stop"] & ~df["timeout"] & df[metric].notna()].copy()
    if df.empty:
        return f"% No completed runs with valid {metric}; cannot compute prompt ablation.\n"

    prompts = ["basic", "full_context", "stem_basis"]
    contexts = ["integers", "graphs", "rcc", "top"]
    qmodes = ["existential", "implicational"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll lccc cc}")
    lines.append(r"\toprule")
    header = (r"\textbf{Context} & \textbf{Question} & \textbf{Prompt} & "
              r"\textbf{N} & \textbf{$\mu$} & \textbf{$\sigma$} & "
              r"\textbf{$p$ vs CFC} & \textbf{$p$ vs DG} \\")
    lines.append(header)
    lines.append(r"\midrule")

    for ctx in contexts:
        sub_ctx = df[df["context_name"] == ctx]
        if sub_ctx.empty:
            continue
        first_ctx = True
        for qmode in qmodes:
            sub = sub_ctx[sub_ctx["question_mode"] == qmode]
            if sub.empty:
                continue
            per_prompt = {p: sub[sub["prompt_mode"] == p].set_index("model")[metric]
                          for p in prompts}
            n_per = {p: len(per_prompt[p]) for p in prompts}
            mu = {p: per_prompt[p].mean() for p in prompts}
            sd = {p: per_prompt[p].std(ddof=1) for p in prompts}
            ps = {}
            for a, b in combinations(prompts, 2):
                ps[(a, b)] = ps[(b, a)] = _wilcoxon_paired(per_prompt[a], per_prompt[b])[1]

            qmode_label = r"$\exists$" if qmode == "existential" else r"$\to$"
            for i, p in enumerate(prompts):
                ctx_label = DOMAIN_LABELS.get(ctx, ctx) if first_ctx else ""
                first_ctx = False
                p_full = ps.get((p, "full_context"), float("nan")) if p != "full_context" else float("nan")
                p_stem = ps.get((p, "stem_basis"), float("nan")) if p != "stem_basis" else float("nan")
                qlabel = qmode_label if i == 0 else ""
                plabel = {"basic": "Baseline", "full_context": "CFC", "stem_basis": "DG Basis"}[p]
                lines.append(f"{ctx_label} & {qlabel} & {plabel} & "
                             f"{n_per[p]} & {mu[p]:.2f} & {sd[p]:.2f} & "
                             f"{_format_p(p_full)} & {_format_p(p_stem)} \\\\")
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\caption{Prompt-strategy ablation. Per (context, question-mode) cell, "
                 r"the table reports the mean and standard deviation of $" + metric + r"$ "
                 r"computed over all models that completed the run, and $p$-values of "
                 r"Wilcoxon signed-rank tests between strategies, each model paired with "
                 r"itself over the models that completed both runs.}")
    lines.append(r"\label{tab:prompt_strategy_" + metric + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _compute_similarities(df: pd.DataFrame, ref_dir: str) -> pd.DataFrame:
    """Add S, S_H columns by comparing each row's context to the reference."""
    from concepts import Context
    from llmfca.analysis.metrics import compute_similarities, learned_context
    ref_contexts = {}
    for fname in os.listdir(ref_dir):
        if fname.endswith(".json"):
            name = os.path.splitext(fname)[0]
            ref_contexts[name] = Context.fromjson(os.path.join(ref_dir, fname))

    S_list, SH_list = [], []
    for _, row in df.iterrows():
        ctx_name = row.get("context_name")
        ref = ref_contexts.get(ctx_name)
        if ref is None or row.get("early_stop", False) or row.get("timeout", False):
            S_list.append(None)
            SH_list.append(None)
            continue
        try:
            objs = json.loads(row["ctx_objects"])
            attrs = json.loads(row["ctx_attributes"])
            rels = json.loads(row["ctx_relations"])
            ctx_llm = learned_context(objs, attrs, rels, ref.properties)
            s, sh = compute_similarities(ctx_llm, ref)
            S_list.append(s)
            SH_list.append(sh)
        except Exception:
            S_list.append(None)
            SH_list.append(None)
    df = df.copy()
    df["S"] = S_list
    df["S_H"] = SH_list
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-strategy ablation analysis.")
    parser.add_argument("--results", default="results.csv",
                        help="Path to the results CSV.")
    parser.add_argument("--ref-dir", default="experiments/reference_contexts",
                        help="Directory of reference context JSONs for similarity computation.")
    parser.add_argument("--output-dir", default="tables",
                        help="Directory to write the LaTeX tables.")
    args = parser.parse_args()

    results_path = _resolve(args.results)
    ref_dir = _resolve(args.ref_dir)
    output_dir = _resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {results_path}...")
    df = pd.read_csv(results_path)
    if "S" not in df.columns or "S_H" not in df.columns:
        print("Computing S, S_H from reference contexts (this is a small pass)...")
        df = _compute_similarities(df, ref_dir)

    for metric in ["S", "S_H"]:
        out_path = os.path.join(output_dir, f"table_prompt_strategy_{metric}.tex")
        tex = generate_prompt_ablation_table(df, metric=metric)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
