from typing import Dict, List, Tuple, Optional
import pandas as pd

DOMAIN_LABELS = {
    "integers": "Integer Numbers",
    "graphs": "Graph Theory",
    "rcc": "Region Connection Calculus",
    "top": "KR Ontology"
}

SHORT_LABELS = DOMAIN_LABELS

PROMPT_LABELS = {
    "basic": "Baseline",
    "full_context": "CFC",
    "stem_basis": "Stem Basis"
}


def _safe_score(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "---"
    if isinstance(val, str):
        return val
    return f"{val:.2f}"


def _es_score(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == "ES":
        return "ES"
    return f"{val:.2f}"


def _dash_score(val) -> str:
    """Hidratación para celdas con cómputo abortado (early_stop/timeout).

    Devuelve '—' en vez de 'ES' para ocultar cómputos fallidos (R2.8).
    Mantiene el formato '0.xx' para valores numéricos válidos.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == "ES":
        return "—"
    return f"{val:.2f}"


def _format_pair(S, S_H):
    return f"${_dash_score(S)}$ & ${_dash_score(S_H)}$"


def _agg_score_per_model(model_df: pd.DataFrame) -> float:
    """Mean of all (S, S_H) cells for one model, treating early-stop/NaN/None as 0.0.

    Used to sort the per-domain comparative table by aggregate performance
    (descending). A model with a row full of em-dash cells scores 0.0;
    a model with all 12 cells at 1.00 scores 1.0.
    """
    import math
    vals = []
    for _, r in model_df.iterrows():
        if r.get("early_stop", False) or r.get("timeout", False):
            vals.extend([0.0, 0.0])
            continue
        for col in ("S", "S_H"):
            v = r.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append(0.0)
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
    return sum(vals) / len(vals) if vals else 0.0


def generate_domain_table(df: pd.DataFrame, context_name: str) -> str:
    sub = df[df["context_name"] == context_name].copy()
    prompt_order = ["basic", "full_context", "stem_basis"]
    prompt_order = [p for p in prompt_order if p in sub["prompt_mode"].unique()]

    lines = []
    label = DOMAIN_LABELS.get(context_name, context_name)
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    n_strategies = len(prompt_order)
    total_cols = 2 + n_strategies * 4
    col_spec = "l" + " cc" * (n_strategies * 2)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{" + str(total_cols) + r"}{\textit{" + label + r"}} \\")
    lines.append(r"\toprule")
    cols = []
    for p in prompt_order:
        cols.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{PROMPT_LABELS[p]}}}}}")
    header1 = " & " + " & ".join(cols) + r" \\"
    lines.append(header1)
    parts = []
    for i in range(n_strategies):
        start = 2 + i * 4
        end = 5 + i * 4
        parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" " + " ".join(parts))
    header2 = "\\textbf{Model}"
    for _ in prompt_order:
        header2 += " & \\multicolumn{2}{c}{$\\exists$} & \\multicolumn{2}{c}{$\\Rightarrow$}"
    lines.append(header2 + r" \\")
    parts2 = []
    for i in range(n_strategies):
        start = 2 + i * 4
        end = 3 + i * 4
        parts2.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" " + " ".join(parts2))
    header3 = " "
    for _ in prompt_order:
        header3 += " & $S$ & $S_H$ & $S$ & $S_H$"
    lines.append(header3 + r" \\")
    lines.append(r"\midrule")

    # Sort by aggregate performance (mean of S, S_H) descending; ties broken
    # by model name for deterministic output. em-dash / early-stop / NaN count as 0.0.
    models_in_domain = sorted(
        sub["model"].unique(),
        key=lambda m: (-_agg_score_per_model(sub[sub["model"] == m]), m)
    )
    for model in models_in_domain:
        row_str = model
        for prompt in prompt_order:
            for qmode in ["existential", "implicational"]:
                mask = (sub["model"] == model) & (sub["prompt_mode"] == prompt) & (sub["question_mode"] == qmode)
                match = sub[mask]
                if not match.empty:
                    r = match.iloc[0]
                    if r.get("early_stop", False) or r.get("timeout", False):
                        S_val, SH_val = "ES", "ES"
                    else:
                        S_val = r.get("S", "---")
                        SH_val = r.get("S_H", "---")
                    row_str += f" & {_dash_score(S_val)} & {_dash_score(SH_val)}"
                else:
                    row_str += " & — & —"
        lines.append(row_str + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Comparative similarity scores for the \textit{" + label + r"} context.}")
    lines.append(r"\label{tab:" + context_name + r"_comparative}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _size_bin(size: float) -> str:
    if size <= 4:
        return "Small ($\\leq$4B)"
    if size <= 14:
        return "Medium (5-14B)"
    return "Large ($\\geq$15B)"


def generate_complexity_table(df: pd.DataFrame, df_sizes: Optional[pd.DataFrame] = None) -> str:
    """Complexity and scalability statistics from the AE run metrics.

    For each formal context, reports:
        - mean and median `wallclock_seconds`
        - mean `n_iterations` (pseudo-intent steps)
        - mean `n_retries_total` and rate of `n_retry_exhausted > 0`
        - rate of `early_stop=True`

    Optional `df_sizes` adds a per-model-size-bin breakdown.
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Context} & \textbf{$\mu_t$ (s)} & \textbf{$\\tilde{t}$ (s)} & "
                 r"\textbf{$\\mu_{iter}$} & \textbf{$\\mu_{retries}$} & \textbf{ES-rate} \\ \midrule")

    for ctx in ["integers", "graphs", "rcc", "top"]:
        sub = df[df["context_name"] == ctx]
        if sub.empty:
            continue
        n = len(sub)
        t_mean = sub["wallclock_seconds"].mean()
        t_median = sub["wallclock_seconds"].median()
        iter_mean = sub["n_iterations"].mean()
        retries_mean = sub["n_retries_total"].mean()
        es_rate = sub["early_stop"].mean()
        label = DOMAIN_LABELS.get(ctx, ctx)
        lines.append(f"{label} & {t_mean:.2f} & {t_median:.2f} & "
                     f"{iter_mean:.1f} & {retries_mean:.2f} & {es_rate:.2f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Complexity and scalability statistics of the Attribute Exploration runs. "
                 r"$\\mu_t$: mean wallclock per run (s). $\\tilde{t}$: median. "
                 r"$\\mu_{iter}$: mean pseudo-intent steps. $\\mu_{retries}$: mean retries. "
                 r"ES-rate: fraction of runs that aborted early.}")
    lines.append(r"\label{tab:complexity}")
    lines.append(r"\end{table}")

    if df_sizes is not None and "size" in df_sizes.columns:
        if "size" not in df.columns:
            merged = df.merge(df_sizes[["model", "size"]], on="model", how="left")
        else:
            merged = df
        merged["size_bin"] = merged["size"].apply(_size_bin)
        lines.append("")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model size bin} & \textbf{N runs} & \textbf{$\\mu_t$ (s)} & "
                     r"\textbf{$\\mu_{iter}$} & \textbf{ES-rate} \\ \midrule")
        for bin_name in ["Small ($\\leq$4B)", "Medium (5-14B)", "Large ($\\geq$15B)"]:
            sub = merged[merged["size_bin"] == bin_name]
            if sub.empty:
                continue
            n = len(sub)
            t_mean = sub["wallclock_seconds"].mean()
            iter_mean = sub["n_iterations"].mean()
            es_rate = sub["early_stop"].mean()
            lines.append(f"{bin_name} & {n} & {t_mean:.2f} & {iter_mean:.1f} & {es_rate:.2f} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Complexity metrics grouped by model size bin. "
                     r"Large models ($\\geq$15B) include \\texttt{{qwen2.5:72b}}, \\texttt{{llama3.3:70b}}, "
                     r"\\texttt{{llama4:16x17b}}, \\texttt{{gpt-oss:120b}}, etc.}")
        lines.append(r"\label{tab:complexity_size}")
        lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def generate_aggregated_table(df: pd.DataFrame) -> str:
    sub = df.copy()
    contexts = ["graphs", "integers", "rcc", "top"]
    prompts = ["basic", "full_context", "stem_basis"]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"{\footnotesize")
    lines.append(r"\begin{tabular}{llcrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Context} & \textbf{Prompt} & \textbf{Rep.} & \textbf{$R_{ext}$} & \textbf{$\mu_S$} & \textbf{$\mu_{S_H}$} & \textbf{Valid/Total} \\ \midrule")

    first_ctx = True
    for ctx in contexts:
        ctx_data = sub[sub["context_name"] == ctx]
        first_prompt = True
        for prompt in prompts:
            prompt_data = ctx_data[ctx_data["prompt_mode"] == prompt]
            for qmode in ["existential", "implicational"]:
                data = prompt_data[prompt_data["question_mode"] == qmode]
                total = len(data)
                if total == 0:
                    continue
                valid = data.dropna(subset=["S"]).shape[0]
                rext = valid / total if total > 0 else 0.0
                valid_data = data.dropna(subset=["S"])
                mu_S = valid_data["S"].mean() if len(valid_data) > 0 else 0.0
                mu_SH = valid_data["S_H"].mean() if len(valid_data) > 0 else 0.0

                ctx_label = f"\\multirow{{{len(prompts)*2}}}{{*}}{{{DOMAIN_LABELS.get(ctx, ctx)}}}" if first_ctx and first_prompt and qmode == "existential" else ""
                prompt_label = f"\\multirow{{2}}{{*}}{{{PROMPT_LABELS.get(prompt, prompt)}}}" if first_prompt and qmode == "existential" else ""
                qmode_label = "$\\exists$" if qmode == "existential" else "$\\to$"
                row = f"{ctx_label} & {prompt_label} & {qmode_label} & {rext:.2f} & {mu_S:.2f} & {mu_SH:.2f} & {valid}/{total} \\\\"
                lines.append(row)
                if qmode == "implicational":
                    lines.append(r"\cline{2-7}")
                else:
                    first_prompt = False
            first_prompt = False
        first_ctx = False
        lines.append(r"\cline{1-6} \cline{2-6}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\caption{Aggregated performance statistics across all experimental contexts.}")
    lines.append(r"\label{tab:aggregated_stats}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"
