import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


def plot_performance_vs_size(
    df: pd.DataFrame,
    df_sizes: pd.DataFrame,
    context_name: str,
    prompt_mode: str,
    output_path: str,
    title: str = None
):
    sub = df[(df["context_name"] == context_name) & (df["prompt_mode"] == prompt_mode)].copy()
    if sub.empty:
        print(f"  No data for {context_name}/{prompt_mode}")
        return

    df_plot = sub[["model", "question_mode", "S"]].copy()
    df_plot = df_plot.dropna(subset=["S"])
    df_plot = df_plot[~df_plot["S"].astype(str).str.contains("ES|T")]
    df_plot["S"] = pd.to_numeric(df_plot["S"], errors="coerce")
    df_plot = df_plot.dropna(subset=["S"])
    df_final = pd.merge(df_plot, df_sizes, on="model", how="inner")
    if df_final.empty:
        print(f"  No valid scores for {context_name}/{prompt_mode}")
        return

    sns.set_style("white")
    plt.figure(figsize=(10, 6))
    palette = {"existential": "#D62728", "implicational": "#1F77B4"}
    ax = sns.scatterplot(
        data=df_final, x="size", y="S",
        hue="question_mode", palette=palette,
        s=100, alpha=0.8, edgecolor="k", linewidth=0.8
    )
    texts = []
    for _, row in df_final.iterrows():
        texts.append(ax.text(row["size"], row["S"], row["model"],
                             fontsize=8, color="#333333"))
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                    expand_points=(1.2, 1.3), force_text=(0.1, 0.2))

    x_vals = df_final["size"].values
    y_vals = df_final["S"].values
    logx = np.log(x_vals)
    slope, intercept, r_value, _, _ = linregress(logx, y_vals)
    r2 = r_value ** 2
    x_range = np.logspace(np.log10(0.5), np.log10(df_final["size"].max() * 1.1), 200)
    reg_line = ax.plot(x_range, slope * np.log(x_range) + intercept, linestyle="--", linewidth=1.5, color="gray", alpha=0.7)[0]

    plt.xscale("log")
    plt.xlim(1, df_final["size"].max() * 1.5)
    plt.ylim(-0.05, 1.05)
    plt.title(title or f"{context_name} - {prompt_mode}", fontsize=14, pad=15)
    plt.xlabel("Model Size (Billion parameters - Log Scale)", fontsize=11)
    plt.ylabel("Similarity Score ($S$)", fontsize=11)
    sns.despine(trim=False)
    plt.grid(axis="y", color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.grid(axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(reg_line)
    labels.append(f"$R^2={r2:.3f}$")
    plt.legend(handles=handles, labels=labels, title="Query Mode", loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_performance_plots(df: pd.DataFrame, df_sizes: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    configs = [
        ("integers", "basic", "Integer Numbers (Baseline)"),
        ("rcc", "basic", "Region Connection Calculus (Baseline)"),
        ("graphs", "basic", "Graph Theory (Baseline)"),
        ("top", "basic", "KR Ontology (Baseline)"),
    ]
    for ctx, prompt, title in configs:
        fname = {"integers": "integer_numbers", "rcc": "rcc", "graphs": "graphs", "top": "top_performance"}.get(ctx, ctx) + ".png"
        plot_performance_vs_size(
            df, df_sizes, ctx, prompt,
            os.path.join(output_dir, fname),
            title=title
        )


def plot_extraction_heatmap(df: pd.DataFrame, output_path: str):
    sub = df.copy()
    contexts = ["integers", "graphs", "rcc", "top"]
    prompts = ["basic", "full_context", "stem_basis"]
    heat_data = pd.DataFrame(index=contexts, columns=prompts, dtype=float)
    for ctx in contexts:
        for prompt in prompts:
            data = sub[(sub["context_name"] == ctx) & (sub["prompt_mode"] == prompt)]
            total = data["model"].nunique()
            valid = data.dropna(subset=["S"])["model"].nunique()
            heat_data.loc[ctx, prompt] = valid / total if total > 0 else 0.0
    heat_data = heat_data.astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Extraction Rate"})
    ax.set_xticklabels([PROMPT_LABELS.get(p.get_text(), p.get_text()) for p in ax.get_xticklabels()])
    ax.set_yticklabels([DOMAIN_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


DOMAIN_LABELS = {
    "integers": "Integers",
    "graphs": "Graphs",
    "rcc": "RCC",
    "top": "KR"
}
PROMPT_LABELS = {
    "basic": "Baseline",
    "full_context": "CFC",
    "stem_basis": "DG"
}
