import os
import pandas as pd
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

    df_plot = sub.melt(
        id_vars=["model", "S", "S_H"],
        value_vars=[],
    ).drop(columns=["variable", "value"])
    df_plot = sub[["model", "S", "S_H"]].copy()
    df_plot = df_plot.melt(id_vars=["model"], var_name="metric", value_name="score")
    df_plot = df_plot.dropna(subset=["score"])
    df_plot = df_plot[~df_plot["score"].astype(str).str.contains("ES|T")]
    df_plot["score"] = pd.to_numeric(df_plot["score"], errors="coerce")
    df_plot = df_plot.dropna(subset=["score"])
    df_final = pd.merge(df_plot, df_sizes, on="model", how="inner")
    if df_final.empty:
        print(f"  No valid scores for {context_name}/{prompt_mode}")
        return

    sns.set_style("white")
    plt.figure(figsize=(10, 6))
    colors = {"S": "#D62728", "S_H": "#1F77B4"}
    ax = sns.scatterplot(
        data=df_final, x="size", y="score",
        hue="metric", palette=colors,
        s=100, alpha=0.8, edgecolor="k", linewidth=0.8
    )
    texts = []
    for _, row in df_final.iterrows():
        t = ax.text(row["size"], row["score"], row["model"],
                    fontsize=8, color="#333333")
        texts.append(t)
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                    expand_points=(1.2, 1.3), force_text=(0.1, 0.2))

    plt.xlim(-5, df_final["size"].max() * 1.1 + 5)
    plt.ylim(-0.05, 1.05)
    plt.title(title or f"{context_name} - {prompt_mode}", fontsize=14, pad=15)
    plt.xlabel("Model Size (Billion parameters)", fontsize=11)
    plt.ylabel("Similarity Score", fontsize=11)
    sns.despine(trim=False)
    plt.grid(axis="y", color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.grid(axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
    plt.legend(title="Metric", loc="lower right", frameon=True, framealpha=0.9)
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
        fname = f"{ctx}.png" if ctx != "top" else "top_performance.png"
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
            total = len(data)
            valid = data.dropna(subset=["S"]).shape[0]
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
    "stem_basis": "STEM"
}
