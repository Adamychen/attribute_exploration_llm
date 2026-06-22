"""
Generate all paper assets (tables and figures) from the results CSV.

This script reads the FCA sweep results (--results) and writes:
  * `tables/table_<domain>.tex`     — per-domain comparative tables (4 files)
  * `tables/table_aggregated.tex`   — per-(domain, prompt, query) aggregate stats
  * `tables/table_complexity.tex`   — per-context and per-size-bin complexity
  * `figuras_rendimiento/*.png`     — per-domain performance + complexity heatmap

Use it as a one-shot regeneration step after running the sweep and the
external benchmarks. The script is paper-agnostic: it does not assume
any particular output directory layout, only the conventions used by
the rest of the experiments/ package.

Usage:
    python -m experiments.generate_paper_assets
    python -m experiments.generate_paper_assets --results results.csv --output-dir ./out
"""
import os
import sys
import json
import argparse
import pandas as pd
from concepts import Context

current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(current_dir)
sys.path.insert(0, code_dir)

from llmfca.analysis.metrics import compute_similarities
from experiments.tables import generate_domain_table, generate_aggregated_table, generate_complexity_table
from experiments.figures import generate_all_performance_plots, plot_extraction_heatmap


def build_context_from_row(row) -> Context:
    objs = json.loads(row["ctx_objects"]) if isinstance(row["ctx_objects"], str) else row["ctx_objects"]
    attrs = json.loads(row["ctx_attributes"]) if isinstance(row["ctx_attributes"], str) else row["ctx_attributes"]
    rels = json.loads(row["ctx_relations"]) if isinstance(row["ctx_relations"], str) else row["ctx_relations"]
    return Context(objs, attrs, rels)


def load_reference_contexts(ref_dir: str):
    contexts = {}
    for fname in os.listdir(ref_dir):
        if fname.endswith(".json"):
            name = os.path.splitext(fname)[0]
            ctx = Context.fromjson(os.path.join(ref_dir, fname))
            contexts[name] = ctx
    return contexts


def main():
    parser = argparse.ArgumentParser(description="Generate paper assets from experiment results")
    parser.add_argument("--results", default="results.csv",
                        help="Path to the results CSV (one row per (model, prompt, "
                             "question_mode, context)). Relative paths resolve "
                             "against the current working directory.")
    parser.add_argument("--models", default=os.path.join(current_dir, "models.csv"),
                        help="Path to models.csv (lives inside this package).")
    parser.add_argument("--ref-dir", default=os.path.join(current_dir, "reference_contexts"),
                        help="Path to reference contexts directory (lives inside this package).")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory. The script creates `tables/` "
                             "and `figuras_rendimiento/` underneath it.")
    args = parser.parse_args()

    print("Loading results...")
    df = pd.read_csv(args.results)
    df_models = pd.read_csv(args.models)

    reemplazos = {
        "qwen3-coder:latest": "qwen3-coder:30b",
        "phi4-mini:latest": "phi4-mini:3.8b"
    }
    df["model"] = df["model"].replace(reemplazos)
    df_models["model"] = df_models["model"].replace(reemplazos)
    df = pd.merge(df, df_models, on="model", how="left")

    print("Loading reference contexts...")
    ref_contexts = load_reference_contexts(args.ref_dir)

    print("Computing similarity metrics (S and S_H) for all experiments...")
    S_list, SH_list = [], []
    for idx, row in df.iterrows():
        ctx_name = row["context_name"]
        ref = ref_contexts.get(ctx_name)
        if ref is None:
            S_list.append(None)
            SH_list.append(None)
            continue
        if row.get("early_stop", False) or row.get("timeout", False):
            S_list.append(None)
            SH_list.append(None)
            continue
        try:
            ctx_llm = build_context_from_row(row)
            S, SH = compute_similarities(ctx_llm, ref)
            S_list.append(S)
            SH_list.append(SH)
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            S_list.append(None)
            SH_list.append(None)
    df["S"] = S_list
    df["S_H"] = SH_list

    output_dir = os.path.abspath(args.output_dir)
    tables_dir = os.path.join(output_dir, "tables")
    figuras_dir = os.path.join(output_dir, "figuras_rendimiento")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figuras_dir, exist_ok=True)

    print("Generating LaTeX tables...")
    for ctx_name in ["integers", "graphs", "rcc", "top"]:
        if ctx_name in df["context_name"].unique():
            table_tex = generate_domain_table(df, ctx_name)
            table_path = os.path.join(tables_dir, f"table_{ctx_name}.tex")
            with open(table_path, "w") as f:
                f.write(table_tex)
            print(f"  Saved: {table_path}")

    agg_tex = generate_aggregated_table(df)
    agg_path = os.path.join(tables_dir, "table_aggregated.tex")
    with open(agg_path, "w") as f:
        f.write(agg_tex)
    print(f"  Saved: {agg_path}")

    print("Generating complexity table (R1.8)...")
    complexity_tex = generate_complexity_table(df, df_models)
    complexity_path = os.path.join(tables_dir, "table_complexity.tex")
    with open(complexity_path, "w") as f:
        f.write(complexity_tex)
    print(f"  Saved: {complexity_path}")

    print("Generating performance figures...")
    generate_all_performance_plots(df, df_models, figuras_dir)

    print("Generating extraction heatmap...")
    heatmap_path = os.path.join(figuras_dir, "extraction_heatmap.png")
    plot_extraction_heatmap(df, heatmap_path)

    print("All assets generated successfully.")


if __name__ == "__main__":
    main()
