#!/usr/bin/env bash
# Pipeline orchestrator for the LLM-FCA experiments.
#
# Runs the seven analysis steps end-to-end and writes all artefacts into the
# current working directory (or into --output-dir if you pass one). The
# pipeline is paper-agnostic: it does not assume any particular output
# directory layout beyond the `tables/` and `figuras_rendimiento/` folders
# created by experiments.generate_paper_assets.
#
# Usage:
#   bash experiments/run_all.sh                       # writes to ./output/
#   bash experiments/run_all.sh --output-dir ./build  # custom output dir
#   bash experiments/run_all.sh --skip-extract        # reuse existing results.csv
#
# Steps:
#   1. experiments.run_extraction        — AE sweep over the model × context × prompt grid
#   2. experiments.generate_paper_assets — regenerate tables (5) and figures (5 PNG)
#   3. experiments.analysis_prompts      — prompt-strategy ablation + Wilcoxon tests
#   4. experiments.run_benchmarks        — MMLU + GSM8K from HuggingFace
#   5. experiments.correlation           — FCA similarity vs benchmark accuracy
#   6. experiments.dump_prompts          — dump system prompts to LaTeX or Markdown
#   7. experiments.analyze_traces        — quantitative post-mortem of JSON traces
#
# Requires .env with LLAMUS_API_KEY and access to HuggingFace (the first run
# downloads MMLU/GSM8K to _hf_cache/, ~30 MB).

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="output"
SKIP_EXTRACT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --skip-extract) SKIP_EXTRACT="1"; shift ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

step() {
    local name="$1"
    local module="$2"
    shift 2
    local logfile="${LOG_DIR}/${name}.log"
    echo "=== $name ==="
    if python3 -c "import importlib.util; importlib.util.find_spec('${module}')" 2>/dev/null; then
        if python3 -m "${module}" "$@" 2>&1 | tee "$logfile"; then
            echo "✓ $name OK"
        else
            echo "✗ $name FAILED (see $logfile)"
            return 1
        fi
    else
        echo "⚠ ${module} not available, skipping."
    fi
}

RESULTS_CSV="${OUTPUT_DIR}/results.csv"
TRACES_DIR="${OUTPUT_DIR}/traces"
BENCH_CSV="${OUTPUT_DIR}/benchmark_results.csv"
ASSETS_DIR="${OUTPUT_DIR}"

# Step 1: main extraction (resumes, with availability probe)
if [[ -z "$SKIP_EXTRACT" ]]; then
    step "extract" "experiments.run_extraction" \
        --output "$RESULTS_CSV" --traces-dir "$TRACES_DIR"
else
    echo "=== extract: SKIPPED (--skip-extract) ==="
fi

# Step 2: regenerate assets (tables + figures, includes the complexity table)
step "assets" "experiments.generate_paper_assets" \
    --results "$RESULTS_CSV" --output-dir "$ASSETS_DIR"

# Step 2b: worked Attribute Exploration trace (Table 1 of the paper)
step "ae_trace" "experiments.render_ae_trace" \
    --output "${ASSETS_DIR}/tables/table_ae_trace.tex"

# Step 2c: prompt sensitivity (variants of the prompt vs. C0; design in SENSITIVITY.md)
step "sensitivity" "experiments.analyze_sensitivity" \
    --output-dir "${ASSETS_DIR}/tables"

# Step 3: prompt ablation (Wilcoxon)
step "prompts" "experiments.analysis_prompts" \
    --results "$RESULTS_CSV" --output-dir "${ASSETS_DIR}/tables"

# Step 4: external benchmarks (MMLU + GSM8K from HuggingFace)
step "benchmarks" "experiments.run_benchmarks" \
    --output "$BENCH_CSV" --cache-dir "${OUTPUT_DIR}/_hf_cache"

# Step 5: correlation between FCA similarity and benchmark accuracy
step "correlation" "experiments.correlation" \
    --results "$RESULTS_CSV" --benchmarks "$BENCH_CSV" \
    --output "${ASSETS_DIR}/tables/table_correlation.tex"

# Step 6: dump prompts (LaTeX by default; switch to --format md for Markdown)
step "prompts_dump" "experiments.dump_prompts" \
    --format tex --output "${ASSETS_DIR}/appendix_prompts.tex"

# Step 7: quantitative post-mortem of the JSON traces
step "traces" "experiments.analyze_traces" \
    --traces-dir "$TRACES_DIR" \
    --output "${ASSETS_DIR}/tables/table_traces_summary.tex"

echo
echo "Pipeline complete. Outputs in ${OUTPUT_DIR}/:"
echo "  - results.csv"
echo "  - traces/*.json"
echo "  - benchmark_results.csv"
echo "  - _hf_cache/  (HuggingFace datasets, gitignored)"
echo "  - tables/*.tex"
echo "  - appendix_prompts.tex"
echo "  - figuras_rendimiento/*.png"
