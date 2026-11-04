#!/usr/bin/env bash
# Prompt-sensitivity analysis (see experiments/SENSITIVITY.md).
#
# Runs the prompt variants of llmfca/prompting/variants.py on a subset of models,
# the integers and graphs domains, and both query modes. Needs LLAMUS_API_KEY in
# .env and many hours of endpoint time; use --dry-run first to see the plan.
#
# Usage:
#   bash experiments/run_sensitivity.sh --dry-run
#   bash experiments/run_sensitivity.sh                     # writes to ./output/sensitivity/
#   bash experiments/run_sensitivity.sh --output-dir DIR
#
# Part A: 7 models x {integers, graphs} x 13 strategy+condition names x 2 query modes (364 runs)
# Part B: graphs x stem_basis+C0 for the remaining 17 models (34 runs); with part A this repeats the
#         graphs x Duquenne-Guigues-basis cell for all 24 models with the repaired prompt.
# Same decoding settings and per-exploration limit as the main sweep (4 hours).

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="output/sensitivity"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) EXTRA+=(--dry-run); shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SUBSET=(qwen2.5-coder:7b llama3.1:8b qwen2.5:14b qwen3-coder:30b qwen2.5:72b llama4:16x17b gpt-oss:20b)
TIMEOUT=14400   # 4 hours per exploration

# run_extraction resolves relative paths against the parent of this repository, so use an absolute path
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;; esac
mkdir -p "$OUTPUT_DIR"

# Part A: all variants on the subset
python3 -m experiments.run_extraction \
    --contexts integers graphs \
    --models "${SUBSET[@]}" \
    --prompts basic+C0 basic+V3_wording basic+V4_attrs \
              full_context+C0 full_context+V1_order full_context+V2_format full_context+V3_wording full_context+V4_attrs \
              stem_basis+C0 stem_basis+V1_order stem_basis+V2_format stem_basis+V3_wording stem_basis+V4_attrs \
    --timeout "$TIMEOUT" --concurrency 4 \
    --output "$OUTPUT_DIR/results.csv" --traces-dir "$OUTPUT_DIR/traces" \
    ${EXTRA[@]+"${EXTRA[@]}"}

# Part B: repaired graphs x stem_basis prompt for the models not in the subset
OTHERS=$(python3 - "${SUBSET[@]}" <<'PY'
import sys, pandas as pd
subset = set(sys.argv[1:])
print(" ".join(m for m in pd.read_csv("experiments/models.csv")["model"] if m not in subset))
PY
)
# shellcheck disable=SC2086
python3 -m experiments.run_extraction \
    --contexts graphs --prompts stem_basis+C0 \
    --models $OTHERS \
    --timeout "$TIMEOUT" --concurrency 4 \
    --output "$OUTPUT_DIR/results.csv" --traces-dir "$OUTPUT_DIR/traces" \
    ${EXTRA[@]+"${EXTRA[@]}"}
