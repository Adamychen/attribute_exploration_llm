# LLM-FCA: Attribute Exploration with LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)]()

A LangGraph state machine that drives Attribute Exploration (AE) over Formal
Concept Analysis (FCA) using open-source LLMs as stochastic oracles. The
explorer asks the model to validate each candidate implication; on a "no"
reply it demands a counterexample with a full attribute vector, and updates
the formal context accordingly. Bounded retries use a token-level entropy
signal to cap the number of malformed-response repairs.

This is the codebase that produced the results in the KBS submission
*Evaluating the Intent Dimension in Large Language Models through Attribute
Exploration* (Borrego-Díaz & Chen-Abolacio, 2026). The library
(`llmfca/`) is fully reusable in other FCA + LLM settings; the
`experiments/` package contains the concrete reproduction scripts
(model sweep, prompt ablations, MMLU/GSM8K benchmark correlation, LaTeX
table + figure generation).

## Layout

```
code/
├── llmfca/              # importable library (the public API)
│   ├── attribute_explorer.py
│   ├── entropy_analyzer.py
│   ├── fca_helpers.py
│   ├── extraction.py
│   ├── llm/client.py
│   ├── analysis/metrics.py
│   └── prompting/
├── experiments/         # paper-repro scripts (CLI + data + reference contexts)
│   ├── run_extraction.py
│   ├── generate_paper_assets.py
│   ├── check_endpoint.py
│   ├── models.csv
│   ├── reference_contexts/
│   ├── data/            # external benchmark inputs (benchmark_results.csv)
│   └── run_all.sh
├── tests/               # pytest suite
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Install

```bash
cd code/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env to add your LLAMUS_API_KEY
```

For just the library (no experiments):

```bash
pip install -e .
```

## Quick start (library)

```python
from langchain_core.messages import AIMessage
import json

from llmfca import (
    AttributeExplorerGraph,
    make_model,
    get_message_function,
)
from llmfca.prompting.prompts import (
    INTEGER_ATTRIBUTES, INTEGER_PROMPT, INTEGER_PARSER,
)

model = make_model(
    model_name="llama3.2:3b",
    base_url="https://llamus.cs.us.es/ollama/v1",
    api_key="<your-token>",
)
explorer = AttributeExplorerGraph(
    model=model,
    message_function=get_message_function("implicational", arity=1),
    system_prompt=INTEGER_PROMPT,
    attributes=INTEGER_ATTRIBUTES,
    parser=INTEGER_PARSER,
)
# explorer.build() returns a compiled LangGraph StateGraph ready to .ainvoke(initial_state).
```

## Reproducing the paper

The full pipeline is one command:

```bash
cd code/
bash experiments/run_all.sh                       # writes to ./output/
bash experiments/run_all.sh --output-dir ./build  # custom output dir
bash experiments/run_all.sh --skip-extract        # reuse existing results.csv
```

Or step by step:

```bash
cd code/

# (1) Quick sanity check: does the endpoint return logprobs?
python -m experiments.check_endpoint --model llama3.2:3b

# (2) Run the full sweep
python -m experiments.run_extraction \
    --contexts integers graphs rcc top \
    --prompts basic full_context stem_basis \
    --output ./output/results.csv \
    --traces-dir ./output/traces \
    --timeout 7200 \
    --concurrency 4

# (3) Regenerate LaTeX tables (4 domain + aggregated + complexity) and PNG figures
python -m experiments.generate_paper_assets \
    --results ./output/results.csv --output-dir ./output

# (4) Prompt-strategy ablation
python -m experiments.analysis_prompts \
    --results ./output/results.csv --output-dir ./output/tables

# (5) External benchmarks: real MMLU + GSM8K subsets via HuggingFace
python -m experiments.run_benchmarks \
    --output ./output/benchmark_results.csv --cache-dir ./output/_hf_cache
#     Smoke (no endpoint needed for the loader, just `--no-hf` for offline probe):
python -m experiments.run_benchmarks --no-hf
#     Smaller sample for quick iteration:
python -m experiments.run_benchmarks --n-mmlu-per-subject 1 --n-gsm8k 5

# (6) Correlation between FCA similarity and benchmark accuracy
python -m experiments.correlation \
    --results ./output/results.csv \
    --benchmarks ./output/benchmark_results.csv \
    --output ./output/tables/table_correlation.tex

# (7) Dump all system prompts to a LaTeX appendix
python -m experiments.dump_prompts --format tex \
    --output ./output/appendix_prompts.tex

# (8) Quantitative post-mortem of JSON traces
python -m experiments.analyze_traces \
    --traces-dir ./output/traces \
    --output ./output/tables/table_traces_summary.tex
```

The 7 steps write, in this order:
- `results.csv` (1 row per `(model, prompt, question_mode, context)` = 576 rows for a full sweep)
- `traces/<model>__<ctx>__<prompt>__<qmode>.json` (per-experiment log + final context)
- `benchmark_results.csv` (MMLU + GSM8K accuracy per model)
- `tables/table_{integers,rcc,graphs,top}.tex` + `table_aggregated.tex` + `table_complexity.tex`
- `tables/table_correlation.tex`
- `appendix_prompts.tex`
- `tables/table_traces_summary.tex`
- `figuras_rendimiento/<domain>_performance.png` + `extraction_heatmap.png` + `complexity_iter.png`

All paths above are defaults; every script accepts absolute paths to write
into a different directory.

## Library API

`llmfca.AttributeExplorerGraph` — the core state machine. Nodes: `find_next_implication → ask_llm → parse_response → validate_logic → update_state`. Errors are routed back to `ask_llm` for bounded retries. Each step appends a structured entry to `state['iteration_log']` for post-mortem analysis.

> **Note on `recursion_limit`**: LangGraph's default is 25 super-steps. Long AE runs (e.g. the `top` domain has 25 attributes → 24 pseudo-intent steps) exceed this and crash with `GraphRecursionError`. We pass `config={"recursion_limit": 100_000}` in `llmfca.extraction.run_explorer` to give AE effectively unlimited headroom. The runtime cap is still the per-experiment wallclock `--timeout` (default 2h).

`llmfca.EntropyAnalyzer` — Shannon entropy over the LLM logprobs, both joint (semantic tokens) and per-token (mean). Exposes `calculate_bound()` which returns `min(N_op, ceil(2^H*))` with `N_op=10` as the operational cap.

`llmfca.make_model(model_name, base_url, api_key)` — factory for an `OfficialOpenAIClient` (a thin wrapper around the official `openai` Python SDK) with `logprobs=True`, `top_logprobs=20`, `max_tokens=2000` by default. Works with any OpenAI-compatible endpoint (Ollama, vLLM, LlamUS, OpenAI). Pass `logprobs=False` for non-AE consumers (e.g. MMLU/GSM8K benchmarks) that don't need the entropy signal. The `run_benchmarks.py` script sets `logprobs=False, max_tokens=512` by default to keep MMLU/GSM8K inference cheap.

`llmfca.context_to_dict(ctx)` / `llmfca.get_message_function(qmode, arity)` — utilities for serialising formal contexts and formatting FOL queries.

`llmfca.analysis.metrics.compute_similarities(ctx_llm, ctx_ref)` — returns the `(S, S_H)` similarity pair used to score the LLM-discovered context against the canonical reference.

`llmfca.extraction.build_trace_payload(...)` — converts an AE run state into a JSON-serialisable trace (per-iteration log + final context + stem basis + cumulative metrics), persisted by `experiments.run_extraction` to `--traces-dir`.

## Adding a new domain

1. Append `CONTEXT_NAME` to `llmfca/extraction.py` (`CONTEXT_NAMES`, `ARITY`, `PROMPTS`, `ATTRIBUTES`, `PARSERS`).
2. Add the prompts, attributes, parser, and reference context under `llmfca/prompting/prompts.py` and `experiments/reference_contexts/<name>.json`.
3. Add the model column to `experiments/models.csv`.
4. Run `python -m experiments.run_extraction --contexts <name> --prompts basic`.

## Output schema

After `bash experiments/run_all.sh --output-dir ./output`:

- `output/results.csv` — main sweep results (576 rows when complete; one row per `(model, question_mode, context, prompt_mode)`). S and S_H are NOT stored; they are recomputed on the fly from the per-row `ctx_objects` / `ctx_attributes` / `ctx_relations` columns.
- `output/traces/` — per-experiment JSON traces containing the full `iteration_log`, `stem_basis`, `final_context` (objects, properties, bool matrix), and the cumulative AE metrics. Useful for post-mortem analysis of any single run.
- `output/benchmark_results.csv` — external benchmark accuracies from real MMLU + GSM8K (default: 285 MMLU questions stratified across 57 subjects + 100 GSM8K test problems, loaded from HuggingFace `cais/mmlu` and `openai/gsm8k`). Models unavailable at the endpoint get `NaN` (not 0).
- `output/_hf_cache/` — local HuggingFace dataset cache (gitignored).
- `output/tables/*.tex` and `output/figuras_rendimiento/*.png` — LaTeX tables and 300 dpi PNG figures ready to be `\input{}`'d into a paper.
- `output/appendix_prompts.tex` — verbatim listing of all system prompts (integer / graphs / rcc / top × basic / full_context / stem_basis + the shared FOL evaluation rules), ready for `\input{}` from a paper appendix.

## Testing

```bash
cd code/
python3 -m pytest tests/ -v
# or simply:
python3 tests/test_metrics.py
python3 tests/test_explorer.py
```

## License

MIT — see `LICENSE`.

## Citing

If you use this codebase in academic work, please cite the accompanying
paper:

> Borrego-Díaz, J. & Chen-Abolacio, A. L.-Y. (2026). *Evaluating the
> Intent Dimension in Large Language Models through Attribute
> Exploration*. Knowledge-Based Systems (under revision).
