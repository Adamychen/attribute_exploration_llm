# Data guide

This directory holds the output of the sweep reported in the paper:
24 models x 4 domains x 3 prompt strategies x 2 query modes = 576 configurations.
The sweep was executed twice with the same models, code version and decoding
settings (temperature 0, top-k 20, 4-hour limit per run):

- the *first execution* is recorded in `results.csv`;
- the *second execution* was run to record the iteration log of each run (needed
  to analyse early stops) and is recorded in `traces/`.

Language-model outputs are not deterministic, so the two executions differ (see
"Agreement between the two executions" below). Re-running the sweep is not
expected to reproduce either of them exactly.

```
experiments/data/
├── results.csv            raw: one row per run (learned context + stop flags)
├── traces/                raw: one JSON per run (full iteration log)
├── benchmark_results.csv  raw: MMLU / GSM8K accuracy per model
└── derived/               computed from the raw files, no LLM calls
    ├── run_summary.csv
    ├── conjectures.csv
    ├── reference_bases.json
    └── lattices/
```

`derived/` is regenerated with

```bash
python -m experiments.build_derived_data
```

## Run key

Every run is identified by `(model, context_name, prompt_mode, question_mode)`.
`context_name` is `integers`, `graphs`, `rcc` or `top`. `prompt_mode` is `basic`,
`full_context` or `stem_basis` (the Duquenne-Guigues basis condition; the name
is kept for compatibility with the committed files). `question_mode` is
`implicational` or `existential`. The derived files also carry
`run_id = model__context__prompt__qmode`, which is the trace file name without
`.json`. Model names `qwen3-coder:latest` and `phi4-mini:latest` are reported as
`qwen3-coder:30b` and `phi4-mini:3.8b`, as in the paper.

## Raw files

### `results.csv`

| column | meaning |
|---|---|
| `model`, `context_name`, `prompt_mode`, `question_mode` | run key |
| `ctx_objects` | JSON list: names of the objects of the context learned by the model |
| `ctx_attributes` | JSON list: attribute names |
| `ctx_relations` | JSON boolean matrix, one row per object, one column per attribute |
| `early_stop` | true if the run was aborted (retry bound reached or time limit) |
| `finished` | true if Attribute Exploration terminated normally |
| `timeout` | true if the wall-clock limit was reached |

For aborted runs the context columns hold a placeholder (`["'"]`, `["None"]`,
`[[true]]`) and carry no information. The same placeholder was written for finished
runs in which the model accepted the first implication `[] -> all attributes`
(46 runs in this file, 71 in `traces/`): such an answer says that every object has
every attribute, so the learned lattice has the single intent M. The code that
computes `S` and `S_H` (`llmfca.analysis.metrics.learned_context`) replaces the
placeholder by that context, a single object with all attributes; the explorer
writes that context directly since the fix in `AttributeExplorerGraph.update_state`.
`S` and `S_H` are not stored here; they are computed from the context columns
(see `derived/run_summary.csv`).

### `traces/<run_id>.json`

| field | meaning |
|---|---|
| `finished`, `early_stop`, `timeout`, `wallclock_seconds` | terminal state and duration |
| `n_iterations` | number of implications processed (including the terminal step) |
| `n_retries_total`, `n_retry_exhausted` | malformed-answer retries; times the retry bound was hit |
| `joint_entropy`, `mean_entropy` | entropy of the last model answer |
| `stem_basis` | implications the model accepted during exploration |
| `final_context` | learned context (`objects`, `properties`, `context`), `null` if the run was aborted |
| `iteration_log` | events per step: `find_next_implication` (the implication posed), `ask_llm`, `validate_logic`, `update_state` (`accept`, or `add_counterexample` with the object and its attribute vector) |

The log contains every implication the explorer posed, not only accepted ones.
A trace with few implications is a run that stopped early.

## Derived files

Support convention: `supp(A -> B) = |A'| / |G|`, the fraction of objects that
have every attribute of the premise `A`. An implication holds in a context when
every object having all of `A` also has all of `B`. An implication with
support 0 holds vacuously.

### `derived/run_summary.csv` (576 rows)

| column | source | meaning |
|---|---|---|
| run key, `run_id` | | |
| `outcome_results_csv`, `outcome_trace` | both | `finished`, `early_stop` or `timeout`, in the first and second execution |
| `sources_agree` | both | whether the two executions ended in the same state |
| `n_iterations`, `n_retries_total`, `n_retry_exhausted`, `wallclock_seconds`, `joint_entropy`, `mean_entropy` | trace | as in the trace |
| `n_conjectures`, `n_accepted`, `n_refuted`, `n_unresolved` | trace (second execution) | implications posed, and how each ended (see `conjectures.csv`) |
| `stem_basis_size` | trace | length of the trace's `stem_basis` |
| `accepted_at_initialization` | trace | the model accepted the first implication (`[] -> all attributes`); the explorer then stops at once; the sweep recorded a placeholder context and an empty basis (see `results.csv` above) |
| `n_ctx_objects`, `n_ctx_attributes`, `n_concepts_llm` | results.csv (first execution) | size of the learned context and of its concept lattice (empty if the run was aborted according to `results.csv`) |
| `n_concepts_ref` | reference | number of concepts of the reference lattice |
| `S`, `S_H` | results.csv (first execution) | similarity scores of the learned lattice against the reference (empty if aborted); same computation as `experiments.generate_paper_assets` |

### `derived/conjectures.csv` (one row per implication posed)

| column | meaning |
|---|---|
| run key, `run_id`, `step` | which run and which exploration step |
| `stage` | `initialization` (first implication) or `exploration` |
| `premise`, `conclusion` | JSON lists; `conclusion` lists the attributes added to the premise |
| `premise_size` | number of attributes in the premise |
| `outcome` | `accepted` (model agreed), `refuted` (model gave a counterexample), `unresolved` (no valid answer: retry bound or time limit) |
| `n_llm_calls`, `n_retries`, `retry_exhausted` | effort spent on this implication |
| `counterexample_object` | name given by the model, for refuted implications |
| `joint_entropy`, `mean_entropy` | entropy of the last answer for this implication |
| `support_count_ref`, `support_ref`, `holds_ref`, `n_counterexamples_ref` | support of the premise, whether the implication holds, and the number of counterexample objects, in the reference context of the domain |
| `support_count_final`, `support_final`, `holds_final`, `n_counterexamples_final` | the same in the model's final context (empty if the trace has no final context) |

### `derived/reference_bases.json`

The Duquenne-Guigues basis of each reference context
(`experiments/reference_contexts/`), computed with Ganter's NextClosure
algorithm, with the support of every implication (`support_count`, `support`)
and the context sizes. The implication count is 4 (integers), 8 (graphs),
23 (rcc) and 41 (top); for integers, graphs and rcc it coincides with the
exhaustive search implemented in `llmfca.attribute_explorer` (see
`tests/test_build_derived_data.py`).

### `derived/lattices/<run_id>.json`

For each run whose context is non-empty according to `results.csv`: `concepts`
(`id`, `extent`, `intent`) and `covers`, the covering relation as
`[lower_id, upper_id]` pairs.

## Agreement between the two executions

`run_summary.csv` reports the terminal state of each configuration in both
executions (`outcome_results_csv` for the first, `outcome_trace` for the second)
and `sources_agree` when they coincide.

- Terminal state: they agree in 435 of 576 configurations (76 %). 221
  configurations finished in the first execution and 181 in the second; 134
  finished in both.
- Where both finished, `S` is identical in 65 of the 134; the mean absolute
  difference is 0.14 for `S` and 0.16 for `S_H`.
- 71 traces show the model accepting the very first implication
  (`accepted_at_initialization = true`): the explorer then stops at once with
  `finished = true` (46 runs of the first execution do the same). Without these
  runs, 175 configurations finished in the first execution and 110 in the second.

The paper's tables and figures of similarity (`S`, `S_H`) use the first
execution (`results.csv`). The exploration statistics (iterations, retries,
early-stop analysis) use the second (`traces/`). Columns of `run_summary.csv`
that come from the first execution are `S`, `S_H`, `n_ctx_objects`,
`n_ctx_attributes`, `n_concepts_llm` and the lattices; all others describing the
exploration come from the second.
