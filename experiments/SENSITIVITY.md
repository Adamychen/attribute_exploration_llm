# Prompt-sensitivity analysis

Purpose: check whether the results of the main sweep change when the prompt is
presented differently while its content stays the same. It is a bounded,
exploratory design (a subset of the sweep), not a repetition of the whole sweep.

Code: `llmfca/prompting/variants.py` (variants), `experiments/run_sensitivity.sh`
(runs), `tests/test_prompt_variants.py` (tests). The prompts of the main sweep
are not modified; variants are registered as `<strategy>+<condition>`.

## Conditions

Each condition changes one block of the prompt (header, attribute descriptions,
evaluation rules, JSON-format instructions, strategy block).

| condition | change | strategies |
|---|---|---|
| `C0` | reference: the original prompt, with the only repair described below | all |
| `V1_order` | the strategy block moves before the evaluation rules and format instructions | `full_context`, `stem_basis` |
| `V2_format` | the strategy block is rewritten: the formal context as a table, the basis as plain-English sentences | `full_context`, `stem_basis` |
| `V3_wording` | header, evaluation rules and response instructions are reworded; the JSON schema text is unchanged | all |
| `V4_attrs` | attribute descriptions as a bulleted list instead of a printed Python dict | all |

`basic` has no strategy block, so `V1_order` and `V2_format` do not apply to it.

## Repair applied to the reference prompt

In the prompt of the main sweep for graphs with the Duquenne-Guigues basis
(`GRAPHS_PROMPT_STEM_BASIS`), the first rule contains a form-feed character
instead of `\f` in `\forall` (a `"\forall"` written in a non-raw Python string).
`C0` repairs it and is otherwise identical to the original prompt. For the other
five prompts of the two domains `C0` is the original prompt, i.e. a new run of the
same configuration.

## Design

- Models (7): `qwen2.5-coder:7b`, `llama3.1:8b`, `qwen2.5:14b`, `qwen3-coder:30b`,
  `qwen2.5:72b`, `llama4:16x17b`, `gpt-oss:20b`.
- Domains: `integers`, `graphs`. Query modes: `implicational`, `existential`.
- Part A: 7 models x 2 domains x 2 query modes x 13 strategy+condition names = 364 runs.
- Part B: `graphs` x `stem_basis+C0` for the other 17 models (34 runs), so that the
  repaired graph/basis cell exists for all 24 models.
- Decoding and limits as in the main sweep: temperature 0, 20 log-probabilities per
  token for the entropy, 4 hours per exploration.

Models were chosen to span 7 B to 108 B parameters among those that finish a
non-trivial share of explorations in these two domains in the main sweep, and for the
cost of a run; the main sweep's `rcc` and `top` domains and the 1 B to 4 B models
are left out because few runs finish there. Conclusions apply to that range only.

## Planned analysis

Reference for every comparison: `C0` on the same (model, domain, strategy, query mode).

- Primary outcome: whether the exploration finishes (exact sign test on discordant pairs).
- Secondary: `S` and `S_H` on pairs that finish under both conditions (Wilcoxon signed-rank).
- Noise reference: the same comparisons between `C0` and each of the two executions of
  the main sweep (`results.csv`, `traces/`), restricted to the same cells. A variant effect
  is interpreted against this run-to-run difference.
- Four variants are compared with `C0`; p-values are adjusted (Holm) and reported with
  effect sizes. The analysis is exploratory.

Power: in a simulation calibrated with the run-to-run agreement observed in the main
sweep (about 75 % of explorations end in the same state in the two executions), 72 pairs
detect a change of 0.20 in the probability of finishing about 72 % of the time and a
change of 0.10 about 25 % of the time. The design can reveal large sensitivity; it cannot
rule out small or moderate effects.

## Cost

Estimate from the wall-clock times of the second execution: about 10 to 13 hours with
`--concurrency 4`, plus the time of explorations that reach the 4-hour limit. Use
`bash experiments/run_sensitivity.sh --dry-run` to list the planned runs.
