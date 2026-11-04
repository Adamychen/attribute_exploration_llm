"""Analyse the prompt-sensitivity runs (design in ``experiments/SENSITIVITY.md``).

Every comparison is paired by cell, a cell being (model, domain, strategy, query
mode). Each prompt variant is compared with ``C0`` on the cells where it applies.
The same statistics are computed between repetitions of the unchanged prompt
(``C0`` and the two executions of the main sweep), which gives the run-to-run
difference against which a variant effect is read. A separate comparison covers
the repair of the graph prompt with the Duquenne-Guigues basis (form feed in
``\\forall``), the only cells where ``C0`` differs from the main-sweep prompt.

Outcomes, per run:
  * ``finished``    Attribute Exploration terminated normally (primary outcome);
  * ``nontrivial``  finished without accepting the first implication
                    (``[] -> all attributes``); accepting it ends the exploration at
                    once, and the learned lattice has the single intent M;
  * ``S``, ``S_H``  similarity of the learned lattice to the reference lattice, for
                    finished runs, computed as for the paper's tables.

Tests: exact two-sided sign test on the pairs that differ in ``finished`` (or
``nontrivial``); Wilcoxon signed-rank test on ``S`` and ``S_H`` over the pairs that
finish in both runs (zero differences dropped). p-values of the four variants are
adjusted with Holm's method within each outcome. The repetition rows are not tested
against each other; they show the size of the differences between runs of the same
prompt.

Usage:
    python -m experiments.analyze_sensitivity --output-dir output/tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MODEL_RENAMES = {"qwen3-coder:latest": "qwen3-coder:30b", "phi4-mini:latest": "phi4-mini:3.8b"}
CELL = ["model", "context_name", "strategy", "question_mode"]
VARIANTS = ["V1_order", "V2_format", "V3_wording", "V4_attrs"]
VARIANT_LABELS = {
    "V1_order": "Strategy block first",
    "V2_format": "Strategy block reformatted",
    "V3_wording": "Instructions reworded",
    "V4_attrs": "Attributes as a list",
}
REPETITION_LABELS = {
    ("main1", "C0"): r"First execution vs.\ $C_0$",
    ("main2", "C0"): r"Second execution vs.\ $C_0$",
    ("main1", "main2"): r"First vs.\ second execution",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _reference_contexts(ref_dir: str):
    from concepts import Context
    return {os.path.splitext(os.path.basename(p))[0]: Context.fromjson(p)
            for p in sorted(glob.glob(os.path.join(ref_dir, "*.json")))}


def _scores(objects, attributes, matrix, ref) -> Tuple[float, float]:
    from llmfca.analysis.metrics import compute_similarities, learned_context
    return compute_similarities(learned_context(objects, attributes, matrix, ref.properties), ref)


def _record(key: dict, finished: bool, ctx: Optional[tuple], refs) -> dict:
    from llmfca.analysis.metrics import accepted_first_implication
    trivial = bool(finished and ctx is not None and accepted_first_implication(ctx[0], ctx[1]))
    S = S_H = None
    if finished and ctx is not None:
        S, S_H = _scores(*ctx, refs[key["context_name"]])
    return {**key, "finished": bool(finished), "nontrivial": bool(finished) and not trivial,
            "S": S, "S_H": S_H}


def _split_prompt_mode(prompt_mode: str) -> Tuple[str, str]:
    strategy, _, condition = prompt_mode.partition("+")
    return strategy, condition


def load_traces(traces_dir: str, refs, source: str) -> pd.DataFrame:
    """One row per trace. ``source`` names the execution (``sens`` or ``main2``)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
        strategy, condition = _split_prompt_mode(t["prompt_mode"])
        key = {"model": MODEL_RENAMES.get(t["model"], t["model"]),
               "context_name": t["context_name"], "strategy": strategy,
               "question_mode": t["question_mode"], "condition": condition or source}
        fc = t.get("final_context")
        ctx = (fc["objects"], fc["properties"], fc["context"]) if fc else None
        rows.append(_record(key, t["finished"], ctx, refs))
    return pd.DataFrame(rows)


def load_results_csv(path: str, refs, source: str = "main1") -> pd.DataFrame:
    """One row per run of ``results.csv`` (the first execution of the main sweep)."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        strategy, condition = _split_prompt_mode(r["prompt_mode"])
        key = {"model": MODEL_RENAMES.get(r["model"], r["model"]),
               "context_name": r["context_name"], "strategy": strategy,
               "question_mode": r["question_mode"], "condition": condition or source}
        finished = bool(r["finished"]) and not bool(r["timeout"])
        ctx = None
        if finished:
            ctx = (json.loads(r["ctx_objects"]), json.loads(r["ctx_attributes"]),
                   json.loads(r["ctx_relations"]))
        rows.append(_record(key, finished, ctx, refs))
    return pd.DataFrame(rows)


def load_all(data_dir: str = DATA, ref_dir: str = os.path.join(HERE, "reference_contexts")
             ) -> pd.DataFrame:
    refs = _reference_contexts(ref_dir)
    return pd.concat([
        load_traces(os.path.join(data_dir, "sensitivity", "traces"), refs, "sens"),
        load_results_csv(os.path.join(data_dir, "results.csv"), refs, "main1"),
        load_traces(os.path.join(data_dir, "traces"), refs, "main2"),
    ], ignore_index=True)


# --------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------

def paired(runs: pd.DataFrame, ref: str, cmp: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Join the runs of conditions ``ref`` and ``cmp`` on the given cells."""
    a = runs[runs.condition == ref].drop(columns="condition")
    b = runs[runs.condition == cmp].drop(columns="condition")
    out = cells[CELL].merge(a, on=CELL).merge(b, on=CELL, suffixes=("_ref", "_cmp"))
    if len(out) != len(cells):
        raise ValueError(f"{ref} vs {cmp}: {len(out)} pairs for {len(cells)} cells")
    return out


def sign_test(ref: pd.Series, cmp: pd.Series) -> dict:
    lost = int((ref & ~cmp).sum())    # finishes under the reference only
    gained = int((~ref & cmp).sum())  # finishes under the comparison only
    n = len(ref)
    p = binomtest(gained, lost + gained, 0.5).pvalue if lost + gained else 1.0
    return {"n": n, "k_ref": int(ref.sum()), "k_cmp": int(cmp.sum()),
            "lost": lost, "gained": gained, "agree": 1 - (lost + gained) / n,
            "delta": (gained - lost) / n, "p": p}


def signed_rank(ref: pd.Series, cmp: pd.Series) -> dict:
    d = (cmp - ref).astype(float)
    nz = d[d != 0]
    out = {"n": len(d), "n_equal": int((d == 0).sum()),
           "mean_delta": float(d.mean()) if len(d) else float("nan"),
           "mean_abs_delta": float(d.abs().mean()) if len(d) else float("nan"),
           "r_rb": float("nan"), "p": float("nan")}
    if len(nz):
        ranks = nz.abs().rank()
        w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        out["r_rb"] = float((w_pos - w_neg) / (w_pos + w_neg))
        out["p"] = float(wilcoxon(nz, zero_method="wilcox").pvalue)
    return out


def compare(pairs: pd.DataFrame) -> dict:
    both = pairs[pairs.finished_ref & pairs.finished_cmp]
    return {
        "finished": sign_test(pairs.finished_ref, pairs.finished_cmp),
        "nontrivial": sign_test(pairs.nontrivial_ref, pairs.nontrivial_cmp),
        "S": signed_rank(both.S_ref, both.S_cmp),
        "S_H": signed_rank(both.S_H_ref, both.S_H_cmp),
    }


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjustment; NaN p-values are left as NaN and not counted."""
    idx = [i for i, p in enumerate(pvalues) if p == p]
    order = sorted(idx, key=lambda i: pvalues[i])
    m, adjusted, running = len(order), [float("nan")] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def cells_of(runs: pd.DataFrame, condition: str) -> pd.DataFrame:
    return runs[runs.condition == condition][CELL].drop_duplicates().reset_index(drop=True)


def is_repaired(cells: pd.DataFrame) -> pd.Series:
    return (cells.context_name == "graphs") & (cells.strategy == "stem_basis")


def analyse(runs: pd.DataFrame) -> Dict[str, Dict[str, dict]]:
    """Return ``{"variants": ..., "repetitions": ..., "repair": ...}``, each a dict of rows."""
    variants = {}
    for v in VARIANTS:
        variants[v] = compare(paired(runs, "C0", v, cells_of(runs, v)))
    for outcome in ("finished", "nontrivial", "S", "S_H"):
        adj = holm([variants[v][outcome]["p"] for v in VARIANTS])
        for v, p in zip(VARIANTS, adj):
            variants[v][outcome]["p_holm"] = p

    # Cells of the sensitivity models where C0 is the main-sweep prompt.
    sens_cells = runs[runs.condition.isin(VARIANTS)][CELL].drop_duplicates()
    same = sens_cells[~is_repaired(sens_cells)].reset_index(drop=True)
    repetitions = {pair: compare(paired(runs, *pair, same)) for pair in REPETITION_LABELS}

    c0 = cells_of(runs, "C0")
    fixed = c0[is_repaired(c0)].reset_index(drop=True)
    repair = {pair: compare(paired(runs, *pair, fixed)) for pair in REPETITION_LABELS}
    return {"variants": variants, "repetitions": repetitions, "repair": repair}


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def _p(p: float) -> str:
    if p != p:
        return "--"
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def _signed(x: float, digits: int = 2) -> str:
    if x != x:
        return "--"
    s = f"{x:+.{digits}f}"
    return "$0.00$" if s in ("+0.00", "-0.00") else f"${s}$"


def _finish_cells(r: dict, tested: bool) -> List[str]:
    return [f"${r['k_ref']}/{r['k_cmp']}$", f"${r['lost']}/{r['gained']}$",
            f"${100 * r['agree']:.0f}\\%$", _signed(r["delta"]),
            _p(r["p_holm"]) if tested else "--"]


def _score_cells(r: dict, tested: bool) -> List[str]:
    return [f"${r['n']}$", f"${r['n_equal']}$", f"${r['mean_abs_delta']:.2f}$",
            _signed(r["mean_delta"]), _p(r["p_holm"]) if tested else "--"]


def _rows(results: Dict, labels: Dict, tested: bool, kind: str) -> List[str]:
    out = []
    for key, label in labels.items():
        r = results[key]
        if kind == "finish":
            cells = [f"${r['finished']['n']}$"] + _finish_cells(r["finished"], tested) \
                + _finish_cells(r["nontrivial"], tested)[1:]
        else:
            cells = _score_cells(r["S"], tested) + _score_cells(r["S_H"], tested)[2:]
        out.append(" & ".join([label] + cells) + r" \\")
    return out


def table_finishing(res: dict) -> str:
    head = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\hline",
        r" & & \multicolumn{5}{c}{\textit{Finished}} & \multicolumn{4}{c}{\textit{Finished, first implication refuted}} \\",
        r"\textit{Comparison} & $n$ & \textit{ref/cmp} & \textit{lost/gained} & \textit{agree} & $\Delta$ & $p_{\text{Holm}}$"
        r" & \textit{lost/gained} & \textit{agree} & $\Delta$ & $p_{\text{Holm}}$ \\",
        r"\hline",
        r"\multicolumn{11}{l}{\textit{Prompt variant vs.\ $C_0$}} \\",
    ]
    body = _rows(res["variants"], {v: VARIANT_LABELS[v] for v in VARIANTS}, True, "finish")
    body += [r"\hline", r"\multicolumn{11}{l}{\textit{Repetitions of the same prompt}} \\"]
    body += _rows(res["repetitions"], REPETITION_LABELS, False, "finish")
    body += [r"\hline", r"\multicolumn{11}{l}{\textit{Graphs, Duquenne--Guigues basis: original vs.\ repaired prompt ($C_0$)}} \\"]
    body += _rows(res["repair"], REPETITION_LABELS, False, "finish")
    tail = [
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\caption{Prompt sensitivity: whether the exploration finishes. Each comparison pairs the runs "
        r"of two conditions on the same (model, domain, strategy, query mode); $n$ is the number of pairs. "
        r"\textit{ref/cmp}: runs that finish under the first and the second condition; "
        r"\textit{lost/gained}: pairs that finish under the first condition only, and under the second only; "
        r"\textit{agree}: pairs that end in the same state; $\Delta$: change in the proportion of runs that finish. "
        r"The right block counts as finished only the runs in which the model refutes the first implication "
        r"$\emptyset \to M$; accepting it ends the exploration at once with a one-object context. "
        r"$p_{\text{Holm}}$: exact sign test on the discordant pairs, Holm-adjusted over the four variants. "
        r"Repetitions are computed on the cells of the variant comparisons where $C_0$ is the prompt of the main sweep; "
        r"the last block covers the graph cells with the Duquenne--Guigues basis for the 24 models.}",
        r"\label{tab:sensitivity_finish}",
        r"\end{table}",
        "",
    ]
    return "\n".join(head + body + tail)


def table_similarity(res: dict) -> str:
    head = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r" & & & \multicolumn{3}{c}{$S$} & \multicolumn{3}{c}{$S_H$} \\",
        r"\textit{Comparison} & $n$ & \textit{equal} & $|\Delta|$ & $\Delta$ & $p_{\text{Holm}}$"
        r" & $|\Delta|$ & $\Delta$ & $p_{\text{Holm}}$ \\",
        r"\hline",
        r"\multicolumn{9}{l}{\textit{Prompt variant vs.\ $C_0$}} \\",
    ]
    body = _rows(res["variants"], {v: VARIANT_LABELS[v] for v in VARIANTS}, True, "score")
    body += [r"\hline", r"\multicolumn{9}{l}{\textit{Repetitions of the same prompt}} \\"]
    body += _rows(res["repetitions"], REPETITION_LABELS, False, "score")
    body += [r"\hline", r"\multicolumn{9}{l}{\textit{Graphs, Duquenne--Guigues basis: original vs.\ repaired prompt ($C_0$)}} \\"]
    body += _rows(res["repair"], REPETITION_LABELS, False, "score")
    tail = [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Prompt sensitivity: similarity of the learned lattice to the reference, over the pairs of "
        r"Table~\ref{tab:sensitivity_finish} that finish under both conditions ($n$). "
        r"\textit{equal}: pairs with the same $S$; $|\Delta|$ and $\Delta$: mean absolute and mean signed "
        r"difference (second condition minus first). $p_{\text{Holm}}$: Wilcoxon signed-rank test, zero "
        r"differences dropped, Holm-adjusted over the four variants.}",
        r"\label{tab:sensitivity_similarity}",
        r"\end{table}",
        "",
    ]
    return "\n".join(head + body + tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--output-dir", default="output/tables")
    args = ap.parse_args()
    runs = load_all(args.data_dir)
    res = analyse(runs)
    os.makedirs(args.output_dir, exist_ok=True)
    for name, text in (("table_sensitivity_finish.tex", table_finishing(res)),
                       ("table_sensitivity_similarity.tex", table_similarity(res))):
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {os.path.join(args.output_dir, name)}")


if __name__ == "__main__":
    main()
