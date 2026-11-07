"""Tables and figure of the Results and Discussion that slice the main sweep.

Every number comes from the raw runs, through the loaders of
``experiments.analyze_sensitivity`` (same S and S_H computation as the paper's
other tables, including the scoring of runs that accept [] -> M on the lattice with
the single intent M). Integer arithmetic is reported as the control domain, apart
from the three case studies.

Outputs (written to --output-dir):
  * ``table_replication.tex``     both executions of the sweep, per domain and strategy
  * ``table_query_asymmetry.tex`` existential vs implicational queries, Baseline
  * ``table_cfc_gains.tex``       Baseline vs CFC, existential queries
  * ``table_dg_graphs.tex``       the three strategies on graphs, existential queries
  * ``fig_tradeoff_scatter.tex``  PGFPlots figure: extraction rate vs mean S
  * ``table_domains.tex``         size of the reference context, lattice and basis of each domain

Rates are over the 24 models; means of S and S_H are over the runs that finished,
as in ``table_aggregated.tex``. The first execution (``results.csv``) is the source
of all outputs except the second column group of ``table_replication.tex``.

Usage:
    python -m experiments.discussion_tables --output-dir output/tables
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from experiments.analyze_sensitivity import CELL, DATA, HERE, _reference_contexts, \
    load_results_csv, load_traces

CASE_STUDIES = ["graphs", "rcc", "top"]
CONTROL = "integers"
LABELS = {"integers": "Integers", "graphs": "Graphs", "rcc": "RCC", "top": "KR"}
STRATEGIES = ["basic", "full_context", "stem_basis"]
STRATEGY_LABELS = {"basic": "Baseline", "full_context": "CFC", "stem_basis": "DG Basis"}


def load_executions(data_dir: str = DATA,
                    ref_dir: str = os.path.join(HERE, "reference_contexts"),
                    rerun_dir: str | None = None) -> Dict[str, pd.DataFrame]:
    refs = _reference_contexts(ref_dir)
    ex = {
        "first": load_results_csv(os.path.join(data_dir, "results.csv"), refs, "main1"),
        "second": load_traces(os.path.join(data_dir, "traces"), refs, "main2"),
    }
    if rerun_dir is not None:
        ex["rerun"] = load_traces(os.path.join(rerun_dir, "traces"), refs, "rerun2")
    return ex


def summary(runs: pd.DataFrame) -> dict:
    """Rate of finished runs, runs that accepted [] -> M, and mean scores."""
    fin = runs[runs.finished]
    nt = runs[runs.nontrivial]
    return {
        "n": len(runs),
        "rate": runs.finished.mean(),
        "k": int(runs.finished.sum()),
        "accepted_first": int((runs.finished & ~runs.nontrivial).sum()),
        "S": fin.S.mean(), "S_H": fin.S_H.mean(),
        "S_nt": nt.S.mean(), "S_H_nt": nt.S_H.mean(),
    }


def _select(runs, context, strategy=None, qmode=None):
    m = runs.context_name == context
    if strategy is not None:
        m &= runs.strategy == strategy
    if qmode is not None:
        m &= runs.question_mode == qmode
    return runs[m]


def _f(x: float) -> str:
    return "--" if x != x else f"{x:.2f}"


def _d(x: float) -> str:
    if x != x:
        return "--"
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def _domain_blocks() -> List[tuple]:
    """(heading, domains): the three case studies, then the control domain."""
    return [("Case studies", CASE_STUDIES), ("Control domain", [CONTROL])]


def _state(r: pd.Series) -> str:
    if not r.finished:
        return "stopped"
    return "finished" if r.nontrivial else "accepted_first"


# --------------------------------------------------------------------------
# Replication
# --------------------------------------------------------------------------

def table_replication(ex: Dict[str, pd.DataFrame]) -> str:
    """Per domain and strategy: first / second execution and, if present, the repetition."""
    a = ex["first"].set_index(CELL)
    b = ex["second"].set_index(CELL)
    s1 = a.apply(_state, axis=1)
    same12 = (s1 == b.loc[a.index].apply(_state, axis=1))
    runs = [ex["first"], ex["second"]]
    rep = ex.get("rerun")
    if rep is not None:
        runs.append(rep)
        same13 = (s1 == rep.set_index(CELL).loc[a.index].apply(_state, axis=1))
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"\textit{Domain} & \textit{Strategy} & \textit{Finished} & \textit{Accepted} $\emptyset \to M$"
        r" & $\mu_S$ & $\mu_{S_H}$ & \textit{Same state} \\",
        r"\hline",
    ]
    for heading, domains in _domain_blocks():
        lines.append(r"\multicolumn{7}{l}{\textit{" + heading + r"}} \\")
        for d in domains:
            for i, s in enumerate(STRATEGIES):
                sums = [summary(_select(df, d, s)) for df in runs]
                idx = [k for k in a.index if k[1] == d and k[2] == s]
                same = f"${100 * same12.loc[idx].mean():.0f}\\%$"
                if rep is not None:
                    same += f" / ${100 * same13.loc[idx].mean():.0f}\\%$"
                lines.append(
                    f"{LABELS[d] if i == 0 else ''} & {STRATEGY_LABELS[s]} & "
                    + " / ".join(f"${x['k']}$" for x in sums) + " & "
                    + " / ".join(f"${x['accepted_first']}$" for x in sums) + " & "
                    + " / ".join(_f(x['S_nt']) for x in sums) + " & "
                    + " / ".join(_f(x['S_H_nt']) for x in sums) + " & "
                    + same + r" \\")
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}%",
        r"}",
        r"\caption{The executions of the sweep, per domain and strategy: first / second"
        + (r" / repetition" if rep is not None else "") + r". Each cell pools the 24 models and the two query "
        r"modes (48 runs). \textit{Finished}: runs in which Attribute Exploration terminated. "
        r"\textit{Accepted $\emptyset \to M$}: finished runs that ended by accepting the first implication, "
        r"so that every object has every attribute. $\mu_S$, $\mu_{S_H}$: mean over the other finished runs. "
        r"\textit{Same state}: configurations that end in the same state in the first and the second execution"
        + (r", and, after the slash, in the first execution and the repetition" if rep is not None else "")
        + r" (stopped, accepted $\emptyset \to M$, or finished otherwise).}",
        r"\label{tab:replication}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_query_asymmetry(runs: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textit{Domain} & $\exists$ \textit{rate} & $\to$ \textit{rate} & \textit{Difference}"
        r" & $\mu_S$ ($\exists$ vs $\to$) & $\mu_{S_H}$ ($\exists$ vs $\to$) \\",
        r"\midrule",
    ]
    for heading, domains in _domain_blocks():
        for d in domains:
            e = summary(_select(runs, d, "basic", "existential"))
            i = summary(_select(runs, d, "basic", "implicational"))
            label = LABELS[d] + (" (control)" if d == CONTROL else "")
            lines.append(f"{label} & {_f(e['rate'])} & {_f(i['rate'])} & {_d(e['rate'] - i['rate'])} & "
                         f"{_f(e['S'])} vs {_f(i['S'])} & {_f(e['S_H'])} vs {_f(i['S_H'])} \\\\")
        if heading == "Case studies":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Existential ($\exists$) and implicational ($\to$) queries under the Baseline strategy "
        r"(first execution). \textit{Rate}: proportion of the 24 models whose exploration finished. "
        r"\textit{Difference}: $\exists$ rate minus $\to$ rate. $\mu_S$, $\mu_{S_H}$: mean over the "
        r"finished runs.}",
        r"\label{tab:query_asymmetry}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_cfc_gains(runs: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textit{Domain} & \textit{Base} $\mu_S$ & \textit{Base} $\mu_{S_H}$ & \textit{CFC} $\mu_S$"
        r" & \textit{CFC} $\mu_{S_H}$ & $\Delta S$ & $\Delta S_H$ \\",
        r"\midrule",
    ]
    for heading, domains in _domain_blocks():
        for d in domains:
            b = summary(_select(runs, d, "basic", "existential"))
            c = summary(_select(runs, d, "full_context", "existential"))
            label = LABELS[d] + (" (control)" if d == CONTROL else "")
            lines.append(f"{label} & {_f(b['S'])} & {_f(b['S_H'])} & {_f(c['S'])} & {_f(c['S_H'])} & "
                         f"{_d(c['S'] - b['S'])} & {_d(c['S_H'] - b['S_H'])} \\\\")
        if heading == "Case studies":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Baseline and CFC strategies under existential queries (first execution): mean $S$ and "
        r"$S_H$ over the runs that finished, and their differences $\Delta = \text{CFC} - \text{Baseline}$. "
        r"The means are over different sets of models, those that finished under each strategy.}",
        r"\label{tab:cfc_gains}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_dg_graphs(runs: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textit{Strategy} & \textit{Extraction rate} ($\exists$) & $\mu_S$ & $\mu_{S_H}$ \\",
        r"\midrule",
    ]
    for s in STRATEGIES:
        r = summary(_select(runs, "graphs", s, "existential"))
        lines.append(f"{STRATEGY_LABELS[s]} & {_f(r['rate'])} & {_f(r['S'])} & {_f(r['S_H'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{The three strategies in the Graphs domain under existential queries (first execution): "
        r"proportion of the 24 models whose exploration finished, and mean $S$ and $S_H$ over those runs.}",
        r"\label{tab:stem_graphs}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


_MARKS = {
    "basic": ("mark=o, mark size=3pt, blue!80!black, fill=blue!20", "Baseline"),
    "full_context": ("mark=square*, mark size=3pt, red!80!black, fill=red!20", "CFC"),
    "stem_basis": ("mark=triangle*, mark size=4pt, green!60!black, fill=green!20", "DG Basis"),
}
_LINE_COLORS = {"integers": "blue!50", "graphs": "orange!70", "rcc": "purple!50", "top": "brown!60"}
_TEXT_COLORS = {"integers": "blue!80!black", "graphs": "orange!80!black",
                "rcc": "purple!70!black", "top": "brown!70!black"}


def fig_tradeoff_scatter(runs: pd.DataFrame) -> str:
    domains = CASE_STUDIES + [CONTROL]
    pts = {(d, s): summary(_select(runs, d, s, "existential")) for d in domains for s in STRATEGIES}
    xy = {k: (v["rate"], v["S"]) for k, v in pts.items()}
    c = lambda p: f"({p[0]:.2f},{p[1]:.2f})"
    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.7\textwidth, height=0.5\textwidth,",
        r"    xlabel={Extraction rate}, ylabel={Mean similarity $\mu_S$},",
        r"    xmin=0, xmax=1.0, ymin=0, ymax=1.0,",
        r"    grid=major, grid style={gray!30},",
        r"    legend pos=south east,",
        r"    legend style={font=\scriptsize, draw=black, fill=white, fill opacity=0.8, text opacity=1},",
        r"    axis line style={-stealth}, tick label style={font=\scriptsize},",
        r"]",
    ]
    for s in STRATEGIES:
        style, label = _MARKS[s]
        coords = " ".join(c(xy[(d, s)]) for d in domains)
        lines += [f"\\addplot[only marks, {style}] coordinates {{{coords}}};",
                  f"\\addlegendentry{{{label}}}"]
    for d in domains:
        coords = " ".join(c(xy[(d, s)]) for s in STRATEGIES)
        lines.append(f"\\addplot[{_LINE_COLORS[d]}, dashed, thin, no marks, forget plot] coordinates {{{coords}}};")
    for d in domains:
        # Label next to the Baseline point: below it near the right edge, to its right otherwise.
        x, y = xy[(d, "basic")]
        label = LABELS[d] + (" (control)" if d == CONTROL else "")
        anchor, at = ("north", (x, y - 0.03)) if x > 0.7 else ("west", (x + 0.015, y))
        lines.append(f"\\node[anchor={anchor}, font=\\tiny, text={_TEXT_COLORS[d]}] "
                     f"at (axis cs:{at[0]:.2f},{at[1]:.2f}) {{{label}}};")
    lines += [
        r"\fill[green!10, opacity=0.3] (axis cs:0.6,0.7) rectangle (axis cs:1.0,1.0);",
        r"\fill[red!10, opacity=0.3] (axis cs:0,0) rectangle (axis cs:0.3,0.3);",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Mean similarity ($\mu_S$, over the runs that finished) against extraction rate (proportion "
        r"of the 24 models whose exploration finished), per domain and strategy, existential queries, first "
        r"execution. Dashed lines join the three strategies of a domain. The shaded regions mark the upper-right "
        r"and lower-left corners of the plane.}",
        r"\label{fig:tradeoff_scatter}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)


def table_domains(ref_dir: str = os.path.join(HERE, "reference_contexts"),
                  bases_path: str = os.path.join(DATA, "derived", "reference_bases.json")) -> str:
    """Objects, attributes, concepts and Duquenne-Guigues basis size of each reference context."""
    import json
    with open(bases_path, encoding="utf-8") as f:
        bases = json.load(f)
    refs = _reference_contexts(ref_dir)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textit{Domain} & \textit{Objects} & \textit{Attributes} & \textit{Concepts} & \textit{DG basis} \\",
        r"\midrule",
    ]
    for heading, domains in _domain_blocks():
        for d in domains:
            b = bases[d]
            label = LABELS[d] + (" (control)" if d == CONTROL else "")
            lines.append(f"{label} & {b['n_objects']} & {b['n_attributes']} & "
                         f"{len(refs[d].lattice)} & {b['n_implications']} \\\\")
        if heading == "Case studies":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Reference formal context of each domain: number of objects and attributes, number of "
        r"concepts of its lattice, and number of implications of its Duquenne--Guigues basis.}",
        r"\label{tab:domains}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--output-dir", default="output/tables")
    ap.add_argument("--rerun-dir", default=None, help="output directory of the repetition (has traces/)")
    args = ap.parse_args()
    ex = load_executions(args.data_dir, rerun_dir=args.rerun_dir)
    first = ex["first"]
    outputs = {
        "table_replication.tex": table_replication(ex),
        "table_query_asymmetry.tex": table_query_asymmetry(first),
        "table_cfc_gains.tex": table_cfc_gains(first),
        "table_dg_graphs.tex": table_dg_graphs(first),
        "fig_tradeoff_scatter.tex": fig_tradeoff_scatter(first),
        "table_domains.tex": table_domains(),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    for name, text in outputs.items():
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
