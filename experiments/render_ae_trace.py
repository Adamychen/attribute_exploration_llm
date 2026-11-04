"""Generate the worked Attribute Exploration trace on integer arithmetic.

The trace is produced by the candidate generator used in the experiments
(``AttributeExplorerGraph.next_implication``) with a deterministic oracle that
knows the definitions of the five integer attributes. When a candidate
implication is not valid, the oracle returns the first refuting number from
``COUNTEREXAMPLE_ORDER`` (the counterexamples of the worked example in the
paper) and, if none refutes it, the smallest positive integer that does.

For each candidate implication P -> Q the table reports its support in the
final context K_E, |(P u Q)'| / |G u E|.

Usage:
    python -m experiments.render_ae_trace --output tables/table_ae_trace.tex
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from concepts import Context

from llmfca.attribute_explorer import AttributeExplorerGraph, Implication

ATTRIBUTES = ["even", "odd", "prime", "compound", "quadratic"]
COUNTEREXAMPLE_ORDER = [25, 2, 6, 3, 1, 4, 15]
SEARCH_LIMIT = 10_000


def integer_attributes(n: int) -> dict[str, bool]:
    """Attributes of a positive integer (1 is neither prime nor compound)."""
    is_prime = n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1))
    return {
        "even": n % 2 == 0,
        "odd": n % 2 == 1,
        "prime": is_prime,
        "compound": n > 1 and not is_prime,
        "quadratic": math.isqrt(n) ** 2 == n,
    }


def refutes(n: int, premise: set[str], conclusion: set[str]) -> bool:
    a = integer_attributes(n)
    return all(a[m] for m in premise) and not all(a[m] for m in conclusion)


def find_counterexample(premise: set[str], conclusion: set[str]) -> Optional[int]:
    for n in COUNTEREXAMPLE_ORDER:
        if refutes(n, premise, conclusion):
            return n
    for n in range(1, SEARCH_LIMIT + 1):
        if refutes(n, premise, conclusion):
            return n
    return None


@dataclass
class Step:
    premise: list[str]
    conclusion: list[str]
    accepted: bool
    counterexample: Optional[int]
    n_objects: int
    n_basis: int
    support: int = 0


def _ordered(attrs) -> list[str]:
    return sorted(set(attrs), key=ATTRIBUTES.index)


def run_trace() -> tuple[list[Step], list[int]]:
    explorer = AttributeExplorerGraph(None, None, "", ATTRIBUTES, None)
    ctx: Optional[Context] = None
    basis: list[Implication] = []
    objects: list[int] = []
    steps: list[Step] = []
    while True:
        imp = Implication([], ATTRIBUTES) if ctx is None else explorer.next_implication(ctx, basis)
        if imp is None:
            break
        premise = set(imp.antecedent)
        conclusion = set(imp.consequent) - premise
        n = find_counterexample(premise, conclusion)
        if n is None:
            basis.append(imp)
        else:
            objects.append(n)
            ctx = Context([str(o) for o in objects], ATTRIBUTES,
                          [[integer_attributes(o)[m] for m in ATTRIBUTES] for o in objects])
        steps.append(Step(_ordered(premise), _ordered(conclusion), n is None, n,
                          len(objects), len(basis)))
    for s in steps:
        s.support = sum(1 for o in objects
                        if all(integer_attributes(o)[m] for m in s.premise + s.conclusion))
    return steps, objects


def _set(attrs: list[str]) -> str:
    return r"\emptyset" if not attrs else r"\{" + ", ".join(attrs) + r"\}"


def to_latex(steps: list[Step], objects: list[int]) -> str:
    n = len(objects)
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cclccll}",
        r"\hline",
        r"\textit{Step} & \textit{Premise} $P$ & \textit{Candidate Implication} $P \to P'' \setminus P$"
        r" & $\text{supp}$ & \textit{Decision} & \textit{Action / Update} & \textit{State} \\",
        r"\hline",
    ]
    for i, s in enumerate(steps, 1):
        decision = r"\textsc{yes}" if s.accepted else r"\textsc{no}"
        action = (r"Accept into basis $\mathcal{L}$" if s.accepted
                  else f"Counterexample ${s.counterexample}$")
        lines.append(
            f"{i} & ${_set(s.premise)}$ & ${_set(s.premise)} \\to {_set(s.conclusion)}$ & "
            f"${s.support}/{n}$ & {decision} & {action} & "
            f"$|E|={s.n_objects}, |\\mathcal{{L}}|={s.n_basis}$ \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\caption{Step-by-step trace of Attribute Exploration on the Integer Arithmetic domain. "
        r"At each step $t$, candidate premise $P$ satisfies $P = \mathcal{L}(P)$ and $P \subsetneq P''$. "
        r"The oracle either confirms the candidate rule (which enters the Duquenne--Guigues basis $\mathcal{L}$) "
        r"or refutes it by providing an object violating the rule (which enters the counterexample set $E$). "
        r"Candidates already closed ($P = P''$) or entailed by $\mathcal{L}$ ($P \neq \mathcal{L}(P)$) "
        r"are pruned without querying the oracle. "
        f"The column $\\text{{supp}}$ gives the support of each candidate implication in the final context "
        f"$\\mathbb{{K}}_E$ ({n} objects).}}",
        r"\label{tab:ae_trace_example}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="output/tables/table_ae_trace.tex")
    args = parser.parse_args()
    steps, objects = run_trace()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_latex(steps, objects))
    print(f"Wrote {out} ({len(steps)} steps, final context {objects})")


if __name__ == "__main__":
    main()
