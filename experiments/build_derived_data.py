"""
Build the derived datasets that accompany the raw sweep output.

Inputs (raw, never modified):
  * ``results.csv``            one row per run (model, context, prompt, query mode)
  * ``traces/*.json``          one trace per run (iteration log, final context)
  * ``reference_contexts/*.json``  reference formal contexts of the four domains

Where a quantity exists in both raw sources, this is the one that
feeds each derived column. ``S``, ``S_H``, ``n_concepts_llm`` and the lattices
come from ``results.csv`` (the source of the paper's tables); everything
describing the exploration itself (conjectures, retries, entropy, wall-clock)
comes from the traces. The terminal state is reported from both sources.

Outputs (written under ``--output-dir``; see ``DATA.md`` for the column guide):
  * ``run_summary.csv``        one row per run: outcome, effort, S and S_H
  * ``conjectures.csv``        one row per implication posed by Attribute
                               Exploration, with its support in the reference context
  * ``reference_bases.json``   Duquenne-Guigues basis of every reference context,
                               with the support of each implication
  * ``lattices/<run_id>.json`` concept lattice of every run that produced a final context

Everything is computed deterministically from the raw files (no LLM calls),
so the outputs can be regenerated and compared bit by bit.

Support convention: ``supp(A -> B) = |A'| / |G|``, the fraction of objects of
the context that have every attribute of the premise ``A``. An implication
*holds* in a context when every object that has all of ``A`` also has all of ``B``.

Usage:
    python -m experiments.build_derived_data \
        --results experiments/data/results.csv \
        --traces-dir experiments/data/traces \
        --output-dir experiments/data/derived
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(current_dir)
sys.path.insert(0, code_dir)

# Same renaming as experiments.generate_paper_assets, so that model names match the paper.
MODEL_RENAMES = {
    "qwen3-coder:latest": "qwen3-coder:30b",
    "phi4-mini:latest": "phi4-mini:3.8b",
}

RUN_KEY = ["model", "context_name", "prompt_mode", "question_mode"]


# --------------------------------------------------------------------------
# Reference contexts and Duquenne-Guigues basis
# --------------------------------------------------------------------------

class RefContext:
    """A finite formal context stored as one attribute bitmask per object."""

    def __init__(self, objects: Sequence[str], attributes: Sequence[str],
                 object_intents: Sequence[int]):
        self.objects = list(objects)
        self.attributes = list(attributes)
        self.object_intents = list(object_intents)
        self.index = {a: i for i, a in enumerate(self.attributes)}

    @classmethod
    def from_json(cls, path: str) -> "RefContext":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        masks = [sum(1 << j for j in row) for row in d["context"]]
        return cls(d["objects"], d["properties"], masks)

    @classmethod
    def from_matrix(cls, objects, attributes, matrix) -> "RefContext":
        masks = [sum(1 << j for j, v in enumerate(row) if v) for row in matrix]
        return cls(objects, attributes, masks)

    @property
    def full_mask(self) -> int:
        return (1 << len(self.attributes)) - 1

    def mask(self, names: Sequence[str]) -> int:
        return sum(1 << self.index[a] for a in names)

    def names(self, mask: int) -> List[str]:
        return [a for i, a in enumerate(self.attributes) if mask >> i & 1]

    def extent_count(self, mask: int) -> int:
        """|A'|: number of objects having every attribute in ``mask``."""
        return sum(1 for g in self.object_intents if g & mask == mask)

    def closure(self, mask: int) -> int:
        """A'': attributes shared by all objects having every attribute in ``mask``."""
        result = self.full_mask
        for g in self.object_intents:
            if g & mask == mask:
                result &= g
        return result

    def holds(self, premise: int, conclusion: int) -> bool:
        return all(g & conclusion == conclusion
                   for g in self.object_intents if g & premise == premise)

    def n_counterexamples(self, premise: int, conclusion: int) -> int:
        return sum(1 for g in self.object_intents
                   if g & premise == premise and g & conclusion != conclusion)


def duquenne_guigues_basis(ctx: RefContext) -> List[Tuple[int, int]]:
    """Duquenne-Guigues basis ``{P -> P'' : P pseudo-intent}`` by NextClosure.

    Implements Ganter's algorithm with the pseudo-closure operator
    ``L(X) = X + union of P'' for the pseudo-intents P strictly inside X``.
    Returns the pairs ``(P, P'')`` as bitmasks, in lectic order.
    """
    n = len(ctx.attributes)
    basis: List[Tuple[int, int]] = []

    def pseudo_closure(x: int) -> int:
        changed = True
        while changed:
            changed = False
            for p, c in basis:
                if p & x == p and p != x and c & ~x:
                    x |= c
                    changed = True
        return x

    a = pseudo_closure(0)
    while True:
        closed = ctx.closure(a)
        if closed != a:
            basis.append((a, closed))
        if a == ctx.full_mask:
            break
        for i in range(n - 1, -1, -1):
            if a >> i & 1:
                continue
            low = (1 << i) - 1
            candidate = pseudo_closure((a & low) | (1 << i))
            if candidate & low == a & low:
                a = candidate
                break
        else:
            break
    return basis


def reference_bases_payload(ref_contexts: Dict[str, RefContext]) -> dict:
    payload = {}
    for name, ctx in sorted(ref_contexts.items()):
        n_obj = len(ctx.objects)
        impls = []
        for premise, conclusion in duquenne_guigues_basis(ctx):
            count = ctx.extent_count(premise)
            impls.append({
                "premise": ctx.names(premise),
                "conclusion": ctx.names(conclusion & ~premise),
                "support_count": count,
                "support": count / n_obj if n_obj else None,
            })
        payload[name] = {
            "n_objects": n_obj,
            "n_attributes": len(ctx.attributes),
            "attributes": ctx.attributes,
            "n_implications": len(impls),
            "implications": impls,
        }
    return payload


# --------------------------------------------------------------------------
# Traces -> conjectures
# --------------------------------------------------------------------------

_IMPL_RE = re.compile(r"^(\[.*\]) => (\[.*\])$", re.DOTALL)


def parse_implication(text: str) -> Tuple[List[str], List[str]]:
    """Parse the ``str(Implication)`` form ``"[a, b] => [c]"`` stored in the traces."""
    m = _IMPL_RE.match(text.strip())
    if not m:
        raise ValueError(f"Cannot parse implication: {text!r}")
    return ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))


def run_id(model: str, context: str, prompt: str, qmode: str) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9.\-]", "_", p) for p in (model, context, prompt, qmode))


def extract_conjectures(log: List[dict]) -> List[dict]:
    """One record per implication posed by the explorer.

    ``outcome`` is one of:
      * ``accepted``    the oracle agreed with the implication (it enters the basis)
      * ``refuted``     the oracle supplied a counterexample object
      * ``unresolved``  no valid answer was obtained (retry bound reached, or the
                        run was cut by the wall-clock limit)
    """
    by_step: Dict[int, List[dict]] = {}
    for e in log:
        by_step.setdefault(e["step"], []).append(e)
    records = []
    for step in sorted(by_step):
        events = by_step[step]
        posed = next((e for e in events
                      if e["event"] == "find_next_implication" and e.get("implication")), None)
        if posed is None:
            continue  # terminal step: the exploration reported nothing left to ask
        premise, conclusion = parse_implication(posed["implication"])
        update = next((e for e in events if e["event"] == "update_state"), None)
        validations = [e for e in events if e["event"] == "validate_logic"]
        asks = [e for e in events if e["event"] == "ask_llm"]
        if update is not None and update.get("decision") == "accept":
            outcome = "accepted"
        elif update is not None and update.get("decision") == "add_counterexample":
            outcome = "refuted"
        else:
            outcome = "unresolved"
        retries = max((v.get("retries_used", 0) for v in validations), default=0)
        exhausted = any(v.get("retries_used", 0) > v.get("max_retries", 0)
                        for v in validations if "retries_used" in v)
        last_ask = asks[-1] if asks else {}
        records.append({
            "step": step,
            "stage": posed.get("stage"),
            "premise": premise,
            "conclusion": conclusion,
            "outcome": outcome,
            "n_llm_calls": len(asks),
            "n_retries": retries,
            "retry_exhausted": bool(exhausted),
            "counterexample_object": update.get("object") if outcome == "refuted" else None,
            "joint_entropy": last_ask.get("joint_entropy"),
            "mean_entropy": last_ask.get("mean_entropy"),
        })
    return records


def conjecture_rows(trace: dict, key: dict, ref: Optional[RefContext]) -> List[dict]:
    final_ctx = None
    fc = trace.get("final_context")
    if fc:
        final_ctx = RefContext.from_matrix(fc["objects"], fc["properties"], fc["context"])
    rows = []
    for rec in extract_conjectures(trace.get("iteration_log", [])):
        full_conclusion = sorted(set(rec["premise"]) | set(rec["conclusion"]))
        row = {
            **key,
            "run_id": run_id(*(key[k] for k in RUN_KEY)),
            "step": rec["step"],
            "stage": rec["stage"],
            "premise": json.dumps(rec["premise"], ensure_ascii=False),
            "conclusion": json.dumps(rec["conclusion"], ensure_ascii=False),
            "premise_size": len(rec["premise"]),
            "outcome": rec["outcome"],
            "n_llm_calls": rec["n_llm_calls"],
            "n_retries": rec["n_retries"],
            "retry_exhausted": rec["retry_exhausted"],
            "counterexample_object": rec["counterexample_object"],
            "joint_entropy": rec["joint_entropy"],
            "mean_entropy": rec["mean_entropy"],
        }
        for label, ctx in (("ref", ref), ("final", final_ctx)):
            if ctx is None or not all(a in ctx.index for a in full_conclusion):
                row.update({f"support_count_{label}": None, f"support_{label}": None,
                            f"holds_{label}": None, f"n_counterexamples_{label}": None})
                continue
            p, c = ctx.mask(rec["premise"]), ctx.mask(full_conclusion)
            count = ctx.extent_count(p)
            row.update({
                f"support_count_{label}": count,
                f"support_{label}": count / len(ctx.objects),
                f"holds_{label}": ctx.holds(p, c),
                f"n_counterexamples_{label}": ctx.n_counterexamples(p, c),
            })
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Lattices and per-run summary
# --------------------------------------------------------------------------

def lattice_payload(objects, attributes, matrix) -> dict:
    from concepts import Context
    ctx = Context(objects, attributes, matrix)
    concepts = list(ctx.lattice)
    ids = {id(c): i for i, c in enumerate(concepts)}
    return {
        "n_concepts": len(concepts),
        "concepts": [{"id": i, "extent": list(c.extent), "intent": list(c.intent)}
                     for i, c in enumerate(concepts)],
        # (lower, upper) pairs of the covering relation: lower < upper in the concept order
        "covers": [[i, ids[id(u)]] for i, c in enumerate(concepts)
                   for u in c.upper_neighbors],
    }


def similarity_scores(row: pd.Series, ref_json_ctx) -> Tuple[Optional[float], Optional[float]]:
    from concepts import Context
    from llmfca.analysis.metrics import compute_similarities
    if row.get("early_stop", False) or row.get("timeout", False) or ref_json_ctx is None:
        return None, None
    llm = Context(json.loads(row["ctx_objects"]), json.loads(row["ctx_attributes"]),
                  json.loads(row["ctx_relations"]))
    return compute_similarities(llm, ref_json_ctx)


def outcome_of(record) -> str:
    """Terminal state of a run from its ``finished`` / ``timeout`` / ``early_stop`` flags."""
    if record.get("timeout"):
        return "timeout"
    if record.get("finished"):
        return "finished"
    if record.get("early_stop"):
        return "early_stop"
    return "unknown"


def main() -> None:
    from concepts import Context

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=os.path.join(current_dir, "data", "results.csv"))
    ap.add_argument("--traces-dir", default=os.path.join(current_dir, "data", "traces"))
    ap.add_argument("--ref-dir", default=os.path.join(current_dir, "reference_contexts"))
    ap.add_argument("--output-dir", default=os.path.join(current_dir, "data", "derived"))
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(out, "lattices"), exist_ok=True)

    print("Loading reference contexts...")
    refs, ref_concepts = {}, {}
    for path in sorted(glob.glob(os.path.join(args.ref_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        refs[name] = RefContext.from_json(path)
        ref_concepts[name] = Context.fromjson(path)

    print("Computing reference Duquenne-Guigues bases...")
    bases = reference_bases_payload(refs)
    with open(os.path.join(out, "reference_bases.json"), "w", encoding="utf-8") as f:
        json.dump(bases, f, indent=1, ensure_ascii=False)
    for name, b in bases.items():
        print(f"  {name}: {b['n_implications']} implications over "
              f"{b['n_objects']} objects, {b['n_attributes']} attributes")

    print("Loading raw results and traces...")
    results = pd.read_csv(args.results)
    results["model"] = results["model"].replace(MODEL_RENAMES)
    results = results.set_index(RUN_KEY, drop=False)
    n_ref_concepts = {n: len(c.lattice) for n, c in ref_concepts.items()}

    summary_rows, conj_rows = [], []
    trace_paths = sorted(glob.glob(os.path.join(args.traces_dir, "*.json")))
    for path in trace_paths:
        with open(path, encoding="utf-8") as f:
            trace = json.load(f)
        trace["model"] = MODEL_RENAMES.get(trace["model"], trace["model"])
        key = {k: trace[k] for k in RUN_KEY}
        rid = run_id(*(key[k] for k in RUN_KEY))
        row = results.loc[tuple(key[k] for k in RUN_KEY)]
        ref_ctx = refs.get(key["context_name"])

        conj = conjecture_rows(trace, key, ref_ctx)
        conj_rows.extend(conj)

        S, S_H = similarity_scores(row, ref_concepts.get(key["context_name"]))
        n_concepts = None
        if not (row["early_stop"] or row["timeout"]):
            lat = lattice_payload(json.loads(row["ctx_objects"]),
                                  json.loads(row["ctx_attributes"]),
                                  json.loads(row["ctx_relations"]))
            n_concepts = lat["n_concepts"]
            with open(os.path.join(out, "lattices", f"{rid}.json"), "w", encoding="utf-8") as f:
                json.dump({**key, "run_id": rid, **lat}, f, ensure_ascii=False)

        outcomes = [c["outcome"] for c in conj]
        summary_rows.append({
            **key,
            "run_id": rid,
            # results.csv is the first execution of the sweep and the traces the second;
            # the terminal state is kept for both because they do not always agree (see DATA.md).
            "outcome_results_csv": outcome_of(row),
            "outcome_trace": outcome_of(trace),
            "sources_agree": outcome_of(row) == outcome_of(trace),
            "n_iterations": trace["n_iterations"],
            "n_conjectures": len(conj),
            "n_accepted": outcomes.count("accepted"),
            "n_refuted": outcomes.count("refuted"),
            "n_unresolved": outcomes.count("unresolved"),
            "stem_basis_size": len(trace.get("stem_basis", [])),
            # If the oracle accepts the very first implication ([] -> all attributes),
            # the explorer stops at once with a placeholder one-object context and an
            # empty basis (llmfca.attribute_explorer.update_state).
            "accepted_at_initialization": any(
                c["stage"] == "initialization" and c["outcome"] == "accepted" for c in conj),
            "n_retries_total": trace["n_retries_total"],
            "n_retry_exhausted": trace["n_retry_exhausted"],
            "wallclock_seconds": trace["wallclock_seconds"],
            "joint_entropy": trace["joint_entropy"],
            "mean_entropy": trace["mean_entropy"],
            "n_ctx_objects": len(json.loads(row["ctx_objects"])) if n_concepts is not None else None,
            "n_ctx_attributes": len(json.loads(row["ctx_attributes"])) if n_concepts is not None else None,
            "n_concepts_llm": n_concepts,
            "n_concepts_ref": n_ref_concepts.get(key["context_name"]),
            "S": S,
            "S_H": S_H,
        })

    summary = pd.DataFrame(summary_rows).sort_values(RUN_KEY).reset_index(drop=True)
    conjectures = pd.DataFrame(conj_rows).sort_values(RUN_KEY + ["step"]).reset_index(drop=True)
    summary.to_csv(os.path.join(out, "run_summary.csv"), index=False)
    conjectures.to_csv(os.path.join(out, "conjectures.csv"), index=False)
    print(f"Wrote {len(summary)} runs, {len(conjectures)} conjectures, "
          f"{len(os.listdir(os.path.join(out, 'lattices')))} lattices to {out}")
    missing = set(map(tuple, results[RUN_KEY].values)) - set(map(tuple, summary[RUN_KEY].values))
    if missing:
        print(f"WARNING: {len(missing)} runs in results.csv have no trace")


if __name__ == "__main__":
    main()
