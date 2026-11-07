"""Retry-bound statistics of the second execution, from the JSON traces.

For every query that needed at least one retry (a rejected response), the
bound is read from the trace (``max_retries`` of the first rejection) and checked
against the rule of the paper: N_max = min(N_op, ceil(2^H*)), where H* is the
joint entropy of the first rejected response if its JSON is valid and the mean
entropy per token otherwise (Section 4.3). Output: the counts quoted in the paper
(Sections 7.6 and 9.9).

Usage:
    python -m experiments.analyze_retries --traces-dir experiments/data/traces
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

N_OP = 10
SKIPPED = []
DOMAINS = ["integers", "graphs", "rcc", "top"]


def _b(v) -> bool:
    return str(v) in ("True", "true", "1")


def queries(log):
    """Yield one record per query that needed a retry (``retries_used >= 1``).

    A query starts at an ``ask_llm`` with ``retry_attempt == 0`` or without a retry
    index (a request that failed, e.g. a time-out, before any response). The bound
    is the ``max_retries`` stored at the first retried validation of the query, and
    H* comes from the entropies stored in that validation (the response it judged).
    """
    qid = -1
    seen = {}
    for e in log:
        ev = e.get("event")
        if ev == "ask_llm":
            if "retry_attempt" not in e or str(e.get("retry_attempt")) == "0":
                qid += 1
        elif ev == "validate_logic" and qid >= 0 and qid not in seen:
            if int(e.get("retries_used") or 0) >= 1:
                valid = _b(e.get("is_valid_json"))
                H = None
                if "joint_entropy" in e and "mean_entropy" in e:
                    H = float(e["joint_entropy"]) if valid else float(e["mean_entropy"])
                seen[qid] = {"valid_json": valid, "max_retries": int(e["max_retries"]), "H_star": H}
    yield from seen.values()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces-dir", default="experiments/data/traces")
    args = ap.parse_args()

    per_domain = defaultdict(list)
    mismatches = 0
    for path in sorted(glob.glob(os.path.join(args.traces_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
        for q in queries(t["iteration_log"]):
            per_domain[t["context_name"]].append(q)
            if q["H_star"] is None:
                SKIPPED.append(1)
            else:
                calc = min(N_OP, math.ceil(2 ** q["H_star"]))
                if calc != q["max_retries"]:
                    mismatches += 1

    allq = [q for d in DOMAINS for q in per_domain[d]]
    print(f"queries needing a retry: {len(allq)}")
    print(f"N_max recomputed from H* differs from the trace in {mismatches} queries "
          f"({len(SKIPPED)} without entropy in the trace)")
    bounds = Counter(q["max_retries"] for q in allq)
    print(f"bound = 2: {bounds[2]} of {len(allq)};  reached the cap {N_OP}: {bounds[N_OP]}")
    print("bound by validity of the first rejected response:")
    for valid in (False, True):
        bs = [q["max_retries"] for q in allq if q["valid_json"] == valid]
        if bs:
            print(f"  valid JSON={valid}: n={len(bs)}, bound between {min(bs)} and {max(bs)}")
    print("per domain (queries needing a retry; bound <= 3 share; reached the cap):")
    for d in DOMAINS:
        qs = per_domain[d]
        if not qs:
            continue
        le3 = sum(q["max_retries"] <= 3 for q in qs)
        cap = sum(q["max_retries"] == N_OP for q in qs)
        print(f"  {d:9s} n={len(qs):4d}  <=3: {100 * le3 / len(qs):4.0f}%  cap: {cap}")


if __name__ == "__main__":
    main()
