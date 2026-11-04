"""
Prompt variants for the sensitivity analysis.

The prompts used in the main sweep (``llmfca.prompting.prompts``) are all built
from the same five blocks, in this order:

    header     "Welcome, Assistant!", context name and description
    attrs      description of the attributes (a Python dict printed as text)
    rules      evaluation rules for universal / existential questions
    fmt        "You must respond with a SINGLE valid JSON object" + JSON schema
    block      strategy block, only for ``full_context`` and ``stem_basis``

This module derives variants of an existing prompt by editing one block at a
time, so the effect of each change can be measured against a reference prompt.
The prompts of the main sweep are never modified.

Conditions (suffix of the prompt strategy name, ``<strategy>+<condition>``):

    C0          reference: the original prompt with only a corrupted escape
                sequence repaired (see ``repair_escapes``)
    V1_order    the strategy block moves before the rules and format blocks
    V2_format   the strategy block is rewritten: the formal context as a table,
                the basis as plain-English sentences (same content)
    V3_wording  header, rules and instructions are reworded (same content; the
                JSON schema text is untouched because the parser depends on it)
    V4_attrs    attribute descriptions as a bulleted list instead of a dict repr
"""
import ast
import re
from typing import Callable, Dict, List, Tuple

CONDITIONS = ("C0", "V1_order", "V2_format", "V3_wording", "V4_attrs")

_ATTRS_MARK = "Attributes description:\n"
_RULES_MARK = "\nAnalyze the logical formula's main quantifier"
_FMT_MARK = "You must respond with a SINGLE valid JSON object."
_BLOCK_MARK = "Here you have "


def repair_escapes(prompt: str) -> str:
    """Repair ``\\forall`` where a form feed (``\\x0c``) replaced ``\\f``.

    ``GRAPHS_PROMPT_STEM_BASIS`` embeds a rule written ``"\\forall"`` in a
    non-raw Python string, where ``\\f`` is the form-feed escape, so the model
    receives ``\\<FF>orall``. Only that prompt is affected.
    """
    return prompt.replace("\\\x0corall", "\\forall")


def split_prompt(prompt: str) -> Dict[str, str]:
    """Split a prompt of the main sweep into its five blocks (lossless)."""
    i_attrs = prompt.index(_ATTRS_MARK) + len(_ATTRS_MARK)
    i_rules = prompt.index(_RULES_MARK, i_attrs)
    i_fmt = prompt.index(_FMT_MARK, i_rules)
    i_block = prompt.find(_BLOCK_MARK, i_fmt)
    if i_block == -1:
        i_block = len(prompt)
    parts = {
        "header": prompt[:i_attrs],
        "attrs": prompt[i_attrs:i_rules],
        "rules": prompt[i_rules:i_fmt],
        "fmt": prompt[i_fmt:i_block],
        "block": prompt[i_block:],
    }
    assert "".join(parts[k] for k in ("header", "attrs", "rules", "fmt", "block")) == prompt
    return parts


def join_prompt(parts: Dict[str, str]) -> str:
    return "".join(parts[k] for k in ("header", "attrs", "rules", "fmt", "block"))


# --------------------------------------------------------------------------
# V4: attribute descriptions as a bulleted list
# --------------------------------------------------------------------------

def attrs_as_list(attrs_text: str) -> str:
    descriptions = ast.literal_eval(attrs_text.strip())
    if not isinstance(descriptions, dict):
        raise ValueError("attribute block is not a dict")
    return "\n".join(f"- {name}: {desc}" for name, desc in descriptions.items()) + "\n\n"


# --------------------------------------------------------------------------
# V3: reworded header, rules and instructions
# --------------------------------------------------------------------------

REWORDED_RULES = """
Look at the main quantifier of the logical formula and apply the matching procedure:

1. UNIVERSAL QUANTIFIER (∀x ...)
   - Shape: \\forall x (P(x) \\to Q(x))
   - What to do: look for a COUNTEREXAMPLE.
   - How to decide:
     - A counterexample is an object x for which P(x) holds and Q(x) does not. With one such object, the statement is **FALSE** and you must give that object.
     - If no object satisfies P(x), the statement holds vacuously and is **TRUE**; there is no counterexample to give.
     - If some objects satisfy P(x) and all of them also satisfy Q(x), the statement is **TRUE**.
     - If the antecedent is empty (\\forall x \\to ...), the question is whether every object satisfies the consequent.

2. EXISTENTIAL QUANTIFIER (∃x ...)
   - Shape: \\exists x (P(x) \\land Q(x)) \\land (\\lnot K(x) \\lor \\lnot J(x))
   - What to do: look for a WITNESS (a valid example).
   - How to decide:
     - A witness is an object x for which both P(x) and Q(x) hold while K(x) or J(x) (at least one of them) does not. With one such object, the statement is **TRUE** and you must give that object.
     - If, after checking every object, none meets the condition, the statement is **FALSE**.

"""

_REWORDED_FMT_INTRO = (
    "Answer with exactly one valid JSON object.\n"
    "Do not return the JSON Schema itself.\n"
    "Do not write any text outside the JSON.\n"
)
_ORIG_FMT_INTRO_RE = re.compile(
    r"You must respond with a SINGLE valid JSON object\.\n"
    r"Do NOT return the JSON Schema definition\.\n"
    r"Do NOT output explanations outside the JSON\.\n"
)
_HEADER_RE = re.compile(
    r"\nWelcome, Assistant!\n\nContext name: (?P<name>.*)\n\n"
    r"Context description: (?P<desc>.*)\n\nAttributes description:\n", re.S)


def reword_header(header: str) -> str:
    m = _HEADER_RE.fullmatch(header)
    if not m:
        raise ValueError("unexpected header layout")
    return (f"\nYou are an assistant that answers questions about one domain.\n\n"
            f"Domain: {m['name']}\n\nDescription: {m['desc']}\n\nProperties of the domain:\n")


def reword_fmt(fmt: str) -> str:
    new, n = _ORIG_FMT_INTRO_RE.subn(_REWORDED_FMT_INTRO, fmt, count=1)
    if n != 1:
        raise ValueError("unexpected format-instruction layout")
    return new


# --------------------------------------------------------------------------
# V2: strategy block rewritten (same content, different presentation)
# --------------------------------------------------------------------------

_LITERAL = r"(?:\\lnot\s*)?[^\s,()]+\([^)]*\)"
_REL_RE = re.compile(rf"(?P<obj>[^\s:,{{}}]+):\s*(?P<lits>{_LITERAL}(?:,\s*{_LITERAL})*)\s*\.")
_LIT_RE = re.compile(r"(?P<neg>\\lnot\s*)?(?P<attr>[^\s,()]+)\((?P<arg>[^)]*)\)")


def parse_formal_context(block: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, bool]]]:
    """Parse the ``full_context`` block of an arity-1 domain."""
    objs = re.search(r"Objects:\s*\{(.*?)\}", block)
    attrs = re.search(r"Attributes:\s*\{(.*?)\}", block)
    rest = block.split("Object-Attribute Relations:", 1)
    if not (objs and attrs and len(rest) == 2):
        raise ValueError("formal context block not recognised")
    objects = [o.strip() for o in objs.group(1).split(",")]
    attributes = [a.strip() for a in attrs.group(1).split(",")]
    table: Dict[str, Dict[str, bool]] = {}
    for m in _REL_RE.finditer(rest[1]):
        row = {}
        for lit in _LIT_RE.finditer(m["lits"]):
            row[lit["attr"]] = lit["neg"] is None
        table[m["obj"]] = row
    if set(table) != set(objects):
        raise ValueError(f"objects in the relations {sorted(table)} differ from {sorted(objects)}")
    for o, row in table.items():
        if set(row) != set(attributes):
            raise ValueError(f"object {o}: attributes {sorted(row)} differ from {sorted(attributes)}")
    return objects, attributes, table


def formal_context_as_table(block: str) -> str:
    objects, attributes, table = parse_formal_context(block)
    lines = [
        "Here you have the complete formal context as a table (1 means the object has the "
        "attribute, 0 means it does not); use this information to answer my questions:",
        "",
        "| object | " + " | ".join(attributes) + " |",
        "|" + "---|" * (len(attributes) + 1),
    ]
    for o in objects:
        lines.append(f"| {o} | " + " | ".join("1" if table[o][a] else "0" for a in attributes) + " |")
    return "\n".join(lines) + "\n"


_ITEM_SPLIT_RE = re.compile(r"(?:^|(?<=\.)|\s)\d+\.\s+(?=\\)")
_TERM_RE = re.compile(r"[^\s()\\]+\([^)]*\)")


def parse_basis(block: str) -> List[Tuple[str, List[str], List[str]]]:
    """Parse the ``stem_basis`` block into (kind, premise, conclusion) triples.

    ``kind`` is ``"implication"`` (premise -> conclusion) or ``"empty"``
    (no object satisfies all the listed attributes; conclusion is empty).
    """
    body = block.split(":\n", 1)[1] if ":\n" in block else block
    items = [s.strip() for s in _ITEM_SPLIT_RE.split(body) if s.strip()]
    rules = []
    for item in items:
        item = item.rstrip(".").strip()
        if item.startswith("\\nexists") or item.startswith("\\lnot \\exists") or item.startswith("\\lnot\\exists"):
            rules.append(("empty", _TERM_RE.findall(item), []))
        elif item.startswith("\\forall") and "\\to" in item:
            lhs, rhs = item.split("\\to", 1)
            rules.append(("implication", _TERM_RE.findall(lhs), _TERM_RE.findall(rhs)))
        else:
            raise ValueError(f"rule not recognised: {item!r}")
    return rules


def _names(terms: List[str]) -> List[str]:
    return [t.split("(", 1)[0] for t in terms]


def basis_as_sentences(block: str) -> str:
    rules = parse_basis(block)
    lines = ["Here you have the set of rules (the canonical basis) written in plain English; "
             "use these rules as expert knowledge about the context:"]
    for k, (kind, prem, concl) in enumerate(rules, 1):
        if kind == "empty":
            lines.append(f"{k}. No object satisfies all of: {', '.join(_names(prem))}.")
        else:
            lines.append(f"{k}. If an object satisfies {' and '.join(_names(prem))}, "
                         f"then it also satisfies {' and '.join(_names(concl))}.")
    return "\n".join(lines) + "\n"


def reformat_block(block: str) -> str:
    if not block:
        return block
    if "complete formal context" in block:
        return formal_context_as_table(block)
    if "canonical basis" in block:
        return basis_as_sentences(block)
    raise ValueError("strategy block not recognised")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def make_variant(prompt: str, condition: str) -> str:
    """Return ``prompt`` under ``condition`` (one of ``CONDITIONS``)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    parts = split_prompt(repair_escapes(prompt))
    if condition == "C0":
        pass
    elif condition == "V1_order":
        if not parts["block"]:
            raise ValueError("V1_order needs a strategy block")
        parts = {**parts,
                 "rules": "\n" + parts["block"].strip("\n") + "\n" + parts["rules"],
                 "block": ""}
    elif condition == "V2_format":
        if not parts["block"]:
            raise ValueError("V2_format needs a strategy block")
        parts["block"] = reformat_block(parts["block"])
    elif condition == "V3_wording":
        parts["header"] = reword_header(parts["header"])
        parts["rules"] = REWORDED_RULES
        parts["fmt"] = reword_fmt(parts["fmt"])
    elif condition == "V4_attrs":
        parts["attrs"] = attrs_as_list(parts["attrs"])
    return join_prompt(parts)


def applicable(strategy: str) -> Tuple[str, ...]:
    """Conditions that make sense for a strategy (V1/V2 need a strategy block)."""
    if strategy == "basic":
        return ("C0", "V3_wording", "V4_attrs")
    return CONDITIONS


def variant_name(strategy: str, condition: str) -> str:
    return f"{strategy}+{condition}"


def build_variant_prompts(prompts: Dict[Tuple[str, str], str],
                          contexts: Tuple[str, ...] = ("integers", "graphs"),
                          ) -> Dict[Tuple[str, str], str]:
    """Variant prompts keyed ``(context, "<strategy>+<condition>")``."""
    out: Dict[Tuple[str, str], str] = {}
    for (ctx, strategy), prompt in prompts.items():
        if ctx not in contexts:
            continue
        for cond in applicable(strategy):
            out[(ctx, variant_name(strategy, cond))] = make_variant(prompt, cond)
    return out
