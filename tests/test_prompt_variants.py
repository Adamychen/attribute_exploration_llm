"""Tests for llmfca.prompting.variants and its registration in llmfca.extraction."""
import ast
import json
import os

import pytest

from llmfca.extraction import PROMPT_STRATEGIES, PROMPTS, VARIANT_PROMPTS, VARIANT_STRATEGIES
from llmfca.prompting.variants import (
    CONDITIONS,
    applicable,
    attrs_as_list,
    basis_as_sentences,
    formal_context_as_table,
    join_prompt,
    make_variant,
    parse_basis,
    parse_formal_context,
    repair_escapes,
    split_prompt,
)

DOMAINS = ("integers", "graphs")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def original(ctx, strategy):
    return PROMPTS[(ctx, strategy)]


@pytest.mark.parametrize("ctx", DOMAINS)
@pytest.mark.parametrize("strategy", PROMPT_STRATEGIES)
def test_split_is_lossless(ctx, strategy):
    p = original(ctx, strategy)
    assert join_prompt(split_prompt(p)) == p
    parts = split_prompt(p)
    assert bool(parts["block"]) == (strategy != "basic")


def test_only_graphs_stem_basis_has_the_corrupted_escape():
    corrupted = [k for k, v in PROMPTS.items() if k[1] in PROMPT_STRATEGIES and "\x0c" in v]
    assert corrupted == [("graphs", "stem_basis")]
    fixed = make_variant(original("graphs", "stem_basis"), "C0")
    assert "\x0c" not in fixed and "1. \\forall x (acyclic(x)" in fixed
    assert repair_escapes(fixed) == fixed


@pytest.mark.parametrize("ctx", DOMAINS)
@pytest.mark.parametrize("strategy", PROMPT_STRATEGIES)
def test_c0_equals_original_except_the_repair(ctx, strategy):
    p = original(ctx, strategy)
    assert make_variant(p, "C0") == repair_escapes(p)
    if (ctx, strategy) != ("graphs", "stem_basis"):
        assert make_variant(p, "C0") == p


@pytest.mark.parametrize("ctx", DOMAINS)
@pytest.mark.parametrize("strategy", ("full_context", "stem_basis"))
def test_v1_only_moves_the_block(ctx, strategy):
    base = repair_escapes(original(ctx, strategy))
    v1 = make_variant(base, "V1_order")
    assert v1.index("Here you have") < v1.index("Analyze the logical formula")
    assert sorted(v1.split()) == sorted(base.split())          # same words, different order
    assert v1.rstrip().endswith("```")


@pytest.mark.parametrize("ctx", DOMAINS)
def test_v4_keeps_every_description(ctx):
    base = original(ctx, "basic")
    descriptions = ast.literal_eval(split_prompt(base)["attrs"].strip())
    v4 = make_variant(base, "V4_attrs")
    for name, desc in descriptions.items():
        assert f"- {name}: {desc}" in v4
    assert "{'" not in split_prompt(v4)["attrs"]


@pytest.mark.parametrize("ctx", DOMAINS)
@pytest.mark.parametrize("strategy", PROMPT_STRATEGIES)
def test_v3_keeps_the_json_schema_and_context(ctx, strategy):
    base = repair_escapes(original(ctx, strategy))
    v3 = make_variant(base, "V3_wording")
    marker = "The output should be formatted"
    # the schema, and the strategy block that follows it, are untouched
    assert v3[v3.index(marker):] == base[base.index(marker):]
    # the context name, description and attribute descriptions are still there
    a = split_prompt(base)
    assert a["attrs"].strip() in v3
    assert "Welcome, Assistant!" not in v3
    for token in ("COUNTEREXAMPLE", "WITNESS", "vacuously", "\\forall x (P(x) \\to Q(x))"):
        assert token in v3


@pytest.mark.parametrize("ctx", DOMAINS)
def test_v2_table_encodes_the_same_relation(ctx):
    block = split_prompt(original(ctx, "full_context"))["block"]
    objects, attributes, relation = parse_formal_context(block)
    table = formal_context_as_table(block)
    rows = [line.strip("|").split("|") for line in table.splitlines() if line.startswith("| ") and "object" not in line]
    assert len(rows) == len(objects)
    for cells, obj in zip(rows, objects):
        cells = [c.strip() for c in cells]
        assert cells[0] == obj
        assert [c == "1" for c in cells[1:]] == [relation[obj][a] for a in attributes]


def test_v2_table_matches_the_integer_relations_in_the_original_prompt():
    block = split_prompt(original("integers", "full_context"))["block"]
    _, _, rel = parse_formal_context(block)
    assert rel["3"] == {"prime": True, "odd": True, "even": False, "compound": False, "quadratic": False}
    assert rel["9"] == {"prime": False, "odd": True, "even": False, "compound": True, "quadratic": True}


@pytest.mark.parametrize("ctx,n_rules", [("integers", 4), ("graphs", 8)])
def test_v2_basis_keeps_every_rule(ctx, n_rules):
    block = split_prompt(repair_escapes(original(ctx, "stem_basis")))["block"]
    rules = parse_basis(block)
    assert len(rules) == n_rules
    text = basis_as_sentences(block)
    assert len([l for l in text.splitlines() if l[:1].isdigit()]) == n_rules
    for kind, prem, concl in rules:
        for term in prem + concl:
            assert term.split("(")[0] in text


def test_v2_basis_matches_the_reference_basis_for_integers():
    """The integers prompt states the four implications of the Duquenne-Guigues basis."""
    block = split_prompt(original("integers", "stem_basis"))["block"]
    rules = {(k, tuple(sorted(a.split("(")[0] for a in p)), tuple(sorted(a.split("(")[0] for a in c)))
             for k, p, c in parse_basis(block)}
    assert ("implication", ("even", "quadratic"), ("compound",)) in rules
    assert {r[0] for r in rules} == {"implication", "empty"}
    with open(os.path.join(ROOT, "experiments", "data", "derived", "reference_bases.json")) as f:
        ref = json.load(f)["integers"]["implications"]
    assert {tuple(sorted(r[1])) for r in rules} == {tuple(sorted(i["premise"])) for i in ref}


def test_conditions_that_need_a_block_are_rejected_for_basic():
    for cond in ("V1_order", "V2_format"):
        with pytest.raises(ValueError):
            make_variant(original("integers", "basic"), cond)
    with pytest.raises(ValueError):
        make_variant(original("integers", "basic"), "V9")


def test_registry_holds_variants_and_leaves_the_default_grid_alone():
    assert PROMPT_STRATEGIES == ("basic", "full_context", "stem_basis")
    assert all("+" in s for s in VARIANT_STRATEGIES)
    # 3 conditions for basic + 5 for each of the two strategies with a block, per domain
    assert len(VARIANT_PROMPTS) == len(DOMAINS) * (3 + 5 + 5)
    assert {c for (c, _) in VARIANT_PROMPTS} == set(DOMAINS)
    assert ("integers", "basic+V1_order") not in PROMPTS
    for (ctx, strat), prompt in VARIANT_PROMPTS.items():
        assert PROMPTS[(ctx, strat)] is prompt
    assert set(applicable("basic")) < set(CONDITIONS)
