"""Tests for the render_fca_figure pipeline."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.render_fca_figure import (
    hasse_render_svg,
    load_context,
    pseudo_intents,
    stem_to_tikz,
)


FIXTURES = Path(__file__).parent / "fixtures"
TOY_CXT = FIXTURES / "toy.cxt"
PROJECT_ROOT = FIXTURES.parent.parent.parent
REPO_ROOT = FIXTURES.parent.parent
PAPER_CONTEXTOS = PROJECT_ROOT / "paper" / "contextos"


# ---------------------------------------------------------------------------
# Parser / context
# ---------------------------------------------------------------------------

def test_load_context_toy():
    """Toy: 3 objects x 3 attrs; 3 concepts in a chain.

    Incidence: 1: a,b,c   2: a   3: a,c
    """
    ctx = load_context(TOY_CXT)
    assert set(ctx.objects) == {"1", "2", "3"}
    assert set(ctx.properties) == {"a", "b", "c"}
    assert len(list(ctx.lattice)) == 3


# ---------------------------------------------------------------------------
# SVG rendering (direct ``concepts`` graphviz output)
# ---------------------------------------------------------------------------

def test_hasse_render_svg_toy():
    """SVG is valid XML and contains the label text from self-loops."""
    ctx = load_context(TOY_CXT)
    svg = hasse_render_svg(ctx)
    text = svg.decode("utf-8")
    assert text.startswith("<?xml")
    # The toy context has objects labelled "1", "2", "3" and attributes
    # labelled "a", "b", "c" — these appear as headlabel/taillabel text.
    assert "1" in text
    assert "2" in text
    assert "3" in text
    assert "a" in text
    assert "b" in text
    assert "c" in text


def test_hasse_render_svg_livingbeings():
    """LivingBeings SVG contains expected concept labels."""
    cxt = PAPER_CONTEXTOS / "livingbeings.cxt"
    if not cxt.exists():
        pytest.skip("livingbeings.cxt not found")
    ctx = load_context(cxt)
    svg = hasse_render_svg(ctx)
    text = svg.decode("utf-8")
    assert text.startswith("<?xml")
    for label in ("Frog", "Cat", "Legs", "Leech Fish", "Aquatic",
                  "Mobility", "Corn", "Need Water"):
        assert label in text


def test_hasse_render_svg_canonical_integers():
    """Canonical integers SVG is generated (25 attributes + property labels)."""
    cxt = PAPER_CONTEXTOS / "canonical_integers.cxt"
    if not cxt.exists():
        pytest.skip("canonical_integers.cxt not found")
    ctx = load_context(cxt)
    svg = hasse_render_svg(ctx)
    text = svg.decode("utf-8")
    assert text.startswith("<?xml")
    for label in ("prime", "even", "odd", "quadratic", "compound"):
        assert label in text


# ---------------------------------------------------------------------------
# Stem basis
# ---------------------------------------------------------------------------

def test_pseudo_intents_toy():
    """Toy context: 2 pseudo-intents ({b}, {c}) → 2 implications."""
    ctx = load_context(TOY_CXT)
    pis = pseudo_intents(ctx)
    assert {frozenset(p) for p in pis} == {
        frozenset({"b"}), frozenset({"c"}),
    }


def test_pseudo_intents_canonical_integers():
    cxt = PAPER_CONTEXTOS / "canonical_integers.cxt"
    if not cxt.exists():
        pytest.skip("canonical_integers.cxt not found")
    ctx = load_context(cxt)
    pis = pseudo_intents(ctx)
    assert len(pis) == 4


def test_stem_to_tikz_format():
    ctx = load_context(TOY_CXT)
    tikz = stem_to_tikz(ctx)
    assert tikz.startswith("$\\begin{array}")
    assert "\\Rightarrow" in tikz
    assert tikz.count("\\\\") == 2  # two rows, each ending with \\


def test_stem_to_tikz_canonical_integers():
    cxt = PAPER_CONTEXTOS / "canonical_integers.cxt"
    if not cxt.exists():
        pytest.skip("canonical_integers.cxt not found")
    ctx = load_context(cxt)
    tikz = stem_to_tikz(ctx)
    assert tikz.count("\\\\") == 4  # 4 rows


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_renders_toy(tmp_path):
    """End-to-end: invoke the CLI on the toy context and check outputs."""
    if shutil.which("dot") is None:
        pytest.skip("Graphviz `dot` not available")
    result = subprocess.run(
        [
            "python3", "-m", "experiments.render_fca_figure",
            str(TOY_CXT),
            "--out", str(tmp_path),
        ],
        capture_output=True, text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "toy_lattice.svg").exists()
    assert (tmp_path / "toy_stem.tex").exists()
    svg_data = (tmp_path / "toy_lattice.svg").read_bytes()
    assert svg_data.startswith(b"<?xml")
    stem = (tmp_path / "toy_stem.tex").read_text()
    assert "\\begin{array}" in stem
    assert stem.count("\\\\") == 2
