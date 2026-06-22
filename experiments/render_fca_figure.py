"""Render FCA figures (concept lattice + Stem basis) from a .cxt file.

Uses the ``concepts`` library's built-in graphviz integration to render
the Hasse diagram directly as SVG. The ``concepts`` library positions
extent and intent labels around uniform-sized nodes via invisible
self-loops, so all nodes have the same diameter regardless of label
length.

Usage:
    python -m experiments.render_fca_figure <cxt_path> --out <output_dir>

Outputs:
    <name>_lattice.svg  — Concept lattice as an SVG vector graphic
    <name>_lattice.png  — Same, as high-DPI PNG (≥300 DPI, for LaTeX)
    <name>_stem.tex     — Stem basis (Duquenne-Guigues) as a TikZ matrix
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import click
from concepts import Context


# ---------------------------------------------------------------------------
# .cxt parsing
# ---------------------------------------------------------------------------

def load_context(cxt_path: Path) -> Context:
    """Load a formal context from a .cxt file (Burmeister format)."""
    return Context.fromfile(str(cxt_path))


# ---------------------------------------------------------------------------
# Hasse diagram rendering — direct ``concepts`` graphviz output
# ---------------------------------------------------------------------------

def _build_graph(ctx: Context, hide_objects: bool = False):
    """Build a tuned ``graphviz.Digraph`` from the concept lattice.

    All nodes have uniform size; labels are positioned around nodes
    (extent above, intent below) via the invisible self-loop trick
    built into ``concepts``.

    Args:
        hide_objects: When True, strip object labels (extent) from the diagram.
    """
    g = ctx.lattice.graphviz()
    g.graph_attr["splines"] = "line"
    g.graph_attr["fontname"] = "Sans"
    g.graph_attr["fontsize"] = "12"
    g.node_attr["fillcolor"] = "white"
    g.node_attr["fontname"] = "Sans"
    g.node_attr["fontsize"] = "12"
    g.edge_attr["labelfontname"] = "Sans"
    g.edge_attr["labelfontsize"] = "12"
    g.edge_attr["labeldistance"] = "1.5"
    g.edge_attr["minlen"] = "2"
    g.edge_attr["dir"] = "none"

    if hide_objects:
        new_body = []
        for line in g.body:
            if "color=transparent" in line and "headlabel" in line:
                line = re.sub(r' headlabel=(?:"[^"]*"|\S+)', "", line)
                line = re.sub(r" labelangle=270", "", line)
                line = re.sub(r"  +", " ", line)
            new_body.append(line)
        g.body = new_body

    return g


def hasse_render_svg(ctx: Context, hide_objects: bool = False) -> bytes:
    """Render the Hasse diagram as SVG bytes."""
    return _build_graph(ctx, hide_objects=hide_objects).pipe(format="svg")


def hasse_render_png(ctx: Context, dpi: int = 300, hide_objects: bool = False) -> bytes:
    """Render the Hasse diagram as a high-DPI PNG.

    Args:
        dpi: Target resolution in dots per inch (default 300).
    """
    g = _build_graph(ctx, hide_objects=hide_objects)
    g.graph_attr["dpi"] = str(dpi)
    return g.pipe(format="png")


# ---------------------------------------------------------------------------
# Stem basis (Duquenne-Guigues)
# ---------------------------------------------------------------------------

def _closure(ctx: Context, P: frozenset[str]) -> frozenset[str]:
    """P'' in FCA notation: intent of the extent of P."""
    return frozenset(ctx.intension(ctx.extension(P)))


def pseudo_intents(ctx: Context) -> list[frozenset[str]]:
    """Enumerate pseudo-intents in lexicographic (canonical) order."""
    M_list = sorted(ctx.properties)
    accepted: list[frozenset[str]] = []
    for size in range(1, len(M_list)):
        for combo in combinations(M_list, size):
            P = frozenset(combo)
            if _closure(ctx, P) == P:
                continue
            ok = True
            for Q in accepted:
                if Q.issubset(P) and Q != P:
                    if not _closure(ctx, Q).intersection(P).issubset(Q):
                        ok = False
                        break
            if ok:
                accepted.append(P)
    return accepted


def _format_set_inline(s: frozenset[str]) -> str:
    if not s:
        return "\\emptyset"
    return ",".join(sorted(s))


def stem_to_tikz(ctx: Context) -> str:
    """Emit a LaTeX array listing every Stem-basis implication.

    Format::

        $\\begin{array}{r@{\\;\\Rightarrow\\;}l}
          \\{compound,prime\\} & \\{even,odd,quadratic\\} \\\\
          ...
        \\end{array}$
    """
    rows: list[str] = []
    for P in pseudo_intents(ctx):
        head = "\\{" + _format_set_inline(P) + "\\}"
        tail = "\\{" + _format_set_inline(_closure(ctx, P) - P) + "\\}"
        rows.append(f"  {head} & {tail} \\\\")
    body = "\n".join(rows)
    return (
        "$\\begin{array}{r@{\\;\\Rightarrow\\;}l}\n"
        f"{body}\n"
        "\\end{array}$"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("cxt_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path(),
              help="Output directory; created if missing.")
@click.option("--hide-objects", is_flag=True,
              help="Omit object labels (extent) from the Hasse diagram.")
@click.option("--skip-stem", is_flag=True,
              help="Skip the Stem-basis output.")
def render(cxt_path: str, out: str, hide_objects: bool, skip_stem: bool) -> None:
    """Render lattice SVG + Stem basis for a .cxt file."""
    ctx = load_context(Path(cxt_path))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(cxt_path).stem

    click.echo(f"Rendering lattice for {cxt_path} …")
    svg_data = hasse_render_svg(ctx, hide_objects=hide_objects)
    svg_path = out_dir / f"{name}_lattice.svg"
    svg_path.write_bytes(svg_data)
    click.echo(f"  Wrote {svg_path} ({len(svg_data)} bytes)")

    try:
        png_data = hasse_render_png(ctx, dpi=300, hide_objects=hide_objects)
        png_path = out_dir / f"{name}_lattice.png"
        png_path.write_bytes(png_data)
        click.echo(f"  Wrote {png_path} ({len(png_data)} bytes, 300 DPI)")
    except Exception as exc:
        click.echo(f"  [SKIP] PNG not generated: {exc}")

    if not skip_stem:
        stem_tex = stem_to_tikz(ctx)
        stem_path = out_dir / f"{name}_stem.tex"
        stem_path.write_text(stem_tex)
        click.echo(f"  Wrote {stem_path} ({len(stem_tex)} chars)")


if __name__ == "__main__":
    render()
