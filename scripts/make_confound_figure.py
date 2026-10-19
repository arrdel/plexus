"""Figure: star vs ego differ in three ways at once, so the collapse is unattributable.

Glyphs rather than prose, one row per difference. The density row is not drawn to taste: the
proportions are parsed out of `submission/results_generated.tex`, so the bars move when the
measured densities move.

That parse matters more than it looks. On a single small ego graph -- the one `make_figures.py`
draws the mask ladder from -- `ego` comes out *denser* than `star`, because the target happens to
have high degree there. Averaged over the 300 real Cora examples it is the other way round, which
is the direction the paper's argument depends on. A hand-typed bar would have had a coin flip's
chance of pointing the wrong way.

    python scripts/make_confound_figure.py --out submission/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

BLUE, ORANGE = "#0173B2", "#DE8F05"
INK, MUTED, HAIR, FAINT = "#2E3338", "#9AA0A6", "#C9CED6", "#E8EBEE"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "font.family": "serif", "mathtext.fontset": "cm",
})

_ROW = re.compile(r"\\texttt\{([a-z_\\]+)\}\s*&\s*([0-9.]+).*?&\s*([0-9.]+)")


def read_densities(results: Path, task: str = "lp") -> dict[str, float]:
    """Pull the measured mask densities out of the generated results macros.

    Returns family -> rho for the requested task column ("nc" or "lp").
    """
    text = results.read_text()
    m = re.search(r"\\newcommand\{\\densitybody\}\{(.*?)\n\}", text, re.S)
    if m is None:
        raise ValueError(f"no \\densitybody macro in {results}; run the results pipeline first")
    col = 1 if task == "nc" else 2
    out = {}
    for line in m.group(1).splitlines():
        hit = _ROW.search(line)
        if hit:
            out[hit.group(1).replace("\\_", "_")] = float(hit.group(col + 1))
    missing = {"star", "ego"} - out.keys()
    if missing:
        raise ValueError(f"density table is missing {sorted(missing)}")
    return out


# --------------------------------------------------------------------------------------------
# Glyphs. Each is handed the centre of its cell and draws inside a fixed 30 x 16 unit box.
# --------------------------------------------------------------------------------------------

def glyph_neighbour_edges(ax, cx, cy, *, connected: bool):
    """Two nodes, with or without the edge between them."""
    dx, r = 6.5, 2.6
    if connected:
        ax.plot([cx - dx + r, cx + dx - r], [cy, cy], color=BLUE, lw=1.3, zorder=2)
    for sx in (-dx, dx):
        ax.add_patch(Circle((cx + sx, cy), r, ec=BLUE, fc="white", lw=1.2, zorder=3))


def glyph_target_reach(ax, cx, cy, *, broken: int = 0):
    """Hub and spokes; `broken` of the outer spokes are cut, their leaf greyed out.

    The cut spokes are the hop-2 leaves, which is what `ego` actually costs the target: with no
    edge to the target, a hop-2 node is unreadable from node 0.
    """
    leaves = [(-8.2, 4.2), (-8.2, -4.4), (0.4, 6.4), (8.2, 4.2), (8.2, -4.4)]
    cut = set(range(len(leaves) - broken, len(leaves)))
    hub_r, leaf_r = 2.5, 1.7
    for k, (lx, ly) in enumerate(leaves):
        x1, y1 = cx + lx, cy + ly
        if k in cut:
            for t0, t1 in ((0.24, 0.48), (0.74, 0.92)):
                ax.plot([cx + (x1 - cx) * t0, cx + (x1 - cx) * t1],
                        [cy + (y1 - cy) * t0, cy + (y1 - cy) * t1],
                        color=HAIR, lw=1.0, zorder=2)
        else:
            ax.plot([cx, x1], [cy, y1], color=BLUE, lw=1.0, alpha=0.75, zorder=2)
        edge = HAIR if k in cut else BLUE
        ax.add_patch(Circle((x1, y1), leaf_r, ec=edge, fc="white", lw=1.0, zorder=3))
    ax.add_patch(Circle((cx, cy), hub_r, ec=BLUE, fc=BLUE, lw=1.0, zorder=4))


def glyph_density(ax, cx, cy, *, rho: float):
    """A proportion bar, filled to the measured density."""
    w, h = 24.0, 4.6
    x0 = cx - w / 2
    ax.add_patch(Rectangle((x0, cy - h / 2), w, h, ec=HAIR, fc="white", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x0, cy - h / 2), w * rho, h, ec="none", fc=BLUE, zorder=3))
    ax.text(cx, cy - h / 2 - 1.6, f"{rho:.3f}", fontsize=7.5, color=MUTED,
            ha="center", va="top")


def bracket(ax, x, y0, y1, label, *, colour=ORANGE):
    """A right-hand bracket spanning y0..y1, with the label outside its middle nub."""
    ax.plot([x, x], [y0, y1], color=colour, lw=1.1, zorder=3)
    for yy in (y0, y1):
        ax.plot([x - 2.6, x], [yy, yy], color=colour, lw=1.1, zorder=3)
    ymid = (y0 + y1) / 2
    ax.plot([x, x + 2.6], [ymid, ymid], color=colour, lw=1.1, zorder=3)
    ax.text(x + 4.4, ymid, label, rotation=90, rotation_mode="anchor",
            fontsize=9, color=colour, ha="center", va="bottom", zorder=3)


ROWS = [
    ("neighbour\u2013neighbour edges", "present among sampled nodes?"),
    ("target reach", "can node 0 read a hop-2 node?"),
    ("mask density", "Cora link prediction, mean of 300"),
]


def fig_confounds(out: Path, results: Path, stem: str = "three_confounds",
                  png: bool = False, task: str = "lp"):
    rho = read_densities(results, task=task)

    W, H = 142.0, 90.0
    fig, ax = plt.subplots(figsize=(W / 20.0, H / 20.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    label_x = 42.0
    star_cx, ego_cx = 65.0, 103.0
    row_y = [66.0, 47.0, 28.0]
    grid_x0, grid_x1 = 46.0, 122.0

    for cx, name in ((star_cx, "star"), (ego_cx, "ego")):
        ax.text(cx, 80.5, name, fontsize=11.5, color=INK, ha="center", va="center",
                family="monospace")
    ax.plot([grid_x0, grid_x1], [76.5, 76.5], color=HAIR, lw=0.9)
    for yy in (56.5, 37.5, 18.5):
        ax.plot([grid_x0, grid_x1], [yy, yy], color=FAINT, lw=0.7)
    ax.plot([84.0, 84.0], [18.5, 76.5], color=FAINT, lw=0.7)

    for (main, sub), yy in zip(ROWS, row_y, strict=True):
        ax.text(label_x, yy + 1.4, main, fontsize=9.5, color=INK, ha="right", va="center")
        ax.text(label_x, yy - 3.4, sub, fontsize=7, color=MUTED, ha="right", va="center")

    glyph_neighbour_edges(ax, star_cx, row_y[0], connected=False)
    glyph_neighbour_edges(ax, ego_cx, row_y[0], connected=True)
    glyph_target_reach(ax, star_cx, row_y[1], broken=0)
    glyph_target_reach(ax, ego_cx, row_y[1], broken=2)
    glyph_density(ax, star_cx, row_y[2], rho=rho["star"])
    glyph_density(ax, ego_cx, row_y[2], rho=rho["ego"])

    bracket(ax, 126.0, 18.5, 76.5, "all three vary together")

    ax.text(4.0, 11.0, "the reported comparison cannot attribute the collapse to any one of them",
            fontsize=10.5, color=ORANGE, ha="left", va="top")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    if png:
        fig.savefig(out / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"  {stem}.pdf" + (f" + {stem}.png" if png else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("submission/figures"))
    p.add_argument("--results", type=Path, default=Path("submission/results_generated.tex"))
    p.add_argument("--task", choices=["nc", "lp"], default="lp")
    p.add_argument("--png", action="store_true", help="also write a PNG for slides/preview")
    a = p.parse_args()
    fig_confounds(a.out, a.results, png=a.png, task=a.task)


if __name__ == "__main__":
    main()
