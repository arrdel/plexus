"""Figure: the four ways structure gets handed to a language model.

A taxonomy panel, not a results panel -- but the one schematic that depicts *our* method (the
gated attention matrix) is rendered from `plexus.masks.build_attention_bias` rather than drawn
by hand, so the knocked-out pattern cannot drift away from the mask the harness actually builds.

All four schematics are laid out inside an identical 56 x 22 unit frame on a single equal-aspect
axes, so no quadrant can end up physically larger than another by accident.

    python scripts/make_taxonomy_figure.py --out submission/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from plexus.masks import MASK_BIAS, LinearisedGraph, build_attention_bias  # noqa: E402

BLUE, ORANGE = "#0173B2", "#DE8F05"
INK, MUTED, HAIR, FAINT = "#4A4F55", "#9AA0A6", "#C9CED6", "#EDEFF2"

plt.rcParams.update({
    "font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "font.family": "serif", "mathtext.fontset": "cm",
})

# Schematic frame: every quadrant gets exactly this much room for its drawing.
SCHEM_W, SCHEM_H = 56.0, 22.0


# --------------------------------------------------------------------------------------------
# Primitives, all in axes data units so that equal numbers mean equal physical size
# --------------------------------------------------------------------------------------------

def box(ax, x, y, w, h, label="", *, ec=BLUE, fc="white", lw=0.9, fs=5.6, fc_text=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                ec=ec, fc=fc, lw=lw, zorder=3))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=fc_text, zorder=4, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, *, color=MUTED, lw=0.8, head=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_width={head / 6:.2f},"
                                 f"head_length={head / 4:.2f}", mutation_scale=6,
                                 ec=color, fc=color, lw=lw, shrinkA=0, shrinkB=0, zorder=4))


def token_strip(ax, x, y, n, *, w=2.6, gap=0.6, h=5.0, ec=HAIR, fc="white", accent=None):
    """A run of token cells. `accent` indexes one cell to outline in the accent colour."""
    for k in range(n):
        colour = ORANGE if accent is not None and k == accent else ec
        ax.add_patch(Rectangle((x + k * (w + gap), y), w, h, ec=colour, fc=fc,
                               lw=1.0 if colour == ORANGE else 0.5, zorder=3))
    return x + n * (w + gap) - gap  # right edge


def mini_graph(ax, cx, cy, *, scale=1.0, r=1.15, colour=BLUE, target=None):
    """A five-node blob with thin undirected edges. Positions are fixed so it reads the same
    in every quadrant; only `scale` changes."""
    pos = [(-1.0, 0.15), (-0.15, 0.95), (0.05, -0.75), (0.95, 0.45), (0.8, -0.85)]
    pos = [(cx + px * 6.0 * scale, cy + py * 6.0 * scale) for px, py in pos]
    for u, v in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=HAIR, lw=0.6, zorder=2)
    for k, (px, py) in enumerate(pos):
        fill = ORANGE if k == target else "white"
        edge = ORANGE if k == target else colour
        ax.add_patch(Circle((px, py), r * scale, ec=edge, fc=fill, lw=0.9, zorder=3))
    return pos


def caption(ax, x, y, s, *, fs=5.2, color=MUTED, ha="center"):
    ax.text(x, y, s, ha=ha, va="top", fontsize=fs, color=color, zorder=5)


# --------------------------------------------------------------------------------------------
# The four schematics. Each receives the bottom-left corner of its identical frame.
# --------------------------------------------------------------------------------------------

def schem_verbalisation(ax, bx, by):
    """A graph collapsing into a run of text tokens."""
    mid = by + SCHEM_H / 2 + 1.0
    pos = mini_graph(ax, bx + 9.5, mid, scale=0.72)

    waist = (bx + 23.0, mid)
    for k in (1, 3, 4):
        ax.plot([pos[k][0], waist[0]], [pos[k][1], waist[1]], color=HAIR, lw=0.5,
                ls=(0, (2.5, 1.5)), zorder=2)
    arrow(ax, waist[0], mid, bx + 26.6, mid)
    right = token_strip(ax, bx + 27.0, mid - 2.5, 8)
    caption(ax, (bx + 27.0 + right) / 2, mid - 4.4, "\"Title: ... cites: ...\"")


def schem_projection(ax, bx, by):
    """A graph feeding a small GNN box whose output arrow enters a token strip."""
    mid = by + SCHEM_H / 2 + 1.0
    mini_graph(ax, bx + 7.5, mid, scale=0.58, r=1.25)
    arrow(ax, bx + 13.0, mid, bx + 17.6, mid)
    box(ax, bx + 18.0, mid - 3.5, 10.5, 7.0, "GNN")
    arrow(ax, bx + 29.0, mid, bx + 33.6, mid)
    right = token_strip(ax, bx + 34.0, mid - 2.5, 6, accent=0)
    caption(ax, (bx + 34.0 + right) / 2, mid - 4.4, "one slot is a vector")


def schem_distillation(ax, bx, by):
    """A large model box producing text that feeds a visibly smaller model box."""
    mid = by + SCHEM_H / 2 + 1.0
    box(ax, bx + 2.0, mid - 7.0, 14.0, 14.0, "large\nLM")
    arrow(ax, bx + 16.5, mid, bx + 20.6, mid)
    right = token_strip(ax, bx + 21.0, mid - 2.5, 4, h=5.0)
    caption(ax, (bx + 21.0 + right) / 2, mid - 4.4, "explanation")
    arrow(ax, right + 1.4, mid, right + 5.5, mid)
    box(ax, right + 6.0, mid - 3.5, 9.0, 7.0, "small\nLM", fs=5.0)


def star_allow(n_nodes: int = 4, tokens_per_node: int = 2) -> torch.Tensor:
    """The star mask itself, straight out of the code path the experiments use."""
    spans = [(k * tokens_per_node, (k + 1) * tokens_per_node) for k in range(n_nodes)]
    seq_len = n_nodes * tokens_per_node
    g = LinearisedGraph(spans=spans, seq_len=seq_len,
                        edges={(0, k) for k in range(1, n_nodes)},
                        node_ids=list(range(n_nodes)), pad_from=seq_len)
    bias = build_attention_bias([g], family="star", dtype=torch.float32)
    return bias[0, 0] > MASK_BIAS / 2


def schem_attention_gating(ax, bx, by):
    """A token strip with a square attention matrix above it, some cells knocked out."""
    allow = star_allow().numpy()
    n = allow.shape[0]
    cell = 2.0
    mw = n * cell
    mx = bx + (SCHEM_W - mw) / 2.0
    strip_h, gap = 3.2, 2.2
    my = by + 1.6 + strip_h + gap

    for i in range(n):
        for j in range(n):
            on = bool(allow[i, j])
            ax.add_patch(Rectangle((mx + j * cell, my + (n - 1 - i) * cell), cell, cell,
                                   ec="white" if on else HAIR, fc=ORANGE if on else "white",
                                   lw=0.7 if on else 0.4, zorder=3))
    ax.add_patch(Rectangle((mx, my), mw, mw, ec=MUTED, fc="none", lw=0.5, zorder=4))

    for k in range(n):
        ax.add_patch(Rectangle((mx + k * cell, by + 1.6), cell, strip_h,
                               ec=HAIR, fc="white", lw=0.4, zorder=3))
    for xx in (mx, mx + mw):
        ax.plot([xx, xx], [by + 1.6 + strip_h, my], color=FAINT, lw=0.5, zorder=1)

    ax.text(mx - 1.6, my + mw / 2, "reads", rotation=90, ha="right", va="center",
            fontsize=5.0, color=MUTED)
    # Point at a cell the mask really gates off, rather than a hand-picked index that a change
    # of span layout could quietly turn into a permitted one.
    gated = [(i, j) for i in range(n) for j in range(n) if not allow[i, j]]
    gi, gj = min(gated, key=lambda t: (t[0], -t[1]))
    ax.annotate("gated", xy=(mx + (gj + 0.5) * cell, my + (n - 1 - gi + 0.5) * cell),
                xytext=(mx + mw + 3.0, my + mw - 1.0), fontsize=5.0, color=MUTED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.5, color=MUTED, shrinkA=1, shrinkB=1))
    caption(ax, mx + mw / 2, by + 0.9, "same tokens, restricted reading")


PANELS = [
    ("Verbalisation", "the neighbourhood is written into the prompt", schem_verbalisation),
    ("Projection", "a GNN embedding is spliced into the token stream", schem_projection),
    ("Distillation", "a large model explains, a smaller model consumes", schem_distillation),
    ("Attention gating", "structure edits which token may read which", schem_attention_gating),
]


def fig_four_ways(out: Path, stem: str = "four_ways", png: bool = False):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(3, 96.5, "Four ways to hand a graph to a language model",
            fontsize=13, color=INK, ha="left", va="center")
    ax.text(3, 91.5, "four different places to inject the same neighbourhood",
            fontsize=7, color=MUTED, ha="left", va="center")

    qw, qh = 65.0, 41.0
    origins = [(3.0, 44.0), (72.0, 44.0), (3.0, 1.5), (72.0, 1.5)]

    for (title, desc, draw), (qx, qy) in zip(PANELS, origins, strict=True):
        is_ours = draw is schem_attention_gating
        ax.add_patch(FancyBboxPatch((qx, qy), qw, qh,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    ec=ORANGE if is_ours else FAINT, fc="none",
                                    lw=1.2 if is_ours else 0.8, zorder=2))
        ax.text(qx + 3.0, qy + qh - 4.5, title, fontsize=9, color=INK, ha="left", va="center")
        ax.text(qx + 3.0, qy + qh - 9.5, desc, fontsize=6.2, color=MUTED, ha="left", va="center")

        if is_ours:
            ax.add_patch(FancyBboxPatch((qx + qw - 18.0, qy + qh - 7.2), 15.0, 5.0,
                                        boxstyle="round,pad=0,rounding_size=2.4",
                                        ec=ORANGE, fc="white", lw=0.8, zorder=4))
            ax.text(qx + qw - 10.5, qy + qh - 4.7, "this paper", fontsize=5.8,
                    color=ORANGE, ha="center", va="center", zorder=5)

        # Identical frame for every schematic; the guide box makes the equality visible.
        sx = qx + (qw - SCHEM_W) / 2.0
        sy = qy + 4.0
        ax.add_patch(Rectangle((sx, sy), SCHEM_W, SCHEM_H, ec=FAINT, fc="none",
                               lw=0.5, zorder=1))
        draw(ax, sx, sy)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    if png:
        fig.savefig(out / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"  {stem}.pdf" + (f" + {stem}.png" if png else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("submission/figures"))
    p.add_argument("--png", action="store_true", help="also write a PNG for slides/preview")
    a = p.parse_args()
    fig_four_ways(a.out, png=a.png)


if __name__ == "__main__":
    main()
