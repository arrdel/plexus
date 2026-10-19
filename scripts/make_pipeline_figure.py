"""Figure: the whole pipeline on one baseline, with the contribution boxed off.

Slide version of the overview. The argument the figure has to make is that only (c) varies:
the sampling, the linearisation, the backbone and the scoring rule are identical across every
arm, so the dashed enclosure is the most important mark on it and everything else is context.

Two things are drawn rather than asserted. The bias tensor fans into all 32 blocks, because the
same B is added at every layer -- drawing it into the first block only would suggest a one-shot
injection. And the (a)-(c) span gets 60% of the content width against (d)-(e)'s 40%, computed
from BUDGET below and checked at draw time.

    python scripts/make_pipeline_figure.py --out submission/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

BLUE, ORANGE = "#0173B2", "#DE8F05"
INK, MUTED, HAIR, FAINT = "#2E3338", "#9AA0A6", "#C9CED6", "#E8EBEE"
SPAN_TINTS = ["#7FBEDC", "#A8D2E6", "#C6E1EF"]      # v1..v3, lighter with hop distance

N_BLOCKS = 32
LORA_CAPTION = "frozen, LoRA $r{=}64$, 167.8M trainable, 2.09%"
BUDGET = 0.60                                        # share of content width for (a)-(c)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "font.family": "serif", "mathtext.fontset": "cm",
})


def arrow(ax, x0, y0, x1, y1, *, colour=MUTED, lw=0.8, scale=7, z=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>,head_width=0.34,head_length=0.55",
                                 mutation_scale=scale, ec=colour, fc=colour, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def leader(ax, x0, y0, x1, y1, *, colour=HAIR, lw=0.55):
    ax.plot([x0, x1], [y0, y1], color=colour, lw=lw, zorder=2)


def lock_glyph(ax, cx, cy, *, w=3.4, h=2.8, colour=MUTED):
    ax.add_patch(Arc((cx, cy + h / 2), w * 0.66, h * 1.05, theta1=0, theta2=180,
                     ec=colour, lw=0.9, zorder=4))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.4",
                                ec=colour, fc="white", lw=0.9, zorder=5))
    ax.add_patch(Circle((cx, cy + 0.1), 0.42, ec=colour, fc=colour, lw=0, zorder=6))


# --------------------------------------------------------------------------------------------
# (a) the sampled subgraph
# --------------------------------------------------------------------------------------------

def stage_a(ax, x0, x1, cy):
    cx = (x0 + x1) / 2
    # target, three hop-1, two hop-2; one hop-1 pair is adjacent.
    nodes = {
        "v0": (cx - 0.5, cy + 1.0),
        "v1": (cx - 11.0, cy + 8.5),
        "v2": (cx + 10.5, cy + 8.0),
        "v3": (cx + 1.5, cy - 9.5),
        "v4": (cx - 15.5, cy - 2.0),
        "v5": (cx + 15.0, cy - 4.0),
    }
    edges = [("v0", "v1"), ("v0", "v2"), ("v0", "v3"), ("v1", "v4"), ("v2", "v5")]
    for u, v in edges:
        leader(ax, *nodes[u], *nodes[v], colour=HAIR, lw=0.7)
    u, v = nodes["v1"], nodes["v2"]
    ax.plot([u[0], v[0]], [u[1], v[1]], color=ORANGE, lw=0.9, ls=(0, (2.6, 1.6)), zorder=2)

    for name, (px, py) in nodes.items():
        target = name == "v0"
        ax.add_patch(Circle((px, py), 3.1 if target else 2.5,
                            ec=ORANGE if target else BLUE,
                            fc=ORANGE if target else "white", lw=1.1, zorder=3))
        ax.text(px, py - (5.0 if target else 4.2), name, fontsize=5.2,
                color=MUTED, ha="center", va="top")

    ax.text(cx, cy - 16.5, "one neighbour\u2013neighbour edge:\nego keeps it, star does not",
            fontsize=5.6, color=MUTED, ha="center", va="top", linespacing=1.4)


# --------------------------------------------------------------------------------------------
# (b) linearisation
# --------------------------------------------------------------------------------------------

def stage_b(ax, x0, x1, cy, *, n_spans=4):
    """Segment order follows plexus/prompts/templates.py: instruction, node spans, then [M]."""
    h = 9.0
    segs = [("instruction", 12.0, "#EFF1F3", MUTED)]
    for k in range(n_spans):
        tint = ORANGE if k == 0 else SPAN_TINTS[min(k - 1, len(SPAN_TINTS) - 1)]
        segs.append((f"$v_{k}$", 8.5, tint, INK))
    segs.append(("[M]", 6.0, "#E4E7EA", INK))

    total = sum(w for _, w, _, _ in segs)
    x = x0 + ((x1 - x0) - total) / 2
    bounds = {}
    for name, w, fill, fg in segs:
        terminal = name == "[M]"
        ax.add_patch(Rectangle((x, cy - h / 2), w, h, ec=INK if terminal else "white",
                               fc=fill, lw=1.3 if terminal else 0.8, zorder=3))
        ax.text(x + w / 2, cy, name, fontsize=6.0 if terminal else 6.4, color=fg,
                ha="center", va="center", zorder=4,
                family="monospace" if terminal else None)
        bounds[name] = (x, x + w)
        x += w

    ax.text((x0 + x1) / 2, cy + h / 2 + 2.0, "one contiguous span per node (title + abstract)",
            fontsize=5.6, color=MUTED, ha="center", va="bottom")

    note_y = cy - h / 2 - 11.0
    ax.text((x0 + x1) / 2, note_y, "outside every span", fontsize=6.0, color=MUTED,
            ha="center", va="center")
    label_half = 13.0
    for name in ("instruction", "[M]"):
        lo, hi = bounds[name]
        side = -1 if name == "instruction" else 1
        leader(ax, (x0 + x1) / 2 + side * label_half, note_y,
               (lo + hi) / 2, cy - h / 2 - 0.8)
    ax.text((x0 + x1) / 2, note_y - 3.2,
            "globally visible under every family", fontsize=5.2, color=MUTED,
            ha="center", va="top")


# --------------------------------------------------------------------------------------------
# (c) the structure operator -- the only thing that varies
# --------------------------------------------------------------------------------------------

FAMILIES = [
    ("none", "anchor arm", False),
    ("star", "baseline default", False),
    ("ego", "observed edges", False),
    ("ego_star", "+ target reach", True),
    ("random_rewire", "density control", False),
]
GRADED = ("HopBias", "learned $\\lambda_d$", True)


def stage_c(ax, x0, x1, cy):
    bw, bh = 24.0, 22.0
    bx = x0 + 2.0
    ax.add_patch(FancyBboxPatch((bx, cy - bh / 2), bw, bh,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                ec=BLUE, fc="#F2F8FB", lw=1.1, zorder=3))
    ax.text(bx + bw / 2, cy + 3.0, "$\\Phi$", fontsize=17, color=BLUE,
            ha="center", va="center", zorder=4)
    ax.text(bx + bw / 2, cy - 5.5, "structure\noperator", fontsize=5.8, color=MUTED,
            ha="center", va="center", linespacing=1.3, zorder=4)

    mx, mw, n = bx + bw + 4.0, 14.0, 6
    cell = mw / n
    arrow(ax, bx + bw + 0.6, cy, mx - 0.8, cy)
    allow = [[i == 0 or j == 0 or i == j for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            on = allow[i][j]
            ax.add_patch(Rectangle((mx + j * cell, cy + mw / 2 - (i + 1) * cell), cell, cell,
                                   ec="white" if on else HAIR, fc=ORANGE if on else "white",
                                   lw=0.5 if on else 0.35, zorder=3))
    ax.add_patch(Rectangle((mx, cy - mw / 2), mw, mw, ec=MUTED, fc="none", lw=0.5, zorder=4))
    ax.text(mx + mw / 2, cy + mw / 2 + 1.6, "$B \\in \\mathbb{R}^{S \\times S}$",
            fontsize=6.2, color=INK, ha="center", va="bottom")

    def family_row(yy, name, gloss, ours):
        if ours:
            ax.add_patch(Circle((x0 + 1.4, yy), 0.7, ec="none", fc=ORANGE, zorder=4))
        ax.text(x0 + 3.4, yy, name, fontsize=6.0, color=INK if ours else MUTED,
                ha="left", va="center", family="monospace")
        ax.text(x0 + 23.0, yy, gloss, fontsize=5.2, color=MUTED, ha="left", va="center")
        if ours:
            ax.text(x1 - 1.5, yy, "ours", fontsize=5.4, color=ORANGE, ha="right", va="center")

    top = cy - bh / 2 - 6.0
    for k, (name, gloss, ours) in enumerate(FAMILIES):
        family_row(top - k * 5.4, name, gloss, ours)

    rule_y = top - len(FAMILIES) * 5.4 - 1.2
    ax.plot([x0 + 3.4, x1 - 1.5], [rule_y, rule_y], color=FAINT, lw=0.7)
    family_row(rule_y - 4.2, *GRADED)
    return rule_y - 8.0, (mx + mw / 2, cy + mw / 2)


# --------------------------------------------------------------------------------------------
# (d) the frozen backbone, with B entering every block
# --------------------------------------------------------------------------------------------

def stage_d(ax, x0, x1, cy, *, bus_x, src):
    bar_x0, bar_x1 = x0 + 10.0, x1
    pitch, bar_h = 1.80, 1.15
    y_lo = cy - (N_BLOCKS * pitch) / 2
    ys = [y_lo + k * pitch for k in range(N_BLOCKS)]

    for yy in ys:
        ax.add_patch(Rectangle((bar_x0, yy), bar_x1 - bar_x0, bar_h,
                               ec=BLUE, fc="#EAF3F8", lw=0.4, zorder=3))

    # The same B is added at every layer, so it fans in everywhere, not just at the bottom.
    ax.plot([bus_x, bus_x], [ys[0] + bar_h / 2, ys[-1] + bar_h / 2], color=ORANGE,
            lw=1.0, zorder=3)
    for yy in ys:
        arrow(ax, bus_x, yy + bar_h / 2, bar_x0 - 0.5, yy + bar_h / 2,
              colour=ORANGE, lw=0.45, scale=3.2, z=4)
    sx, sy = src
    ax.plot([sx, sx, bus_x, bus_x], [sy, ys[-1] + bar_h / 2 + 6.0,
                                     ys[-1] + bar_h / 2 + 6.0, ys[-1] + bar_h / 2],
            color=ORANGE, lw=1.0, zorder=3)
    ax.text(bus_x - 1.0, ys[-1] + bar_h / 2 + 7.4, "$B$, added in every block",
            fontsize=5.8, color=ORANGE, ha="left", va="bottom")

    ax.text(bar_x1 + 1.2, ys[-1] + bar_h / 2, f"$\\times\\,{N_BLOCKS}$", fontsize=6.0,
            color=MUTED, ha="left", va="center")
    lock_glyph(ax, (bar_x0 + bar_x1) / 2, y_lo - 6.5)
    ax.text((bar_x0 + bar_x1) / 2, y_lo - 10.5, LORA_CAPTION, fontsize=5.8, color=MUTED,
            ha="center", va="top")
    return bar_x1


# --------------------------------------------------------------------------------------------
# (e) scoring
# --------------------------------------------------------------------------------------------

def stage_e(ax, x0, x1, cy):
    s = 8.0
    ax.add_patch(Rectangle((x0, cy - s / 2), s, s, ec=ORANGE, fc="#FDF3E3", lw=1.3, zorder=3))
    ax.text(x0 + s / 2, cy, "[M]", fontsize=5.6, color=INK, ha="center", va="center",
            family="monospace", zorder=4)
    arrow(ax, x0 + s + 1.0, cy, x0 + s + 7.0, cy)
    ax.text(x0 + s + 8.5, cy, "argmax over\nclass names", fontsize=6.6, color=INK,
            ha="left", va="center", linespacing=1.35)
    ax.text(x0, cy - s / 2 - 3.0, "one forward pass; logits read at the single [M]",
            fontsize=5.4, color=MUTED, ha="left", va="top")


# --------------------------------------------------------------------------------------------

STAGES = ["(a) sampled subgraph", "(b) linearise into one sequence",
          "(c) structure operator", "(d) LLaDA-8B, frozen", "(e) score one position"]


def fig_pipeline(out: Path, stem: str = "pipeline", png: bool = False):
    # Data units are square (aspect="equal"); the units-per-inch factor depends on the axes
    # box, not on figsize alone, so glyph widths are checked against the render rather than
    # computed from point sizes.
    W, H = 268.0, 116.0
    fig, ax = plt.subplots(figsize=(W / 20.0, H / 20.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    baseline = 70.0
    left, right = 8.0, 260.0
    gap = 6.0

    # Width budget: (a)-(c) take BUDGET of the content, (d)-(e) the rest.
    content = (right - left) - 2 * gap
    front, back = content * BUDGET, content * (1 - BUDGET)
    a_x0 = left
    a_x1 = a_x0 + front * 0.235
    b_x0, b_x1 = a_x1 + gap, a_x1 + gap + front * 0.375
    c_x0, c_x1 = b_x1 + gap, b_x1 + gap + front * 0.325
    d_x0 = c_x1 + gap
    d_x1 = d_x0 + back * 0.50
    e_x0 = d_x1 + gap

    for label, (sx0, sx1) in zip(STAGES, [(a_x0, a_x1), (b_x0, b_x1), (c_x0, c_x1),
                                          (d_x0, d_x1), (e_x0, right)], strict=True):
        ax.text((sx0 + sx1) / 2, H - 6.0, label, fontsize=7.6, color=INK,
                ha="center", va="center")
    ax.plot([left, right], [H - 11.0, H - 11.0], color=FAINT, lw=0.8)

    stage_a(ax, a_x0, a_x1, baseline)
    stage_b(ax, b_x0, b_x1, baseline)
    stack_bottom, bias_top = stage_c(ax, c_x0, c_x1, baseline)
    stage_d(ax, d_x0, d_x1, baseline, bus_x=d_x0 + 3.0, src=bias_top)
    stage_e(ax, e_x0, right, baseline)

    for sx1, nx0 in [(a_x1, b_x0), (b_x1, c_x0), (c_x1, d_x0), (d_x1, e_x0)]:
        arrow(ax, sx1 + 0.8, baseline, nx0 - 0.8, baseline)

    # The one annotation the figure exists for.
    enc_x0, enc_x1 = c_x0 - 3.5, c_x1 + 2.0
    enc_y0, enc_y1 = stack_bottom, baseline + 20.0
    ax.add_patch(FancyBboxPatch((enc_x0, enc_y0), enc_x1 - enc_x0, enc_y1 - enc_y0,
                                boxstyle="round,pad=0,rounding_size=2.0",
                                ec=ORANGE, fc="none", lw=1.3, ls=(0, (4.5, 2.8)), zorder=5))
    ax.text((enc_x0 + enc_x1) / 2, enc_y1, "the only thing that varies in this paper",
            fontsize=8.2, color=ORANGE, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none"))

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    if png:
        fig.savefig(out / f"{stem}.png", dpi=220)
    plt.close(fig)
    share = front / content
    print(f"  {stem}.pdf" + (f" + {stem}.png" if png else "")
          + f"   [(a)-(c) {share:.0%} / (d)-(e) {1 - share:.0%}]")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("submission/figures"))
    p.add_argument("--png", action="store_true", help="also write a PNG for slides/preview")
    a = p.parse_args()
    fig_pipeline(a.out, png=a.png)


if __name__ == "__main__":
    main()
