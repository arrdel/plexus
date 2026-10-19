"""Design reference for the mask-ladder figure: what the panels actually contain.

Not a paper figure. This exists so the annotations can be checked against the tensor rather than
against an impression of it, and so the layout can be redrawn by hand from exact coordinates.

Every cell colour is derived from the same `build_attention_bias` output the experiments use, and
every band is placed from `LinearisedGraph.spans` and `answer_pos` rather than typed. The token
ruler is printed to stdout as well as drawn, so the numbers are copyable.

Cell (i, j) is row i reading column j. A cell is coloured by *why* it is live, which is the thing
a black-and-white panel cannot show:

    grey    one endpoint is outside every span (instruction, suffix, [M]) -- exempt, always live
    orange  one endpoint is in the target's span
    blue    both endpoints are in the same node span (a diagonal block)
    green   two different node spans, i.e. a node-pair edge
    white   gated at -1e9

    python scripts/make_ladder_reference.py --out submission/figures

Known defects in the current `fig_mask_ladder` that this reference is drawn to expose:
  1. the answer row is marked with `axhline(answer_pos - 0.5)`, which is the *boundary above*
     row 47, so on the last row it renders as the bottom frame rather than as the row;
  2. the instruction prefix (0-3) and the suffix/[M] tail (46-47) carry no band, although their
     global visibility is the caption's main claim;
  3. the target-span marker outlines only the diagonal block 4-9 x 4-9, not the row and column
     bands that are the star mask's actual live entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import demo_graph  # noqa: E402

from plexus.masks import (  # noqa: E402
    MASK_BIAS,
    TARGET,
    LinearisedGraph,
    build_attention_bias,
    mask_density,
)

BLUE, ORANGE, GREEN = "#0173B2", "#DE8F05", "#029E73"
EXEMPT, GATED, GRID, INK, MUTED = "#CFD6DC", "#FFFFFF", "#B9C2CA", "#2E3338", "#7C858D"

FAMILIES = ["none", "star", "ego", "ego_star", "random_rewire"]
FAMILY_LABEL = {
    "none": "none\n(unmodified)", "star": "star\n(TAG-DLM)", "ego": "ego\n(observed edges)",
    "ego_star": "ego_star\n(ours)", "random_rewire": "random_rewire\n(density control)",
}

plt.rcParams.update({
    "font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.family": "serif", "mathtext.fontset": "cm",
})


def _hex(c):
    return tuple(int(c[k:k + 2], 16) / 255 for k in (1, 3, 5))


def bias_for(g: LinearisedGraph, family: str):
    """One seeded call, so `random_rewire` is the same graph everywhere it appears."""
    return build_attention_bias([g], family=family,
                                generator=torch.Generator().manual_seed(0))


def category_rgb(g: LinearisedGraph, family: str) -> np.ndarray:
    """(S, S, 3) image whose colour says why each entry is live."""
    live = (bias_for(g, family)[0, 0] > MASK_BIAS / 2).numpy()
    owner = g.span_of_token().numpy()

    S = g.seq_len
    img = np.ones((S, S, 3))
    oi = owner[:, None].repeat(S, 1)
    oj = owner[None, :].repeat(S, 0)

    exempt = (oi < 0) | (oj < 0)
    is_target = (oi == TARGET) | (oj == TARGET)
    same = oi == oj

    for mask, colour in (
        (live & exempt, EXEMPT),
        (live & ~exempt & is_target, ORANGE),
        (live & ~exempt & ~is_target & same, BLUE),
        (live & ~exempt & ~is_target & ~same, GREEN),
    ):
        img[mask] = _hex(colour)
    img[~live] = _hex(GATED)
    return img, live


def regions(g: LinearisedGraph):
    """The five bands, as inclusive token ranges, straight from the graph object."""
    hop1 = [i for i, h in enumerate(g.hops) if h == 1]
    hop2 = [i for i, h in enumerate(g.hops) if h == 2]
    out = [("instruction", 0, g.spans[TARGET][0] - 1, MUTED),
           ("$v_0$ target", *[g.spans[TARGET][0], g.spans[TARGET][1] - 1], ORANGE)]
    if hop1:
        out.append(("$\\mathcal{N}_1$", g.spans[hop1[0]][0], g.spans[hop1[-1]][1] - 1, BLUE))
    if hop2:
        out.append(("$\\mathcal{N}_2$", g.spans[hop2[0]][0], g.spans[hop2[-1]][1] - 1, GREEN))
    tail_lo = g.spans[-1][1]
    out.append(("suffix, [M]", tail_lo, g.seq_len - 1, INK))
    return out


def draw_bands(ax, g, *, lw=3.2, offset=1.4, labels=True, fs=6.4):
    """Region bands below and left of the matrix, on exact cell boundaries.

    Labels are staggered over two rows: `instruction` is four cells wide and its label is not,
    so a single row of labels collides with the target's.
    """
    S = g.seq_len
    for k, (name, lo, hi, colour) in enumerate(regions(g)):
        ax.plot([lo - 0.5, hi + 0.5], [S - 0.5 + offset] * 2, color=colour, lw=lw,
                solid_capstyle="butt", clip_on=False)
        ax.plot([-0.5 - offset] * 2, [lo - 0.5, hi + 0.5], color=colour, lw=lw,
                solid_capstyle="butt", clip_on=False)
        if labels:
            tier = offset + 1.6 + (2.6 if k % 2 else 0.0)
            ax.text((lo + hi) / 2, S - 0.5 + tier, name, ha="center", va="top",
                    fontsize=fs, color=colour, clip_on=False)


def draw_annotations(ax, g, *, detailed: bool):
    """The two markers the paper figure gets wrong, placed from the graph object."""
    ts, te = g.spans[TARGET]
    ap = g.answer_pos

    # Target span: the row band and the column band are the star's live entries, so mark those,
    # with the diagonal block picked out inside them.
    ax.add_patch(Rectangle((-0.5, ts - 0.5), g.seq_len, te - ts, fill=False, ec=ORANGE,
                           lw=1.1, zorder=5))
    ax.add_patch(Rectangle((ts - 0.5, -0.5), te - ts, g.seq_len, fill=False, ec=ORANGE,
                           lw=1.1, zorder=5))
    ax.add_patch(Rectangle((ts - 0.5, ts - 0.5), te - ts, te - ts, fill=False, ec=ORANGE,
                           lw=2.0, zorder=6))

    # The answer row is a row, not the boundary above it: y from ap-0.5 to ap+0.5.
    ax.add_patch(Rectangle((-0.5, ap - 0.5), g.seq_len, 1.0, fill=False, ec="#D55E00",
                           lw=1.6, zorder=6))

    for s, _e in g.spans:                                # span boundaries, both axes
        ax.axhline(s - 0.5, lw=0.35, color=GRID, zorder=4)
        ax.axvline(s - 0.5, lw=0.35, color=GRID, zorder=4)
    last = g.spans[-1][1]
    ax.axhline(last - 0.5, lw=0.35, color=GRID, zorder=4)
    ax.axvline(last - 0.5, lw=0.35, color=GRID, zorder=4)

    if detailed:
        # Callouts go in the two large gated blocks, which are empty by construction: hop-1 rows
        # against hop-2 columns and the reverse. Outside the axes they would land on the panel
        # next door, and on the diagonal they would cover a span block.
        n1_lo, n2_lo = g.spans[1][0], g.spans[4][0]
        ax.annotate(f"target span: rows {ts}\u2013{te - 1}\nand columns {ts}\u2013{te - 1}",
                    xy=(te + 1.0, te - 0.5), xytext=(n2_lo + 1.0, n1_lo + 4.0),
                    fontsize=7.0, color=ORANGE, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=ORANGE))
        ax.annotate(f"[M] answer row is row {ap}\nitself, not the boundary\nat {ap - 0.5}",
                    xy=(n1_lo + 3.0, ap), xytext=(n1_lo + 1.0, n2_lo + 3.0),
                    fontsize=7.0, color="#D55E00", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#D55E00"))


def ruler(ax, g, *, fs=5.8):
    """Indices on the top and right, so they cannot collide with the bands below and left."""
    ticks = [0, *[s for s, _ in g.spans], g.spans[-1][1]]
    ax.set_xticks(ticks)                       # answer_pos omitted: 46 and 47 collide here
    ax.set_yticks([*ticks, g.answer_pos])      # but on the query axis it is the row that matters
    ax.set_xticklabels(ticks, fontsize=fs)
    ax.set_yticklabels([*ticks, g.answer_pos], fontsize=fs)
    ax.xaxis.set_ticks_position("top")
    ax.yaxis.set_ticks_position("right")
    ax.tick_params(length=2, pad=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(INK)


def fig_reference(out: Path, stem: str = "mask_ladder_reference"):
    g = demo_graph()
    fig = plt.figure(figsize=(11.0, 8.6))
    gs = fig.add_gridspec(2, len(FAMILIES), height_ratios=[2.05, 1.0],
                          hspace=0.34, wspace=0.16)

    # ---- top: one fully annotated panel ----
    axb = fig.add_subplot(gs[0, :3])
    img, live = category_rgb(g, "star")
    axb.imshow(img, interpolation="nearest")
    draw_annotations(axb, g, detailed=True)
    draw_bands(axb, g, lw=4.0, offset=1.6)
    ruler(axb, g)
    axb.set_xlim(-0.5, g.seq_len - 0.5)
    axb.set_ylim(g.seq_len - 0.5, -0.5)
    axb.xaxis.set_label_position("top")
    axb.yaxis.set_label_position("right")
    axb.set_xlabel("key $j$ (what is read)", fontsize=8, labelpad=6)
    axb.set_ylabel("query $i$ (the reader)", fontsize=8, labelpad=8)
    axb.set_title(f"star (TAG-DLM), $S={g.seq_len}$, "
                  f"$\\rho={mask_density(bias_for(g, 'star')):.4f}$",
                  fontsize=9.5, pad=14)

    # ---- top right: the layout, as numbers ----
    axt = fig.add_subplot(gs[0, 3:])
    axt.axis("off")
    lines = [r"$\bf{Token\ layout}$ (from LinearisedGraph, not typed)", ""]
    for name, lo, hi, _c in regions(g):
        plain = name.replace("$", "").replace("\\mathcal{N}", "N").replace("_", "")
        lines.append(f"  {lo:>2}\u2013{hi:<2}   {plain}")
    lines += ["", f"  answer_pos = {g.answer_pos}   (the last token)",
              f"  seq_len    = {g.seq_len}", f"  spans      = {g.spans}", "",
              r"$\bf{Per-family\ densities}$", ""]
    for fam in FAMILIES:
        lines.append(f"  {fam:<14s} rho = {mask_density(bias_for(g, fam)):.4f}")
    lines += ["", r"$\bf{What\ the\ current\ figure\ gets\ wrong}$", "",
              "  1. answer row drawn at y = answer_pos - 0.5,",
              "     i.e. the boundary above row 47, so it",
              "     coincides with the bottom frame",
              "  2. instruction (0\u20133) and suffix/[M] (46\u201347)",
              "     have no band, though their global",
              "     visibility is the caption's main claim",
              "  3. target marker outlines only the diagonal",
              "     block, not the row/column bands that are",
              "     the star's actual live entries"]
    axt.text(0.0, 1.0, "\n".join(lines), fontsize=6.6, family="monospace", color=INK,
             ha="left", va="top", linespacing=1.4, transform=axt.transAxes)

    # ---- bottom: all five families ----
    for k, fam in enumerate(FAMILIES):
        ax = fig.add_subplot(gs[1, k])
        img, _ = category_rgb(g, fam)
        ax.imshow(img, interpolation="nearest")
        draw_annotations(ax, g, detailed=False)
        draw_bands(ax, g, lw=2.4, offset=1.2, labels=False)
        ax.set_title(f"{FAMILY_LABEL[fam]}\n$\\rho$ = "
                     f"{mask_density(bias_for(g, fam)):.4f}", fontsize=7.0,
                     pad=4, linespacing=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_linewidth(0.6)
            ax.spines[side].set_color(INK)
        ax.set_xlim(-0.5, g.seq_len - 0.5)
        ax.set_ylim(g.seq_len - 0.5, -0.5)

    fig.legend(handles=[
        Patch(fc=EXEMPT, ec=GRID, label="live: one endpoint outside every span (exempt)"),
        Patch(fc=ORANGE, ec="none", label="live: touches the target's span"),
        Patch(fc=BLUE, ec="none", label="live: same node span (diagonal block)"),
        Patch(fc=GREEN, ec="none", label="live: two different node spans (an edge)"),
        Patch(fc="white", ec=GRID, label="gated at $-10^{9}$"),
        Patch(fc="none", ec="#D55E00", label="[M] answer row"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3, frameon=False, fontsize=7.6)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.jpg", dpi=300, pil_kwargs={"quality": 95})
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"  {stem}.jpg (300 dpi) + {stem}.pdf")

    print("\ntoken layout, for redrawing by hand:")
    for name, lo, hi, _c in regions(g):
        plain = name.replace("$", "").replace("\\mathcal{N}", "N")
        print(f"  tokens {lo:>2}-{hi:<2}  {plain}")
    print(f"  answer_pos = {g.answer_pos}, seq_len = {g.seq_len}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("submission/figures"))
    a = p.parse_args()
    fig_reference(a.out)


if __name__ == "__main__":
    main()
