"""High-resolution design references for the three chart figures, rendered from the real code.

These are *not* the paper figures. They are oversized, heavily annotated versions of
`graded_bias.pdf`, `layer_reach.pdf` and `seq_lengths.pdf` intended to be redrawn by hand in a
design tool, so every quantity is printed on the artwork rather than left implicit. Each function
also lists, in its docstring, the defect in the shipped figure that it exists to correct.

    ~/miniconda3/envs/plexus/bin/python scripts/make_chart_references.py

Run with the `plexus` environment's interpreter; the base conda python cannot import `plexus`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_figures import demo_graph, reach_by_layer  # noqa: E402

from plexus.masks import (  # noqa: E402
    MASK_BIAS,
    TARGET,
    HopBias,
    LinearisedGraph,
    build_attention_bias,
)

BLUE, ORANGE, GREEN, RED = "#0173B2", "#DE8F05", "#029E73", "#D55E00"
INK, MUTED, HAIR, FAINT = "#2E3338", "#9AA0A6", "#C9CED6", "#EDEFF2"
GATED = "#9AA0A6"          # flat mid grey: "off", deliberately not part of the diverging scale
VMAX = 2.6

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------------------------
# Shared: the token layout of the demo graph, so both matrix figures share one vocabulary
# --------------------------------------------------------------------------------------------

def regions(g: LinearisedGraph) -> list[tuple[int, int, str, str]]:
    """(start, end, label, colour) bands, derived from the graph rather than typed in."""
    hop1 = [s for k, s in enumerate(g.spans) if k != TARGET and g.hops[k] == 1]
    hop2 = [s for k, s in enumerate(g.spans) if k != TARGET and g.hops[k] == 2]
    t0, t1 = g.spans[TARGET]
    out = [(0, t0, "instruction", MUTED), (t0, t1, "target $v_0$", ORANGE)]
    if hop1:
        out.append((hop1[0][0], hop1[-1][1], "hop-1 spans", BLUE))
    if hop2:
        out.append((hop2[0][0], hop2[-1][1], "hop-2 spans", GREEN))
    out.append((g.spans[-1][1], g.seq_len, "suffix, [M]", MUTED))
    return out


def draw_bands(ax, g: LinearisedGraph, thick: float = 1.9, gap: float = 1.1) -> None:
    """Band strips outside the matrix on both axes, so a cell can be located by eye."""
    s = g.seq_len
    for a, b, label, colour in regions(g):
        ax.add_patch(Rectangle((a - 0.5, s - 0.5 + gap), b - a, thick, facecolor=colour,
                               edgecolor="none", clip_on=False, zorder=5))
        ax.add_patch(Rectangle((-0.5 - gap - thick, a - 0.5), thick, b - a, facecolor=colour,
                               edgecolor="none", clip_on=False, zorder=5))
        ax.text((a + b) / 2 - 0.5, s - 0.5 + gap + thick + 0.9, label, ha="center", va="top",
                fontsize=6.0, color=INK, rotation=90, clip_on=False)


# --------------------------------------------------------------------------------------------
# Reference 1: the graded bias
# --------------------------------------------------------------------------------------------

def ref_graded_bias(out: Path) -> None:
    """Corrects four things in the shipped `graded_bias.pdf`.

    1. Gated entries are -1e9 but are clipped to -2.6 and painted with the same diverging
       colormap as the graded values, so "switched off" renders as "strongly negative". Here they
       are flat grey, off the scale, with the count printed.
    2. The bottom row uses a 34-bin histogram to show a distribution supported on at most five
       atoms, which turns exact counts into bar widths. Here each attained value is one bar at its
       own position, with its count printed, which is literally the multiset the caption claims.
    3. The log counts have no visible y-axis, so the bar heights cannot be read. Restored.
    4. No token-layout bands, so the reader cannot tell which block is the target span.
    """
    g = demo_graph()
    lin = HopBias(values=torch.tensor([0.0, -0.5, -1.0, -1.5, -2.0, -3.0]))
    shaped = HopBias(values=torch.tensor([0.0, 0.4, -0.2, -0.9, -1.6, -2.4]))

    def graded(hb):
        return build_attention_bias([g], family="none", hop_bias=hb, dtype=torch.float32)[0, 0]

    def binary(fam):
        return build_attention_bias([g], family=fam, dtype=torch.float32)[0, 0]

    panels = [
        ("star (binary)", binary("star"), r"$\{0,\,-10^9\}$"),
        ("ego (binary)", binary("ego"), r"$\{0,\,-10^9\}$"),
        (r"graded, $\lambda_d = -0.5\,d$", graded(lin), r"$\lambda = (0,-.5,-1,-1.5,-2,-3)$"),
        ("graded, arbitrary shape", graded(shaped), r"$\lambda = (0,.4,-.2,-.9,-1.6,-2.4)$"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 8.2),
                             gridspec_kw={"height_ratios": [2.5, 1.25], "hspace": 0.42,
                                          "wspace": 0.26})
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(GATED)
    im = None

    for j, (title, mat, sub) in enumerate(panels):
        m = mat.numpy()
        gated = m <= MASK_BIAS / 2
        ax = axes[0, j]
        im = ax.imshow(np.ma.masked_where(gated, np.clip(m, -VMAX, VMAX)), cmap=cmap,
                       vmin=-VMAX, vmax=VMAX, interpolation="nearest")
        for s, _e in g.spans:
            ax.axhline(s - 0.5, lw=0.35, color=INK, alpha=0.30)
            ax.axvline(s - 0.5, lw=0.35, color=INK, alpha=0.30)
        draw_bands(ax, g)
        ax.set_title(title, fontsize=10.5, pad=26, color=INK)
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0,
                color=MUTED)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color(HAIR)
        if j == 0:
            ax.set_ylabel("query token", fontsize=9, labelpad=16, color=INK)
        # The band labels hang ~50pt below the axes, so the xlabel has to clear them.
        ax.set_xlabel("key token", fontsize=9, labelpad=54, color=INK)

        # The multiset, drawn as the discrete object it is: one bar per attained value.
        axh = axes[1, j]
        finite = m[~gated]
        vals, counts = np.unique(np.round(finite, 4), return_counts=True)
        n_gated = int(gated.sum())
        axh.bar(vals, counts, width=0.10, color=BLUE if len(vals) > 1 else MUTED, zorder=3)
        for v, c in zip(vals, counts, strict=True):
            axh.text(v, c * 1.6, f"{c}", ha="center", va="bottom", fontsize=7.4, color=INK,
                     rotation=90, zorder=4)
        if n_gated:
            axh.bar([-3.10], [n_gated], width=0.10, color=GATED, zorder=3)
            axh.text(-3.10, n_gated * 1.6, f"{n_gated}", ha="center", va="bottom", fontsize=7.4,
                     color=INK, rotation=90)
            for dx in (-0.02, 0.02):
                axh.plot([-2.86 + dx, -2.78 + dx], [0.42, 1.1], color=INK, lw=0.9,
                         clip_on=False, zorder=5)
        axh.set_yscale("log")
        axh.set_ylim(0.55, 3.0e5)
        axh.set_xlim(-3.35, 0.95)
        axh.set_yticks([1, 10, 100, 1000])
        axh.set_yticklabels(["1", "10", "100", "1000"], fontsize=7)
        # Tick every attained value: the atoms are the object the panel is about.
        ticks = ([-3.10] if n_gated else []) + list(vals)
        names = ([r"off, $-10^9$"] if n_gated else []) + [f"{v:g}" for v in vals]
        axh.set_xticks(ticks)
        axh.set_xticklabels(names, fontsize=7.6, rotation=90, ha="center", va="top")
        axh.grid(axis="y", lw=0.4, color=FAINT, zorder=0)
        axh.set_xlabel("bias value (logits)", fontsize=8.5, color=INK)
        if j == 0:
            axh.set_ylabel("cells (log)", fontsize=8.5, color=INK)
        axh.set_title(f"{len(vals) + (1 if n_gated else 0)} distinct values",
                      fontsize=9.5, pad=6, color=INK)

    cb = fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.014, pad=0.015)
    cb.set_label("additive bias (logits), clipped to $[-2.6, 2.6]$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.legend(handles=[
        Patch(facecolor=GATED, label="gated: $-10^{9}$, off the colour scale (not dark blue)"),
        Patch(facecolor="#f7f6f6", edgecolor=HAIR, label="zero bias: unmodified attention logit"),
        Patch(facecolor=ORANGE, label="target span $v_0$ (tokens 4-9)"),
        Patch(facecolor=BLUE, label="hop-1 spans (10-27)"),
        Patch(facecolor=GREEN, label="hop-2 spans (28-45)"),
        Patch(facecolor=MUTED, label="exempt positions: instruction 0-3, suffix 46, [M] 47"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.055))

    note = ("Only hop buckets 0-4 are exercised: the graph's diameter is 4, so "
            r"$\lambda_5$ never appears and each graded panel attains 5 values, not 6."
            "   Zero bias is white in both graded panels whether the pair is at distance 0 or "
            "exempt from the bias entirely; the grey bands on the axes disambiguate.")
    fig.text(0.5, -0.085, note, ha="center", fontsize=8.5, color=MUTED, wrap=True)

    _save(fig, out, "graded_bias_reference")

    print("\ngraded_bias reference")
    for title, mat, _ in panels:
        m = mat.numpy()
        finite = np.unique(np.round(m[m > MASK_BIAS / 2], 4))
        print(f"  {title:32s} gated={int((m <= MASK_BIAS / 2).sum()):5d}  "
              f"finite values={finite}")


# --------------------------------------------------------------------------------------------
# Reference 2: reach against composed layers
# --------------------------------------------------------------------------------------------

def ref_layer_reach(out: Path) -> None:
    """Corrects the overlap that hides the figure's own point.

    star, ego_star and none all sit at exactly 100% from layer 2 onward, so three of the four
    series collapse into one line and the reader cannot see which family arrived when. Markers are
    offset horizontally here, every value is labelled, and the layer at which each family first
    saturates is called out on its own rather than by a single ambiguous arrow.
    """
    g = demo_graph()
    layers = np.arange(1, 5)
    series = [("star", ORANGE, "o", -0.075), ("ego_star", BLUE, "^", -0.025),
              ("none", "#7F7F7F", "d", 0.025), ("ego", GREEN, "s", 0.075)]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhline(100, ls=":", lw=0.9, color=MUTED, zorder=1)
    ax.text(1.18, 103.2, "every pair connected", fontsize=9, color=MUTED, va="bottom")

    rows = []
    for i, (fam, colour, marker, dx) in enumerate(series):
        r = [100 * v for v in reach_by_layer(g, fam, max_layers=len(layers))]
        first = int(np.argmax(np.isclose(r, 100.0))) + 1
        rows.append((fam, colour, marker, r, first))
        ax.plot(layers + dx, r, ls="--", lw=1.0, color=colour, alpha=0.55, zorder=2)
        ax.plot(layers + dx, r, ls="none", marker=marker, ms=9.0, color=colour, zorder=3)
        # An open ring on the first saturating point: which layer each family arrives at is the
        # figure's whole content, and it is invisible while four series sit on one line.
        ax.plot([first + dx], [100], ls="none", marker="o", ms=17.0, mfc="none", mec=colour,
                mew=1.4, zorder=4)
        for x, y in zip(layers + dx, r, strict=True):
            if y < 99.9:
                side = -0.055 if i % 2 == 0 else 0.055
                ax.text(x + side, y - 4.6, f"{y:.1f}", fontsize=8.5, color=colour,
                        ha="right" if i % 2 == 0 else "left")

    ax.set_xlabel("masked attention layers composed  (integer only; the dashed guides are not "
                  "interpolation)", fontsize=9.5)
    ax.set_ylabel("node pairs able to have exchanged information (%)", fontsize=10)
    ax.set_xticks(layers)
    ax.set_xlim(0.60, 6.75)
    ax.set_ylim(0, 110)
    ax.tick_params(labelsize=9)

    # Right margin: the numbers themselves, keyed by marker, replacing a legend that would
    # otherwise repeat the same information without the values.
    x0, y0, dy = 4.45, 96.0, 7.4
    ax.add_patch(Rectangle((4.30, y0 - 4.6 * dy), 2.32, 5.5 * dy, facecolor=FAINT,
                           edgecolor=HAIR, lw=0.6, zorder=1))
    ax.text(x0 + 0.30, y0, "L =      1     2     3     4   sat.", fontsize=9,
            family="monospace", va="center", color=INK, zorder=3)
    for k, (fam, colour, marker, r, first) in enumerate(rows):
        y = y0 - (k + 1) * dy
        ax.plot([x0 + 0.05], [y], marker=marker, ms=8.0, color=colour, clip_on=False, zorder=3)
        ax.text(x0 + 0.14, y, f"{fam:9s}" + " ".join(f"{v:5.1f}" for v in r) + f"  L={first}",
                fontsize=9, family="monospace", va="center", color=colour, zorder=3)
    ax.text(4.34, y0 - 5.6 * dy,
            "markers are offset horizontally so that\n"
            "coincident series stay visible; the rings\n"
            "mark first saturation. The node graph's\n"
            "diameter is 4, which is why ego, and only\n"
            "ego, needs four composed layers.",
            fontsize=9, va="top", color=MUTED)

    _save(fig, out, "layer_reach_reference")
    print("\nlayer_reach reference")
    for fam, _c, _m, r, first in rows:
        print(f"  {fam:9s} " + "  ".join(f"{v:5.1f}" for v in r) + f"   saturates at L={first}")


# --------------------------------------------------------------------------------------------
# Reference 3: measured sequence lengths and the cost of the fixed budget
# --------------------------------------------------------------------------------------------

def ref_seq_lengths(out: Path, stats_path: Path) -> None:
    """Corrects colliding annotations and an uninformative bar pair.

    p50 and p95 are drawn in #DE8F05 and #D55E00, which are hard to tell apart, and their labels
    are placed at 0.94 of the y-range where they overlap both the tallest bars and each other.
    Here the markers live on a clean strip above the axes. On the right, the "fixed 2048" bars are
    constant 1.0 by construction and carry no information; they become a reference line, which
    leaves room to state the fact that motivates the panel: nothing in the corpus is near 2048.
    """
    s = json.loads(stats_path.read_text())
    lengths = np.array(s["lengths"])
    p50, p95 = np.percentile(lengths, 50), np.percentile(lengths, 95)
    longest, n = int(lengths.max()), len(lengths)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                  gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.24})

    ax.hist(lengths, bins=44, color=BLUE, alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.42)
    strip = top * 1.06

    ax.add_patch(Rectangle((longest, 0), 2048 - longest, top * 1.42, facecolor=RED, alpha=0.07,
                           edgecolor="none", zorder=1))
    ax.text((longest + 2048) / 2, top * 0.55,
            f"{2048 - longest} tokens\nno example ever\nreaches", ha="center", fontsize=9,
            color=RED)

    # Two tiers: p95 and the longest example are only 199 tokens apart, so one tier collides.
    for v, lab, colour, ls, tier in [(p50, f"p50 = {p50:.0f}", ORANGE, "--", 0),
                                     (p95, f"p95 = {p95:.0f}", RED, "-.", 1),
                                     (longest, f"longest = {longest}", GREEN, ":", 0),
                                     (2048, "fixed $S = 2048$", INK, "-", 1)]:
        y = strip + tier * top * 0.115
        ax.axvline(v, color=colour, lw=1.3, ls=ls, zorder=4)
        ax.plot([v], [y], marker="v", ms=7, color=colour, clip_on=False, zorder=5)
        ax.text(v, y + top * 0.045, lab, ha="center", fontsize=9.5, color=colour)

    ax.set_xlabel("tokens per Cora node-classification example", fontsize=10)
    ax.set_ylabel("examples", fontsize=10)
    ax.set_xlim(0, 2160)
    ax.tick_params(labelsize=9)
    ax.set_title(f"Measured prompt lengths  ($n = {n}$, mean {lengths.mean():.0f}, "
                 f"{int((lengths > 2048).sum())} above 2048)", fontsize=11, pad=26)

    mult = 128
    dyn = np.ceil(lengths / mult) * mult
    ratios = [dyn.mean() / 2048, (dyn ** 2).mean() / 2048 ** 2]
    labels = ["linear\n(memory, MLP)", "quadratic\n(attention, bias)"]
    x = np.arange(2)
    ax2.axhline(1.0, color=INK, lw=1.2, zorder=2)
    ax2.text(1.48, 1.0, "fixed $S=2048$\n(reference)", fontsize=9, va="center", color=INK)
    ax2.bar(x, ratios, 0.42, color=GREEN, zorder=3)
    for xi, v in zip(x, ratios, strict=True):
        ax2.text(xi, v + 0.035, f"{v:.3f}$\\times$", ha="center", fontsize=11, color=INK)
        ax2.text(xi, v / 2, f"{(1 - v) * 100:.0f}%\nwasted", ha="center", va="center",
                 fontsize=9.5, color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_xlim(-0.6, 2.05)
    ax2.set_ylim(0, 1.20)
    ax2.tick_params(labelsize=9)
    ax2.set_ylabel("cost relative to the fixed budget", fontsize=10)
    ax2.set_title("What the fixed budget cost, against padding to a multiple of 128",
                  fontsize=11, pad=26)

    _save(fig, out, "seq_lengths_reference")
    print("\nseq_lengths reference")
    print(f"  n={n} min={lengths.min()} p50={p50:.0f} p95={p95:.0f} max={longest} "
          f"mean={lengths.mean():.1f} above2048={int((lengths > 2048).sum())}")
    print(f"  linear={ratios[0]:.4f}  quadratic={ratios[1]:.4f}")


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.jpg", dpi=300, pil_kwargs={"quality": 95})
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {out / stem}.jpg and .pdf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="submission/figures")
    p.add_argument("--seq-stats", default="experiments/seq_lengths.json")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_graded_bias(out)
    ref_layer_reach(out)
    ref_seq_lengths(out, Path(args.seq_stats))


if __name__ == "__main__":
    main()
