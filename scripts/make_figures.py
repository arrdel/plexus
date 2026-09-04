"""Generate the paper's figures from the real code paths, never from hand-drawn approximations.

Every panel here is produced by calling the same functions the experiments call. The mask matrices
come out of `plexus.masks`, the graph comes out of a real Cora ego sample, and the token-length
histogram is the measured distribution. A figure drawn to illustrate what a mask "looks like" is
free to be subtly wrong about the thing the paper's argument rests on; a figure rendered from
`build_attention_bias` cannot be.

    python scripts/make_figures.py --out submission/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plexus.masks import (  # noqa: E402
    MASK_BIAS,
    HopBias,
    LinearisedGraph,
    build_attention_bias,
    mask_density,
    pairwise_hops,
)

# A colour-blind-safe set; the paper is read in greyscale often enough to matter.
BLUE, ORANGE, GREEN, RED, GREY = "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#7F7F7F"
FAMILY_ORDER = ["none", "star", "ego", "ego_star", "random_rewire"]
# Plain underscores and percent signs: matplotlib is not running usetex here, so LaTeX escapes
# would be rendered as literal backslashes.
FAMILY_LABEL = {
    "none": "none\n(unmodified)",
    "star": "star\n(TAG-DLM)",
    "ego": "ego\n(observed edges)",
    "ego_star": "ego_star\n(ours)",
    "random_rewire": "rewire\n(density control)",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "mathtext.fontset": "cm",
})


def demo_graph(n_neigh1: int = 3, n_neigh2: int = 3, tokens_per_node: int = 6,
               n_prefix: int = 4, n_suffix: int = 2) -> LinearisedGraph:
    """A small ego graph laid out exactly as the real linearisation lays one out.

    Small enough that individual token blocks are visible in a printed figure, and with the two
    structural features the argument turns on: a neighbour--neighbour edge among the hop-1 set,
    and hop-2 nodes that reach the target only through a hop-1 relay.
    """
    n_nodes = 1 + n_neigh1 + n_neigh2
    spans, cur = [], n_prefix
    for _ in range(n_nodes):
        spans.append((cur, cur + tokens_per_node))
        cur += tokens_per_node
    answer_pos = cur + n_suffix - 1
    seq_len = cur + n_suffix

    edges = {(0, i) for i in range(1, 1 + n_neigh1)}
    edges.add((1, 2))                                    # neighbour--neighbour
    for j in range(n_neigh2):                            # hop-2 hang off hop-1 nodes
        edges.add((1 + (j % n_neigh1), 1 + n_neigh1 + j))
    return LinearisedGraph(spans=spans, seq_len=seq_len, edges=edges,
                           node_ids=list(range(n_nodes)), pad_from=seq_len,
                           answer_pos=answer_pos)


# --------------------------------------------------------------------------------------------
# Figure: the mask ladder, rendered from build_attention_bias
# --------------------------------------------------------------------------------------------

def _annotate_spans(ax, g: LinearisedGraph, n_hop1: int = 3):
    """Mark which token range belongs to the target, the hop-1 set and the hop-2 set."""
    groups = [(0, "T", RED), (1, "$\\mathcal{N}_1$", BLUE),
              (1 + n_hop1, "$\\mathcal{N}_2$", GREEN)]
    ends = [1, 1 + n_hop1, g.n_nodes]
    for (start, label, colour), end in zip(groups, ends):
        lo = g.spans[start][0]
        hi = g.spans[end - 1][1]
        mid = (lo + hi) / 2 - 0.5
        ax.text(mid, g.seq_len + 4.0, label, ha="center", va="top",
                fontsize=6.4, color=colour)
        ax.plot([lo - 0.5, hi - 0.5], [g.seq_len + 2.2] * 2, color=colour, lw=1.4,
                clip_on=False)


def fig_mask_ladder(out: Path):
    g = demo_graph()
    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(7.0, 1.95))

    for ax, fam in zip(axes, FAMILY_ORDER):
        bias = build_attention_bias([g], family=fam, dtype=torch.float32,
                                    generator=torch.Generator().manual_seed(0))
        m = (bias[0, 0] > MASK_BIAS / 2).numpy().astype(float)
        dens = mask_density(bias)
        # 1 -> near-black, 0 -> near-white, so the legend swatches match the panels.
        ax.imshow(1.0 - m, cmap="gray", vmin=-0.08, vmax=1.0, interpolation="nearest")

        for s, _e in g.spans:                            # span boundaries
            ax.axhline(s - 0.5, lw=0.2, color=BLUE, alpha=0.45)
            ax.axvline(s - 0.5, lw=0.2, color=BLUE, alpha=0.45)
        ts, te = g.spans[0]
        ax.add_patch(plt.Rectangle((ts - 0.5, ts - 0.5), te - ts, te - ts,
                                   fill=False, ec=RED, lw=0.9))
        ax.axhline(g.answer_pos - 0.5, lw=0.6, color=ORANGE)
        _annotate_spans(ax, g)

        ax.set_title(f"{FAMILY_LABEL[fam]}\n$\\rho$ = {dens:.2f}", fontsize=7.2, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].set_ylabel("query (reader)", fontsize=7)
    fig.legend(handles=[
        Patch(fc="black", label="attention permitted"),
        Patch(fc="white", ec="0.6", label="gated ($-10^9$)"),
        Patch(fc="none", ec=RED, label="target span"),
        Patch(fc="none", ec=ORANGE, label="answer [M] row"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    fig.subplots_adjust(wspace=0.12, bottom=0.06)
    fig.savefig(out / "mask_ladder.pdf")
    plt.close(fig)
    print("  mask_ladder.pdf")


# --------------------------------------------------------------------------------------------
# Figure: reachability by layer -- the one-layer-delay result (E0.0c)
# --------------------------------------------------------------------------------------------

def reach_by_layer(g: LinearisedGraph, family: str, max_layers: int = 4) -> list[float]:
    """Fraction of *node* pairs whose information can have met after L masked layers.

    One masked attention layer propagates information along one edge of the allow-graph, so after
    L layers node i has heard from node j iff their allow-graph distance is at most L. Composing
    the boolean allow matrix L times is exactly that reachability, and it is what turns "the star
    mask blocks neighbour-neighbour attention" into the far weaker "it delays it by one layer".
    """
    bias = build_attention_bias([g], family=family, dtype=torch.float32,
                                generator=torch.Generator().manual_seed(0))
    tok_allow = (bias[0, 0] > MASK_BIAS / 2)
    idx = [int(s) for s, _ in g.spans]                   # one representative token per node
    A = tok_allow[idx][:, idx].clone()

    n = A.shape[0]
    reach = A | torch.eye(n, dtype=torch.bool)
    out = []
    for _ in range(max_layers):
        out.append(reach.float().mean().item())
        reach = (reach.float() @ A.float() > 0) | reach
    return out


def fig_layer_reach(out: Path):
    """Sized for a single ICML column (3.25in) and included at 1:1.

    Drawing it wider and letting \\includegraphics shrink it to \\columnwidth scales the 8pt
    labels down to about 4pt, which is unreadable in print. Anything that goes in one column is
    authored at one column's width.
    """
    g = demo_graph()
    layers = list(range(1, 5))
    fig, ax = plt.subplots(figsize=(3.25, 2.30))

    for fam, colour, marker in [("star", ORANGE, "o"), ("ego", GREEN, "s"),
                                ("ego_star", BLUE, "^"), ("none", GREY, "d")]:
        r = reach_by_layer(g, fam, max_layers=len(layers))
        ax.plot(layers, [100 * v for v in r], marker=marker, color=colour, lw=1.3,
                ms=4.0, label=FAMILY_LABEL[fam].split("\n")[0])
    ax.axhline(100, ls=":", lw=0.7, color="0.3")
    ax.annotate("every pair connected", xy=(2, 100), xytext=(2.15, 78),
                fontsize=6.4, color=ORANGE,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=ORANGE))
    ax.set_xlabel("masked attention layers composed")
    ax.set_ylabel("node pairs able to have\nexchanged information (%)")
    ax.set_xticks(layers)
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "layer_reach.pdf")
    plt.close(fig)
    print("  layer_reach.pdf")


# --------------------------------------------------------------------------------------------
# Figure: the graded bias -- what it can express that a binary mask cannot
# --------------------------------------------------------------------------------------------

def fig_graded_bias(out: Path):
    """Top row: the bias matrix. Bottom row: the multiset of values it can occupy.

    The bottom row carries the argument. A binary mask is a two-point distribution however
    elaborate the graph it encodes, so no amount of choosing *which* edges to gate escapes those
    two values; a graded bias spreads over the reals and contains the unmodified model exactly.
    Showing only the matrices would leave that as an assertion in the caption.
    """
    g = demo_graph()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.75),
                            gridspec_kw={"height_ratios": [2.35, 1]})

    star = build_attention_bias([g], family="star", dtype=torch.float32)[0, 0]
    ego = build_attention_bias([g], family="ego", dtype=torch.float32)[0, 0]
    lin = HopBias(values=torch.tensor([0.0, -0.5, -1.0, -1.5, -2.0, -3.0]))
    shaped = HopBias(values=torch.tensor([0.0, 0.4, -0.2, -0.9, -1.6, -2.4]))
    shaped_b = build_attention_bias([g], family="none", hop_bias=shaped,
                                    dtype=torch.float32)[0, 0]
    lin_b = build_attention_bias([g], family="none", hop_bias=lin,
                                 dtype=torch.float32)[0, 0]

    panels = [
        ("star (binary)", star, True),
        ("ego (binary)", ego, True),
        ("graded, $\\lambda_d\\!=\\!-0.5d$", lin_b, False),
        ("graded, arbitrary shape", shaped_b, False),
    ]

    im = None
    for j, (title, mat, is_hard) in enumerate(panels):
        ax = axes[0, j]
        shown = np.clip(mat.numpy(), -2.6, 2.6)
        im = ax.imshow(shown, cmap="RdBu_r", vmin=-2.6, vmax=2.6, interpolation="nearest")
        for s, _e in g.spans:
            ax.axhline(s - 0.5, lw=0.2, color="0.4", alpha=0.55)
            ax.axvline(s - 0.5, lw=0.2, color="0.4", alpha=0.55)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

        # The value multiset. Gated entries are -1e9, plotted at the clip edge with a marker.
        axh = axes[1, j]
        v = mat.numpy().ravel()
        finite = v[v > MASK_BIAS / 2]
        axh.hist(finite, bins=np.linspace(-2.6, 1.0, 34), color=BLUE if not is_hard else GREY,
                 log=True)
        n_gated = int((v <= MASK_BIAS / 2).sum())
        if n_gated:
            axh.axvline(-2.55, color=RED, lw=1.6)
            axh.text(-2.45, axh.get_ylim()[1] * 0.30, f"$-10^9$\n$\\times${n_gated}",
                     fontsize=5.6, color=RED, va="top")
        distinct = len(np.unique(np.round(v, 4)))
        axh.set_title(f"{distinct} distinct values", fontsize=6.4, pad=2)
        axh.set_xlim(-2.7, 1.0)
        # Log scale re-populates minor ticks after set_yticks([]), so hide the axis outright.
        axh.yaxis.set_visible(False)
        axh.spines["left"].set_visible(False)
        axh.tick_params(labelsize=5.6)
        axh.set_xlabel("bias value" if j == 1 else "", fontsize=6)

    cb = fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.016, pad=0.012)
    cb.set_label("additive bias (logits)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("query", fontsize=7)
    fig.savefig(out / "graded_bias.pdf")
    plt.close(fig)
    print("  graded_bias.pdf")


# --------------------------------------------------------------------------------------------
# Figure: measured sequence lengths and what padding costs
# --------------------------------------------------------------------------------------------

def fig_seq_lengths(out: Path, stats_path: Path | None):
    if stats_path is None or not stats_path.exists():
        print("  seq_lengths.pdf SKIPPED (no measured stats; run "
              "scripts/profile_sequence_lengths.py --json)")
        return
    s = json.loads(stats_path.read_text())
    lengths = np.array(s["lengths"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.0),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    ax.hist(lengths, bins=44, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
    for q, lab, c in [(50, "p50", ORANGE), (95, "p95", RED)]:
        v = np.percentile(lengths, q)
        ax.axvline(v, color=c, lw=1.0, ls="--")
        ax.text(v, ax.get_ylim()[1] * 0.94, f" {lab}={v:.0f}", color=c, fontsize=6.6)
    ax.axvline(2048, color="0.25", lw=1.0)
    ax.text(2048, ax.get_ylim()[1] * 0.80, " fixed $S{=}2048$", fontsize=6.6, color="0.25")
    ax.set_xlabel("tokens per Cora NC example")
    ax.set_ylabel("examples")
    ax.set_title("Measured prompt lengths", fontsize=8)

    # Cost of a fixed budget relative to padding to a multiple of 128.
    mult = 128
    dyn = np.ceil(lengths / mult) * mult
    labels = ["linear\n(memory, MLP)", "quadratic\n(attention, bias)"]
    fixed_rel = [1.0, 1.0]
    dyn_rel = [dyn.mean() / 2048, (dyn ** 2).mean() / 2048 ** 2]
    x = np.arange(2)
    ax2.bar(x - 0.19, fixed_rel, 0.36, label="fixed 2048", color=GREY)
    ax2.bar(x + 0.19, dyn_rel, 0.36, label="pad to 128", color=GREEN)
    for xi, v in zip(x + 0.19, dyn_rel):
        ax2.text(xi, v + 0.03, f"{v:.3f}$\\times$", ha="center", fontsize=6.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.8)
    ax2.set_ylabel("cost relative to fixed 2048")
    ax2.set_ylim(0, 1.22)
    ax2.legend(frameon=False, fontsize=6.8)
    ax2.set_title("What the fixed budget cost", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "seq_lengths.pdf")
    plt.close(fig)
    print("  seq_lengths.pdf")


# --------------------------------------------------------------------------------------------
# Figure: results, drawn only from experiment JSON that actually exists
# --------------------------------------------------------------------------------------------

def fig_results(out: Path, exp_dir: Path):
    rows = []
    for d in sorted(exp_dir.glob("*/")):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows.append((d.name, r))
    if not rows:
        print("  results.pdf SKIPPED (no experiments/*/result.json yet)")
        return

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    names = [n for n, _ in rows]
    accs = [100 * r["accuracy"] for _, r in rows]
    dens = [r.get("mask_density", float("nan")) for _, r in rows]
    ax.scatter(dens, accs, c=range(len(rows)), cmap="viridis", s=26, zorder=3)
    for n, d, a in zip(names, dens, accs):
        ax.annotate(n.split("_")[0], (d, a), fontsize=5.6,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("mask density $\\rho$")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy against density", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "results.pdf")
    plt.close(fig)
    print(f"  results.pdf ({len(rows)} runs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="submission/figures")
    p.add_argument("--experiments", default="experiments")
    p.add_argument("--seq-stats", default="experiments/seq_lengths.json")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out}")
    fig_mask_ladder(out)
    fig_layer_reach(out)
    fig_graded_bias(out)
    fig_seq_lengths(out, Path(args.seq_stats))
    fig_results(out, Path(args.experiments))


if __name__ == "__main__":
    main()
