"""Figure: the collapse this paper is about, as two numbers and an arrow.

Deliberately not a bar chart. Two bars 30 points apart at this size say "one is shorter"; two
80pt numerals say 91.62 and 61.76, which is the claim. The gap is computed from the two values
rather than typed, so the annotation cannot drift away from the numerals above it.

Both figures are the numbers TAG-DLM *reports*, not numbers we measured -- hence the source rule
along the bottom. Our own reproduction is a separate figure.

    python scripts/make_collapse_figure.py --out submission/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

INK, MUTED, HAIR = "#2E3338", "#9AA0A6", "#D7DBE0"
WARN = "#D55E00"          # the paper's plxred, reserved for the collapse

STAR_LP = 91.62           # TAG-DLM, Cora link prediction, star mask
EGO_LP = 61.76            # same table, ego mask

plt.rcParams.update({
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "font.family": "serif", "mathtext.fontset": "cm",
})


def _extent(fig, ax, artist):
    """Bounding box of a text artist in data units."""
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transData.inverted())


def fig_collapse(out: Path, stem: str = "ego_collapse", png: bool = False,
                 numeral_pt: float = 80.0):
    # Units are square, but units-per-inch depends on the axes box rather than figsize, so the
    # numeral geometry below is measured from the render instead of derived from `numeral_pt`.
    W, H = 240.0, 74.0
    fig, ax = plt.subplots(figsize=(W / 20.0, H / 20.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    baseline, margin = 44.0, 8.0

    # Anchored to the outer margins rather than to guessed centres: at 80pt each numeral is
    # nearly four inches wide, so a centre computed from an estimated glyph width puts the
    # arrow underneath the digits.
    left = ax.text(margin, baseline, f"{STAR_LP:.2f}", fontsize=numeral_pt, color=INK,
                   ha="left", va="baseline")
    right = ax.text(W - margin, baseline, f"{EGO_LP:.2f}", fontsize=numeral_pt, color=WARN,
                    ha="right", va="baseline")

    lb, rb = _extent(fig, ax, left), _extent(fig, ax, right)
    gap = rb.x0 - lb.x1
    if gap < 24.0:
        raise ValueError(f"numerals leave only {gap:.1f} units for the arrow; widen the "
                         f"figure or lower --numeral-pt")

    mid_y = (lb.y0 + lb.y1) / 2.0
    pad = gap * 0.18
    ax.add_patch(FancyArrowPatch((lb.x1 + pad, mid_y), (rb.x0 - pad, mid_y),
                                 arrowstyle="-|>,head_width=0.44,head_length=0.60",
                                 mutation_scale=34, ec=WARN, fc=WARN, lw=5.0,
                                 shrinkA=0, shrinkB=0))
    ax.text((lb.x1 + rb.x0) / 2.0, mid_y + 4.0, f"$-${STAR_LP - EGO_LP:.2f} points",
            fontsize=14, color=WARN, ha="center", va="bottom")

    ax.text(lb.x0 + (lb.x1 - lb.x0) / 2.0, baseline - 8.0, "star mask (abstraction)",
            fontsize=12, color=MUTED, ha="center", va="top")
    ax.text(rb.x0 + (rb.x1 - rb.x0) / 2.0, baseline - 8.0, "ego mask (real topology)",
            fontsize=12, color=MUTED, ha="center", va="top")

    ax.plot([margin, W - margin], [18, 18], color=HAIR, lw=0.9)
    ax.text(margin, 13.0, "reported by TAG-DLM (2026), Cora link prediction. "
                          "No code released.", fontsize=9, color=MUTED, ha="left", va="top")

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
    p.add_argument("--numeral-pt", type=float, default=80.0)
    a = p.parse_args()
    fig_collapse(a.out, png=a.png, numeral_pt=a.numeral_pt)


if __name__ == "__main__":
    main()
