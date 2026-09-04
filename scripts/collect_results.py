"""Turn experiments/*/result.json into the LaTeX tables the paper includes.

Hand-copying numbers into a paper is how a table ends up disagreeing with the run that produced
it. This emits `submission/results_generated.tex`, which main.tex and the appendix \\input, so a
number in the PDF can only ever come from a file on disk.

Cells with no run behind them are emitted as \\pending, which typesets as a visible marker rather
than a blank or, worse, a plausible-looking number. Nothing here invents a value: if the queue has
not produced a run, the table says so in the PDF.

    python scripts/collect_results.py --experiments experiments --out submission/results_generated.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# The reported TAG-DLM numbers we are trying to reproduce. Transcribed from the paper; we have no
# code, so every one of these is a target, not a verified quantity.
TAGDLM = {
    ("cora", "nc"): 90.96,
    ("cora", "lp"): 91.62,
    ("pubmed", "nc"): 96.30,
    ("pubmed", "lp"): 95.31,
}
# TAG-DLM's own ego-mask ablation, the collapse this paper is about.
TAGDLM_EGO = {("cora", "lp"): 61.76}

FAMILY_TEX = {
    "none": r"\texttt{none}",
    "star": r"\texttt{star}",
    "ego": r"\texttt{ego}",
    "ego_star": r"\texttt{ego\_star}",
    "random_rewire": r"\texttt{random\_rewire}",
}
FAMILY_NOTE = {
    "none": "no structure (unmodified model)",
    "star": "TAG-DLM's default",
    "ego": "observed ego-graph edges",
    "ego_star": "ego edges $\\cup$ target reach (ours)",
    "random_rewire": "density-matched control",
}
ORDER = ["none", "star", "ego", "ego_star", "random_rewire"]


def load_runs(exp_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(exp_dir.iterdir()):
        f = d / "result.json"
        if not (d.is_dir() and f.exists()):
            continue
        r = json.loads(f.read_text())
        h = {}
        hf = d / "history.json"
        if hf.exists():
            try:
                h = json.loads(hf.read_text())
            except json.JSONDecodeError:
                h = {}
        cfg = h.get("config", {})
        r["_dir"] = d.name
        r["_epochs"] = cfg.get("epochs")
        r["_hop_buckets"] = cfg.get("hop_bias_buckets")
        r["_answer_as_target"] = cfg.get("hop_bias_answer_as_target", False)
        r["_dataset"] = cfg.get("dataset", "cora")
        r["_smoke"] = d.name.startswith("smoke")
        runs.append(r)
    return runs


def agg(runs: list[dict]) -> tuple[float, float, int] | None:
    """Mean, sample std and n over seeds. Returns None when there is nothing to report."""
    if not runs:
        return None
    accs = [100 * r["accuracy"] for r in runs]
    m = sum(accs) / len(accs)
    if len(accs) < 2:
        return m, float("nan"), 1
    var = sum((a - m) ** 2 for a in accs) / (len(accs) - 1)
    return m, math.sqrt(var), len(accs)


def cell(a: tuple[float, float, int] | None) -> str:
    if a is None:
        return r"\pending"
    m, s, n = a
    if n < 2 or math.isnan(s):
        # A single seed is not a result under the standing rule; mark it as provisional in place
        # rather than letting it read like a mean over seeds.
        return f"{m:.2f}\\rlap{{\\,\\textsuperscript{{$\\dagger$}}}}"
    return f"{m:.2f} {{\\scriptsize$\\pm$ {s:.2f}}}"


def density_cell(runs: list[dict], measured: dict | None = None, task: str = "nc",
                 family: str = "star") -> str:
    """Prefer the run's own density; fall back to the standalone measurement.

    Density is a property of the mask and the graph, not of a trained model, so there is no
    reason for this column to be empty while the accuracy column waits on the queue.
    scripts/measure_densities.py produces it without a GPU.
    """
    ds = [r.get("mask_density") for r in runs if r.get("mask_density") is not None]
    if ds:
        return f"{sum(ds) / len(ds):.3f}"
    if measured:
        rec = measured.get("tasks", {}).get(task, {}).get(family)
        if rec:
            return f"{rec['mean']:.3f}"
    return r"\pending"


def select(runs, *, task, family, dataset="cora", hop_buckets=None, answer_as_target=False):
    out = []
    for r in runs:
        if r["_smoke"] or r.get("task") != task or r["_dataset"] != dataset:
            continue
        if r.get("mask_family") != family:
            continue
        if (r["_hop_buckets"] or None) != hop_buckets:
            continue
        if bool(r["_answer_as_target"]) != bool(answer_as_target):
            continue
        out.append(r)
    return out


def table_ladder(runs: list[dict], task: str, dataset: str = "cora",
                 measured: dict | None = None) -> str:
    """The mask ladder: the diagnostic table, claims C1 and C2."""
    rows = []
    for fam in ORDER:
        sel = select(runs, task=task, family=fam, dataset=dataset)
        rows.append(
            f"  {FAMILY_TEX[fam]} & {FAMILY_NOTE[fam]} & "
            f"{density_cell(sel, measured, task, fam)} & {cell(agg(sel))}"
            f" & {len(sel) if sel else 0} \\\\"
        )
    graded = select(runs, task=task, family="none", dataset=dataset, hop_buckets=6)
    graded_a = select(runs, task=task, family="none", dataset=dataset, hop_buckets=6,
                      answer_as_target=True)
    rows.append(r"  \midrule")
    rows.append(
        r"  \textsc{HopBias} & graded, learned $\lambda$ (ours) & "
        f"{density_cell(graded, measured, task, 'hop_bias')} & {cell(agg(graded))}"
        f" & {len(graded)} \\\\"
    )
    rows.append(
        r"  \quad + answer grading & also grades the \texttt{[M]} row (ours) & "
        f"{density_cell(graded_a, measured, task, 'hop_bias')} & {cell(agg(graded_a))}"
        f" & {len(graded_a)} \\\\"
    )

    ref = TAGDLM.get((dataset, task))
    ego_ref = TAGDLM_EGO.get((dataset, task))
    ref_rows = [r"  \midrule",
                f"  \\multicolumn{{5}}{{l}}{{\\emph{{reported by \\citet{{tagdlm2026}}, "
                f"not reproduced here}}}} \\\\",
                f"  \\texttt{{star}} & their number & --- & {ref:.2f} & --- \\\\"
                if ref else ""]
    if ego_ref:
        ref_rows.append(f"  \\texttt{{ego}} & their ablation & --- & {ego_ref:.2f} & --- \\\\")
    rows.extend([r for r in ref_rows if r])
    return "\n".join(rows)


def table_density(measured: dict | None) -> str:
    """The standalone density measurement, which needs no training run."""
    if not measured:
        return r"  \multicolumn{5}{l}{\pending{} --- run \texttt{scripts/measure\_densities.py}} \\"
    rows = []
    for fam in ORDER + ["hop_bias"]:
        label = (r"\textsc{HopBias} (graded)" if fam == "hop_bias" else FAMILY_TEX[fam])
        cells = []
        for task in ("nc", "lp"):
            rec = measured.get("tasks", {}).get(task, {}).get(fam)
            cells.append(f"{rec['mean']:.3f} {{\\scriptsize$\\pm$ {rec['std']:.3f}}}"
                         if rec else r"\pending")
        rows.append(f"  {label} & {cells[0]} & {cells[1]} \\\\")
    return "\n".join(rows)


# Cora node-classification training examples, i.e. one epoch. Measured by
# scripts/profile_sequence_lengths.py, not the 2708 nodes of the whole graph.
CORA_NC_TRAIN = 1624

THROUGHPUT_ROWS = [
    ("median step (ms)", "median_step_s", lambda v: f"{1000 * v:.0f}"),
    ("median padded length", "padded_len_mean", lambda v: f"{v:.0f}"),
    ("throughput (tok/s)", "tokens_per_s", lambda v: f"{v:.0f}"),
    ("effective (TFLOP/s)", "tflops", lambda v: f"{v:.1f}"),
    ("MFU (\\%)", "mfu_percent", lambda v: f"{v:.1f}"),
    ("peak memory (GB)", "peak_gb", lambda v: f"{v:.2f}"),
]


def table_throughput(exp_dir: Path) -> tuple[str, str]:
    """The two benchmark columns, and the speedup macro comparing them.

    The paired benchmark is the whole point: a throughput number alone says nothing about whether
    dynamic padding was worth implementing, and the two runs differ in exactly one setting.
    """
    cols = []
    for name in ("bench_star.json", "bench_fixed2048.json"):
        f = exp_dir / name
        cols.append(json.loads(f.read_text()) if f.exists() else None)

    rows = []
    for label, key, fmt in THROUGHPUT_ROWS:
        cells = [fmt(c[key]) if c and c.get(key) is not None else r"\pending" for c in cols]
        rows.append(f"  {label} & {cells[0]} & {cells[1]} \\\\")

    if all(c and c.get("tokens_per_s") for c in cols):
        speedup = cols[0]["tokens_per_s"] / cols[1]["tokens_per_s"]
        # Wall clock is the quantity a reader plans a GPU booking against, and it moves by more
        # than throughput does because the shorter sequences also carry fewer tokens per epoch.
        hours = [
            20 * CORA_NC_TRAIN * c["padded_len_mean"] / c["tokens_per_s"] / 3600 for c in cols
        ]
        rows.append(r"  \midrule")
        rows.append(f"  Cora NC, $20$ epochs (h) & {hours[0]:.2f} & {hours[1]:.2f} \\\\")
        macro = (f"\\newcommand{{\\padspeedup}}{{{speedup:.2f}}}\n"
                 f"\\newcommand{{\\padwallclockspeedup}}{{{hours[1] / hours[0]:.2f}}}\n"
                 f"\\newcommand{{\\benchmfu}}{{{cols[0]['mfu_percent']:.1f}}}\n"
                 f"\\newcommand{{\\benchtoks}}{{{cols[0]['tokens_per_s']:.0f}}}\n"
                 f"\\newcommand{{\\benchhours}}{{{hours[0]:.1f}}}")
    else:
        macro = "\n".join(
            f"\\newcommand{{\\{n}}}{{\\pending}}"
            for n in ("padspeedup", "padwallclockspeedup", "benchmfu", "benchtoks", "benchhours")
        )
    return "\n".join(rows), macro


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", default="experiments")
    p.add_argument("--densities", default="experiments/densities.json")
    p.add_argument("--out", default="submission/results_generated.tex")
    args = p.parse_args()

    dpath = Path(args.densities)
    measured = json.loads(dpath.read_text()) if dpath.exists() else None
    if measured:
        print(f"using measured densities from {dpath} "
              f"({measured.get('limit')} examples per task)")
    else:
        print(f"no {dpath}; density column will be PENDING")

    exp_dir = Path(args.experiments)
    thr_body, thr_macros = table_throughput(exp_dir)
    runs = load_runs(exp_dir)
    real = [r for r in runs if not r["_smoke"]]
    print(f"{len(runs)} run directories, {len(real)} non-smoke")
    for r in runs:
        tag = " (smoke)" if r["_smoke"] else ""
        print(f"  {r['_dir']:34s} {r.get('task')} {r.get('mask_family'):14s} "
              f"acc={100 * r['accuracy']:6.2f} n={r['n']:4d} "
              f"rho={r.get('mask_density', float('nan')):.3f}{tag}")

    body = [
        "% Generated by scripts/collect_results.py -- do not edit by hand.",
        f"% {len(real)} non-smoke runs were on disk when this was written.",
        "",
        r"\newcommand{\pending}{\textcolor{plxred}{\textsc{pending}}}",
        r"\newcommand{\nrunsnc}{" + str(len(real)) + "}",
        "",
        r"\newcommand{\ladderNCbody}{%",
        table_ladder(runs, "nc", "cora", measured),
        "}",
        "",
        r"\newcommand{\ladderLPbody}{%",
        table_ladder(runs, "lp", "cora", measured),
        "}",
        "",
        r"\newcommand{\densitybody}{%",
        table_density(measured),
        "}",
        "",
        r"\newcommand{\densityn}{"
        + (str(measured.get("limit")) if measured else "0") + "}",
        "",
        r"\newcommand{\throughputbody}{%",
        thr_body,
        "}",
        "",
        thr_macros,
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n")
    print(f"\nwrote {out}")
    if not real:
        print("NOTE: every accuracy cell will typeset as PENDING. That is intended -- no "
              "non-smoke run has produced a number yet.")


if __name__ == "__main__":
    main()
