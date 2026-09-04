"""Compare two runs example by example, not by their accuracies.

Two mask families reaching the same accuracy is ambiguous in a way that matters here. They may be
deciding the same examples the same way, in which case the mask changed nothing that reached the
prediction; or they may be trading errors, in which case the mask changed a great deal and the
accuracies coincided. The difference is invisible in a results table and decisive for the claim
that a topology mask contributes nothing on this task.

McNemar's test is the right instrument: it conditions on the discordant pairs, which is exactly
the population that a "the mask does nothing" claim says should be empty or symmetric.

    python scripts/compare_decisions.py experiments/titan3/E0.1_star_nc_s0 \
                                        experiments/titan3/E0.2_none_nc_s0
    python scripts/compare_decisions.py --all experiments/titan3
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def load(d: Path) -> dict:
    r = json.loads((d / "result.json").read_text())
    if not r.get("predictions"):
        raise SystemExit(
            f"{d} has no per-example predictions. Runs started before predictions were stored "
            f"cannot be compared this way; rerun, or compare only newer runs."
        )
    return r


def mcnemar(a: dict, b: dict) -> dict:
    """Exact-ish McNemar on the discordant pairs of two runs over the same examples."""
    pa, pb, y = a["predictions"], b["predictions"], a["labels"]
    if len(pa) != len(pb) or a["labels"] != b["labels"]:
        raise SystemExit(
            "these two runs did not score the same examples in the same order; a decision-level "
            "comparison is only meaningful within one task and seed"
        )
    ca = [p == t for p, t in zip(pa, y, strict=True)]
    cb = [p == t for p, t in zip(pb, y, strict=True)]
    n01 = sum(1 for x, z in zip(ca, cb, strict=True) if not x and z)   # b right, a wrong
    n10 = sum(1 for x, z in zip(ca, cb, strict=True) if x and not z)   # a right, b wrong
    agree = sum(1 for x, z in zip(pa, pb, strict=True) if x == z)
    n = n01 + n10
    # Continuity-corrected chi-square, one degree of freedom. With n small this is only
    # indicative, which is why the discordant counts are printed beside it.
    chi2 = (abs(n01 - n10) - 1) ** 2 / n if n > 0 else 0.0
    # Survival function of chi-square with 1 dof, via the error function.
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    return {
        "n": len(pa),
        "identical_decisions": agree,
        "decision_agreement": agree / len(pa),
        "a_right_b_wrong": n10,
        "b_right_a_wrong": n01,
        "discordant": n,
        "chi2": chi2,
        "p_value": p,
    }


def describe(name_a: str, name_b: str, a: dict, b: dict) -> str:
    m = mcnemar(a, b)
    verdict = (
        "indistinguishable at the decision level"
        if m["p_value"] > 0.05
        else "separable: the discordant pairs are asymmetric"
    )
    return (
        f"{name_a}  vs  {name_b}\n"
        f"  accuracy            {100 * a['accuracy']:.2f}  vs  {100 * b['accuracy']:.2f}\n"
        f"  same decision on    {m['identical_decisions']}/{m['n']} "
        f"({100 * m['decision_agreement']:.1f}%)\n"
        f"  discordant          {m['a_right_b_wrong']} / {m['b_right_a_wrong']} "
        f"(first right, second right)\n"
        f"  McNemar chi2={m['chi2']:.3f}  p={m['p_value']:.4f}  -> {verdict}\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="*", type=Path)
    p.add_argument("--all", type=Path, default=None,
                   help="compare every pair of runs sharing a task and seed under this directory")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    out = {}
    if args.all:
        runs = {}
        for f in sorted(args.all.rglob("result.json")):
            r = json.loads(f.read_text())
            if r.get("predictions"):
                runs[f.parent.name] = r
        if not runs:
            raise SystemExit(f"no runs with stored predictions under {args.all}")
        for (na, a), (nb, b) in itertools.combinations(sorted(runs.items()), 2):
            if (a["task"], a["seed"], len(a["predictions"])) != \
               (b["task"], b["seed"], len(b["predictions"])):
                continue
            print(describe(na, nb, a, b))
            out[f"{na}|{nb}"] = mcnemar(a, b)
    else:
        if len(args.dirs) != 2:
            raise SystemExit("give two run directories, or --all <dir>")
        a, b = (load(d) for d in args.dirs)
        print(describe(args.dirs[0].name, args.dirs[1].name, a, b))
        out[f"{args.dirs[0].name}|{args.dirs[1].name}"] = mcnemar(a, b)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
