"""Measure the token length of real linearised examples, to set `seq_len` from evidence.

RESEARCH_PLAN.md §6 budgets S=2048 because that is the largest length that fits in 24 GB, not
because Cora needs it. Those are different questions. The attention bias is quadratic in S and,
being a dense float tensor, it disables flash SDPA so the score matrix is materialised per layer
too -- so an S chosen well above the actual requirement is paid for on every step of every run.

    python scripts/profile_sequence_lengths.py --task nc --dataset cora

Reports the distribution and the smallest multiple of 128 that covers a given quantile, so the
choice can be made against a stated truncation rate rather than a guess.
"""

from __future__ import annotations

import argparse
import os

from plexus.data.examples import build_lp_example_set, build_nc_examples
from plexus.data.tags import load_tag
from plexus.models.llada import MASK_TOKEN_ID, load_tokenizer

DEFAULT_DATA_ROOT = os.environ.get("PLEXUS_DATA_ROOT", "/tmp/adelechinda/plexus_data")


def quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="nc", choices=["nc", "lp"])
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-hops", type=int, default=2)
    p.add_argument("--max-neighbours", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    # Deliberately generous so nothing is truncated while measuring what is actually needed.
    p.add_argument("--probe-seq-len", type=int, default=8192)
    args = p.parse_args()

    tag = load_tag(args.dataset, args.data_root, seed=args.seed)
    print(tag.summary(), flush=True)
    tokenizer = load_tokenizer()

    kw = dict(k=args.k_hops, K=args.max_neighbours, seed=args.seed,
              seq_len=args.probe_seq_len, mask_token_id=MASK_TOKEN_ID)
    print(f"building {args.task} examples at probe seq_len {args.probe_seq_len} ...", flush=True)

    if args.task == "nc":
        centers = tag.train_idx.tolist()[: args.limit]
        examples = build_nc_examples(tag, tokenizer, centers, **kw)
    else:
        n = args.limit * 2 if args.limit else None
        examples = build_lp_example_set(tag, tokenizer, tag.train_idx, n_examples=n, **kw)

    lengths = sorted(e.meta["n_tokens"] for e in examples)
    nodes = sorted(e.meta["n_nodes"] for e in examples)
    dropped = sum(e.meta["n_dropped"] for e in examples)

    print(f"\n{len(lengths)} examples, {dropped} neighbour blocks dropped at probe length")
    print(f"nodes per example: min {nodes[0]}, median {quantile(nodes, 0.5)}, max {nodes[-1]}")
    print("\ntoken length distribution")
    for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
        print(f"  p{int(q * 100):<3d} {quantile(lengths, q):5d}")
    print(f"  mean {sum(lengths) / len(lengths):8.1f}")

    print("\nchoosing seq_len (truncation rate = fraction of examples that would lose a block)")
    for cand in (512, 768, 1024, 1280, 1536, 2048, 3072, 4096):
        over = sum(1 for x in lengths if x > cand)
        rel = (cand / 2048) ** 2
        print(f"  S={cand:5d}  truncated {over:5d}/{len(lengths)} ({100 * over / len(lengths):5.2f}%)"
              f"   bias+score cost vs S=2048: {rel:5.2f}x")

    print("\nNote: cost scales with S^2 for the bias and the materialised score matrix, which is "
          "why this is worth choosing rather than defaulting.")

    # What dynamic padding actually buys, against the fixed-2048 baseline the budget assumed.
    print("\ndynamic padding vs a fixed 2048 buffer")
    fixed = 2048
    for m in (1, 64, 128, 256):
        padded = [min(fixed, ((x + m - 1) // m) * m) for x in lengths]
        lin = sum(padded) / (fixed * len(padded))
        quad = sum(p * p for p in padded) / (fixed * fixed * len(padded))
        print(f"  pad to multiple of {m:4d}: mean padded {sum(padded) / len(padded):7.1f}, "
              f"max {max(padded):5d}   linear {lin:5.3f}x  quadratic {quad:5.3f}x")
    print("  (linear terms are the projections; quadratic are the bias and the score matrix, "
          "which a dense float bias forces to be materialised)")


if __name__ == "__main__":
    main()
