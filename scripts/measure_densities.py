"""Measure mask density per family over the real dataset. Needs no GPU and no training.

Density is a property of the mask and the graph, not of a trained model, so there is no reason for
the paper's density column to wait on the run queue. It is also the confound the standing rule
exists for: two families that differ in accuracy also differ in how much context each token may
read, and quantifying that gap is what separates "this topology is better" from "this topology
lets more tokens talk".

    python scripts/measure_densities.py --out experiments/densities.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from plexus.data.examples import build_lp_example_set, build_nc_examples
from plexus.data.tags import load_tag
from plexus.masks import HopBias, build_attention_bias, mask_density
from plexus.models.llada import MASK_TOKEN_ID, load_tokenizer

FAMILIES = ["none", "star", "ego", "ego_star", "random_rewire"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data-root", default=os.environ.get(
        "PLEXUS_DATA_ROOT", "/tmp/adelechinda/plexus_data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=400,
                   help="examples per task; density is stable well before the full split")
    p.add_argument("--hop-bias-buckets", type=int, default=6)
    p.add_argument("--out", default="experiments/densities.json")
    args = p.parse_args()

    tag = load_tag(args.dataset, args.data_root, seed=args.seed)
    tokenizer = load_tokenizer()
    kw = dict(k=2, K=10, seed=args.seed, seq_len=2048, mask_token_id=MASK_TOKEN_ID,
              pad_to_multiple_of=128)

    out = {"dataset": args.dataset, "seed": args.seed, "limit": args.limit, "tasks": {}}
    for task in ("nc", "lp"):
        if task == "nc":
            examples = build_nc_examples(tag, tokenizer, tag.train_idx.tolist()[: args.limit],
                                         **kw)
        else:
            examples = build_lp_example_set(tag, tokenizer, tag.train_idx,
                                            n_examples=args.limit, **kw)
        print(f"\n{task}: {len(examples)} examples", flush=True)

        rec = {}
        gen = torch.Generator().manual_seed(args.seed)
        for fam in FAMILIES:
            ds, hops = [], []
            for e in examples:
                b = build_attention_bias([e.graph], family=fam, dtype=torch.float32,
                                         generator=gen)
                ds.append(mask_density(b))
                hops.append(max(h for h in e.graph.hops))
            m = statistics.mean(ds)
            sd = statistics.stdev(ds) if len(ds) > 1 else 0.0
            rec[fam] = {"mean": m, "std": sd, "n": len(ds)}
            print(f"  {fam:15s} rho = {m:.4f} +/- {sd:.4f}", flush=True)

        # A graded bias gates nothing, so its density is exactly 1 by construction. Measured
        # rather than asserted, because it is the property that makes the arm safe.
        hb = HopBias(values=torch.zeros(args.hop_bias_buckets))
        ds = [mask_density(build_attention_bias([e.graph], family="none", hop_bias=hb,
                                                dtype=torch.float32))
              for e in examples[:50]]
        rec["hop_bias"] = {"mean": statistics.mean(ds), "std": 0.0, "n": len(ds)}
        print(f"  {'hop_bias':15s} rho = {statistics.mean(ds):.4f} (gates nothing)")

        rec["_max_hop"] = {"mean": statistics.mean(hops), "n": len(hops)}
        out["tasks"][task] = rec

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
