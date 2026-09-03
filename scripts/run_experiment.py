"""Run one PLEXUS configuration: one task, one mask family, one seed.

    python scripts/run_experiment.py --task nc --mask-family star --seed 0 \
        --out experiments/E0.1_cora_nc_star_s0

Smoke test the whole path in a couple of minutes instead of a couple of days:

    python scripts/run_experiment.py --smoke --out /tmp/smoke

Every run writes `config.json`, `history.json` and `result.json` into `--out`, including the
dataset split provenance, so a number is never separated from the assumptions behind it.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from plexus.data.examples import build_lp_example_set, build_nc_examples
from plexus.data.tags import SPLIT_PROVENANCE, load_tag
from plexus.metrics import Result
from plexus.models.llada import (
    MASK_TOKEN_ID,
    assert_snapshot_present,
    attach_lora,
    enable_activation_checkpointing,
    load_backbone,
    load_tokenizer,
)
from plexus.train import TrainConfig, evaluate, train

DEFAULT_DATA_ROOT = os.environ.get("PLEXUS_DATA_ROOT", "/tmp/adelechinda/plexus_data")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="nc", choices=["nc", "lp"])
    p.add_argument("--mask-family", default="star",
                   choices=["none", "star", "ego", "ego_star", "random_rewire"])
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--effective-batch", type=int, default=128)
    p.add_argument("--k-hops", type=int, default=2)
    p.add_argument("--max-neighbours", type=int, default=10)
    p.add_argument("--hop-bias-buckets", type=int, default=None,
                   help="enable a graded HopBias with this many buckets (Phase 1)")
    p.add_argument("--random-node-control", action="store_true",
                   help="E0.3: replace graph neighbours with random nodes")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every-epochs", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-lora", action="store_true",
                   help="evaluate the frozen backbone (TAG-DLM's near-chance control)")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: 24 train examples, 24 eval, 8 optimiser micro-steps")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.limit_train = args.limit_train or 24
        args.limit_eval = args.limit_eval or 24
        args.max_steps = args.max_steps or 8
        args.epochs = 1

    torch.manual_seed(args.seed)

    print(f"loading {args.dataset} from {args.data_root} ...", flush=True)
    tag = load_tag(args.dataset, args.data_root, seed=args.seed)
    print(" ", tag.summary(), flush=True)

    # Everything that does not need the GPU happens first. On a contended machine a free card is
    # a short window, and loading 16 GB of weights before discovering a tokenisation or prompt
    # bug throws that window away -- which is exactly what happened on 2026-09-03.
    assert_snapshot_present()
    tokenizer = load_tokenizer()

    prompt_kw = {"mask_token_id": MASK_TOKEN_ID}
    print("building examples ...", flush=True)
    if args.task == "nc":
        train_centers = tag.train_idx.tolist()[: args.limit_train]
        test_centers = tag.test_idx.tolist()[: args.limit_eval]
        train_ex = build_nc_examples(
            tag, tokenizer, train_centers, k=args.k_hops, K=args.max_neighbours,
            seed=args.seed, seq_len=args.seq_len,
            random_node_control=args.random_node_control, **prompt_kw,
        )
        test_ex = build_nc_examples(
            tag, tokenizer, test_centers, k=args.k_hops, K=args.max_neighbours,
            seed=args.seed, seq_len=args.seq_len,
            random_node_control=args.random_node_control, **prompt_kw,
        )
    else:
        n_tr = args.limit_train * 2 if args.limit_train else None
        n_te = args.limit_eval * 2 if args.limit_eval else None
        train_ex = build_lp_example_set(
            tag, tokenizer, tag.train_idx, n_examples=n_tr, k=args.k_hops,
            K=args.max_neighbours, seed=args.seed, seq_len=args.seq_len, **prompt_kw,
        )
        test_ex = build_lp_example_set(
            tag, tokenizer, tag.test_idx, n_examples=n_te, k=args.k_hops,
            K=args.max_neighbours, seed=args.seed, seq_len=args.seq_len, **prompt_kw,
        )

    tok_lens = [e.meta["n_tokens"] for e in train_ex]
    dropped = sum(e.meta["n_dropped"] for e in train_ex)
    print(f"  train {len(train_ex)}, eval {len(test_ex)}; "
          f"tokens min/mean/max {min(tok_lens)}/{sum(tok_lens) / len(tok_lens):.0f}/{max(tok_lens)}; "
          f"neighbours dropped {dropped}", flush=True)

    print(f"loading backbone on {args.device} ...", flush=True)
    model, _ = load_backbone(device=args.device, check_snapshot=False)
    n_base = sum(p.numel() for p in model.parameters())

    if not args.no_lora:
        model = attach_lora(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  backbone {n_base / 1e9:.2f}B params, LoRA trainable "
              f"{n_trainable / 1e6:.1f}M ({100 * n_trainable / n_base:.2f}%)", flush=True)
        where = enable_activation_checkpointing(model)
        print(f"  activation checkpointing (whole_layer) set on {where}", flush=True)
    else:
        n_trainable = 0
        print(f"  backbone {n_base / 1e9:.2f}B params, frozen (no LoRA)", flush=True)

    cfg = TrainConfig(
        task=args.task, mask_family=args.mask_family, dataset=args.dataset, seed=args.seed,
        seq_len=args.seq_len, epochs=args.epochs, lr=args.lr,
        effective_batch=args.effective_batch, k_hops=args.k_hops,
        max_neighbours_per_hop=args.max_neighbours,
        hop_bias_buckets=args.hop_bias_buckets, device=args.device,
        eval_every_epochs=args.eval_every_epochs, max_eval_examples=args.limit_eval,
        provenance={
            "split": SPLIT_PROVENANCE,
            "random_node_control": args.random_node_control,
            "no_lora": args.no_lora,
            "text_coverage": tag.text_coverage,
            "n_trainable": n_trainable,
        },
    )

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    if args.no_lora:
        history = {"config": asdict(cfg), "note": "frozen backbone, no training"}
    else:
        print("training ...", flush=True)
        history = train(model, train_ex, cfg, eval_examples=test_ex,
                        out_dir=out, max_steps=args.max_steps)
        print(f"  wall clock {history['wall_clock_s']:.1f}s, "
              f"peak {history.get('peak_gb', 0):.2f} GB", flush=True)

    print("evaluating ...", flush=True)
    result: Result = evaluate(model, test_ex, cfg, max_examples=args.limit_eval)
    print(" ", result.row(), flush=True)

    if out:
        (out / "result.json").write_text(json.dumps(result.__dict__, indent=2, default=str))
        (out / "history.json").write_text(json.dumps(history, indent=2, default=str))
        print(f"wrote {out}/", flush=True)


if __name__ == "__main__":
    main()
