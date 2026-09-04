"""Measure steady-state forward+backward throughput, which sets the whole compute budget.

This is the measurement E0.0e owed. That entry reported ~16% MFU from a forward pass that also
requested `output_hidden_states` (retaining all 33 layers, ~1.1 GB) while a co-tenant job held the
other card, and it flagged both as contamination. The smoke run's 5.6 s over 8 micro-steps is
likewise unusable: with `accum_steps=128` no optimiser step completed and warmup dominates.

So: run N steps, discard the first `--warmup` of them, and report the median. Median rather than
mean because a single allocator stall should not set the project's plan.

    python scripts/benchmark_step.py --steps 24 --warmup 6
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from plexus.data.examples import build_nc_examples, collate
from plexus.data.tags import load_tag
from plexus.models.llada import (
    MASK_TOKEN_ID,
    answer_logits,
    answer_loss,
    attach_lora,
    enable_activation_checkpointing,
    load_backbone,
    load_tokenizer,
)

# bf16 dense matmul peak, measured on this 4090 in E0.0d via an 8192^3 matmul.
PEAK_TFLOPS = 149.5
# LoRA fwd+bwd: 2N forward, ~4N backward (activation grads through frozen weights dominate;
# the 168 M adapter params are negligible). N = 8.02e9.
FLOPS_PER_TOKEN = 6 * 8.02e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--mask-family", default="star")
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data-root", default=os.environ.get(
        "PLEXUS_DATA_ROOT", "/tmp/adelechinda/plexus_data"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pad-to-multiple-of", type=int, default=128)
    p.add_argument("--fixed-seq-len", type=int, default=None,
                   help="pad every example to this, to quantify what dynamic padding saves")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tag = load_tag(args.dataset, args.data_root, seed=0)
    tokenizer = load_tokenizer()

    n_examples = args.steps + args.warmup
    centers = tag.train_idx.tolist()[:n_examples]
    pad_mult = args.fixed_seq_len or args.pad_to_multiple_of
    examples = build_nc_examples(
        tag, tokenizer, centers, k=2, K=10, seed=0, seq_len=2048,
        mask_token_id=MASK_TOKEN_ID, pad_to_multiple_of=pad_mult,
    )
    lengths = [int(e.input_ids.numel()) for e in examples]
    print(f"{len(examples)} examples, padded length min/mean/max "
          f"{min(lengths)}/{sum(lengths) / len(lengths):.0f}/{max(lengths)}", flush=True)

    model, _ = load_backbone(device=args.device, check_snapshot=False)
    model = attach_lora(model)
    enable_activation_checkpointing(model)
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5)

    records = []
    for i, ex in enumerate(examples):
        batch = collate([ex], family=args.mask_family, device=args.device,
                        bias_dtype=torch.bfloat16)
        torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        logits = answer_logits(model, batch["input_ids"], batch["attention_bias"],
                               batch["answer_pos"])
        loss = answer_loss(logits, batch["target_token_ids"])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(args.device)
        dt = time.perf_counter() - t0

        n_tok = int(ex.input_ids.numel())
        records.append({"step": i, "seconds": dt, "tokens": n_tok,
                        "warmup": i < args.warmup})
        if i < args.warmup or i % 4 == 0:
            print(f"  step {i:3d}{' (warmup)' if i < args.warmup else ''}: "
                  f"{dt * 1000:7.1f} ms  {n_tok:5d} tok  "
                  f"{n_tok / dt:7.1f} tok/s", flush=True)

    steady = [r for r in records if not r["warmup"]]
    times = [r["seconds"] for r in steady]
    toks = [r["tokens"] for r in steady]
    med_t = statistics.median(times)
    med_tok = statistics.median(toks)
    tok_per_s = sum(toks) / sum(times)
    tflops = tok_per_s * FLOPS_PER_TOKEN / 1e12
    mfu = 100 * tflops / PEAK_TFLOPS
    peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

    print("\n" + "=" * 68)
    print(f"steady state over {len(steady)} steps (first {args.warmup} discarded)")
    print(f"  median step        {med_t * 1000:8.1f} ms  at median {med_tok:.0f} tokens")
    print(f"  min / max step     {min(times) * 1000:8.1f} / {max(times) * 1000:.1f} ms")
    print(f"  throughput         {tok_per_s:8.1f} tokens/s")
    print(f"  effective          {tflops:8.1f} TFLOPs  of {PEAK_TFLOPS} peak")
    print(f"  MFU                {mfu:8.1f} %")
    print(f"  peak memory        {peak_gb:8.2f} GB")

    # What this implies for the runs the plan is scheduled around.
    print("\nprojected wall clock at this throughput")
    for name, n_ex, epochs in [("Cora NC, 20 epochs", 1624, 20),
                               ("Cora NC, 3 epochs", 1624, 3)]:
        total_tok = n_ex * epochs * (sum(toks) / len(toks))
        hours = total_tok / tok_per_s / 3600
        print(f"  {name:22s} {total_tok:12.0f} tok  {hours:7.2f} h")

    summary = {
        "median_step_s": med_t, "tokens_per_s": tok_per_s, "tflops": tflops,
        "mfu_percent": mfu, "peak_gb": peak_gb, "n_steady": len(steady),
        "warmup": args.warmup, "mask_family": args.mask_family,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "fixed_seq_len": args.fixed_seq_len,
        "padded_len_mean": sum(lengths) / len(lengths),
        "records": records,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
