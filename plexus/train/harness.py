"""LoRA fine-tuning and evaluation for a fixed mask family.

Defaults reproduce TAG-DLM's recipe (E0.0 item 6): 20 epochs, AdamW at lr 5e-5, effective batch
128, sequence length 2048. On a 24 GB 4090 the micro-batch is forced to 1, so the effective batch
is reached by 128 gradient-accumulation steps -- which is why §6's MFU assumption is optimistic
and why `train` records real throughput as it goes rather than trusting an estimate.

Everything that could drift between runs is in `TrainConfig` and gets written next to the results,
including the dataset split provenance. A number without its configuration is not reproducible.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from plexus.data.examples import collate
from plexus.masks import HopBias, describe
from plexus.metrics import Result, accuracy
from plexus.models.llada import answer_logits, answer_loss, candidate_logprobs
from plexus.prompts.templates import LinearisedExample


@dataclass
class TrainConfig:
    """Everything that defines a run. Serialised verbatim into the experiment directory."""

    task: str = "nc"                     # nc | lp
    mask_family: str = "star"
    dataset: str = "cora"
    seed: int = 0
    seq_len: int = 2048

    # TAG-DLM's recipe.
    epochs: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.0
    effective_batch: int = 128
    micro_batch: int = 1                 # forced by 24 GB; see RESEARCH_PLAN.md §6
    warmup_frac: float = 0.03
    grad_clip: float = 1.0

    # Sampling.
    k_hops: int = 2
    max_neighbours_per_hop: int = 10

    # Graded bias (Phase 1). None means a hard mask family only.
    hop_bias_buckets: int | None = None
    hop_bias_init: float = 0.0
    hop_bias_answer_as_target: bool = False

    device: str = "cuda:0"
    bias_dtype: str = "bfloat16"
    eval_every_epochs: int = 0           # 0 = only at the end
    max_eval_examples: int | None = None
    log_every_steps: int = 20
    notes: str = ""
    provenance: dict = field(default_factory=dict)

    @property
    def accum_steps(self) -> int:
        return max(1, self.effective_batch // self.micro_batch)


def _lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay."""
    warmup = max(1, int(total * cfg.warmup_frac))
    if step < warmup:
        return cfg.lr * step / warmup
    import math

    progress = (step - warmup) / max(1, total - warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _make_hop_bias(cfg: TrainConfig, device: str) -> HopBias | None:
    if cfg.hop_bias_buckets is None:
        return None
    values = torch.full(
        (cfg.hop_bias_buckets,), cfg.hop_bias_init, dtype=torch.float32, device=device
    )
    values = torch.nn.Parameter(values)
    return HopBias(values=values, answer_as_target=cfg.hop_bias_answer_as_target)


@torch.no_grad()
def evaluate(
    model,
    examples: list[LinearisedExample],
    cfg: TrainConfig,
    *,
    hop_bias: HopBias | None = None,
    max_examples: int | None = None,
) -> Result:
    """Score every example at its single `[M]` and take the argmax over the answer options."""
    model.eval()
    bias_dtype = getattr(torch, cfg.bias_dtype)
    subset = examples[: max_examples or len(examples)]

    preds: list[int] = []
    labels: list[int] = []
    densities: list[float] = []
    for i in range(0, len(subset), cfg.micro_batch):
        chunk = subset[i : i + cfg.micro_batch]
        batch = collate(
            chunk, family=cfg.mask_family, hop_bias=hop_bias,
            device=cfg.device, bias_dtype=bias_dtype,
            generator=torch.Generator().manual_seed(cfg.seed * 100_003 + i),
        )
        logits = answer_logits(
            model, batch["input_ids"], batch["attention_bias"], batch["answer_pos"]
        )
        lp = candidate_logprobs(logits, batch["candidate_ids"])
        preds.extend(lp.argmax(-1).tolist())
        labels.extend(batch["labels"].tolist())
        if i == 0:
            densities.append(describe(batch["attention_bias"].float())["density"])

    return Result(
        accuracy=accuracy(preds, labels),
        n=len(labels),
        mask_family=cfg.mask_family,
        mask_density=densities[0] if densities else float("nan"),
        task=cfg.task,
        seed=cfg.seed,
    )


def train(
    model,
    train_examples: list[LinearisedExample],
    cfg: TrainConfig,
    *,
    eval_examples: list[LinearisedExample] | None = None,
    out_dir: str | Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """LoRA fine-tune on one mask family. Returns a history dict; also writes it to `out_dir`.

    `max_steps` short-circuits the run and exists for the smoke test: it is the difference between
    validating the harness in two minutes and validating it in two days.
    """
    device = cfg.device
    bias_dtype = getattr(torch, cfg.bias_dtype)
    hop_bias = _make_hop_bias(cfg, device)

    params = [p for p in model.parameters() if p.requires_grad]
    if hop_bias is not None:
        params = params + [hop_bias.values]
    n_trainable = sum(p.numel() for p in params)
    optimiser = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    micro_per_epoch = max(1, len(train_examples) // cfg.micro_batch)
    total_micro = micro_per_epoch * cfg.epochs
    total_opt_steps = max(1, total_micro // cfg.accum_steps)

    history: dict = {
        "config": asdict(cfg),
        "n_trainable": n_trainable,
        "n_train_examples": len(train_examples),
        "total_opt_steps": total_opt_steps,
        "steps": [],
        "evals": [],
    }

    rng = torch.Generator().manual_seed(cfg.seed)
    micro_step = 0
    opt_step = 0
    t_start = time.time()
    tokens_seen = 0
    stop = False

    for epoch in range(cfg.epochs):
        if stop:
            break
        model.train()
        order = torch.randperm(len(train_examples), generator=rng).tolist()
        optimiser.zero_grad(set_to_none=True)

        for j in range(0, len(order) - cfg.micro_batch + 1, cfg.micro_batch):
            chunk = [train_examples[t] for t in order[j : j + cfg.micro_batch]]
            batch = collate(
                chunk, family=cfg.mask_family, hop_bias=hop_bias,
                device=device, bias_dtype=bias_dtype,
                generator=torch.Generator().manual_seed(cfg.seed * 7919 + micro_step),
            )
            logits = answer_logits(
                model, batch["input_ids"], batch["attention_bias"], batch["answer_pos"]
            )
            loss = answer_loss(logits, batch["target_token_ids"]) / cfg.accum_steps
            loss.backward()

            tokens_seen += int(sum(e.meta.get("n_tokens", cfg.seq_len) for e in chunk))
            micro_step += 1

            if micro_step % cfg.accum_steps == 0:
                for group in optimiser.param_groups:
                    # opt_step + 1: step counting is 1-based here, so the first update gets
                    # lr > 0 rather than being wasted at the zero end of the warmup ramp.
                    group["lr"] = _lr_at(opt_step + 1, total_opt_steps, cfg)
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % cfg.log_every_steps == 0 or opt_step == 1:
                    elapsed = time.time() - t_start
                    rec = {
                        "opt_step": opt_step,
                        "epoch": epoch,
                        "loss": float(loss.item() * cfg.accum_steps),
                        "lr": optimiser.param_groups[0]["lr"],
                        "elapsed_s": elapsed,
                        "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
                        "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9
                        if device.startswith("cuda") else 0.0,
                    }
                    history["steps"].append(rec)
                    print(
                        f"  step {opt_step:5d}/{total_opt_steps} epoch {epoch:3d} "
                        f"loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                        f"{rec['tokens_per_s']:.0f} tok/s peak {rec['peak_gb']:.2f} GB",
                        flush=True,
                    )

            if max_steps is not None and micro_step >= max_steps:
                stop = True
                break

        do_eval = (
            eval_examples
            and cfg.eval_every_epochs
            and (epoch + 1) % cfg.eval_every_epochs == 0
        )
        if do_eval:
            res = evaluate(model, eval_examples, cfg, hop_bias=hop_bias,
                           max_examples=cfg.max_eval_examples)
            history["evals"].append({"epoch": epoch, **res.__dict__})
            print(f"  [eval] {res.row()}", flush=True)
            model.train()

    history["wall_clock_s"] = time.time() - t_start
    history["tokens_seen"] = tokens_seen
    if device.startswith("cuda"):
        history["peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(history, indent=2, default=str))
    return history
