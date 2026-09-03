"""LLaDA-8B with LoRA and a topology attention bias.

TAG-DLM's recipe: freeze LLaDA-8B-Instruct, LoRA r=64 alpha=64 on all linear layers, and score a
single `[M]` position by the log-likelihood of a digit token in one forward pass. This module is
that, plus the `attention_bias` entry point the mask families need.

Three things here are not boilerplate and are the reason this file exists:

1. **`ff_out` is an ambiguous name in LLaDA.** It is both the per-block down-projection
   (`transformer.blocks.{i}.ff_out`) and the model's output head (`transformer.ff_out`, 518 M
   parameters, since `weight_tying` is false). Matching LoRA targets by suffix -- what
   `target_modules=["ff_out", ...]` or `"all-linear"` does -- would wrap the output head too,
   silently changing the scoring distribution the whole method is read off. So we enumerate exact
   module paths and assert the head is excluded.
2. **Activation checkpointing is LLaDA's own API**, not HuggingFace's
   `gradient_checkpointing_enable()`. It takes an `ActivationCheckpointingStrategy`; we use
   `whole_layer`, which the memory budget in RESEARCH_PLAN.md §6 assumes.
3. **We never request `output_hidden_states` during training.** E0.0e measured that retaining all
   33 hidden states costs ~1.1 GB at S=2048, which is a third of the headroom on a 24 GB card.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336

# TAG-DLM's LoRA hyperparameters (§5.1 of the paper, transcribed in E0.0 item 6).
LORA_R = 64
LORA_ALPHA = 64

# The linear layers inside each transformer block, for `block_type: "llama"`. Excludes the model's
# output head, which shares the name `ff_out` -- see the module docstring.
BLOCK_LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")


def assert_snapshot_present(model_id: str = MODEL_ID, hf_home: str | None = None) -> Path:
    """Fail immediately if the 16 GB checkpoint is not on disk. Returns the snapshot directory.

    The checkpoint lives under `/tmp` because `/home` is full (E0.0f). `/tmp` is not reaped by age
    on this host, but the `tmpfiles.d` `D` verb does empty it **on reboot**. A run that discovers
    this 30 hours in fails with a confusing load error; this makes it fail in the first second
    with an actionable one. RESEARCH_PLAN.md §5 records the mitigation; this is it.
    """
    root = Path(hf_home or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    repo = root / "hub" / ("models--" + model_id.replace("/", "--"))
    snapshots = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
    for snap in snapshots:
        if (snap / "config.json").exists() and any(snap.glob("*.safetensors")):
            return snap
    raise FileNotFoundError(
        f"no usable snapshot of {model_id} under {repo}. A reboot empties /tmp, so the checkpoint "
        f"may need re-fetching:\n"
        f"  HF_HOME={root} huggingface-cli download {model_id}\n"
        f"(~16 GB; it transferred at ~117 MB/s from titan3 in E0.0d.)"
    )


def load_backbone(
    model_id: str = MODEL_ID,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    check_snapshot: bool = True,
):
    """Load LLaDA and its tokenizer. Returns `(model, tokenizer)`."""
    from transformers import AutoModel, AutoTokenizer

    if check_snapshot:
        assert_snapshot_present(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, torch_dtype=dtype
    )
    return model.to(device), tokenizer


def in_block_linear_names(model) -> list[str]:
    """Exact module paths of the per-block linear layers, with the output head excluded.

    Verified rather than assumed: LLaDA-8B has 32 blocks with 7 linear layers each, so this must
    return 224 names. If a future checkpoint changes `block_type` from `llama` to `sequential` the
    projections fuse into `att_proj`/`ff_proj` and the count changes -- better to fail here than
    to quietly LoRA-tune a different set of weights than the paper did.
    """
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ".blocks." in name
        and name.rsplit(".", 1)[-1] in BLOCK_LINEAR_SUFFIXES
    ]
    heads = [n for n in names if ".blocks." not in n]
    if heads:
        raise AssertionError(f"output head leaked into LoRA targets: {heads}")
    return sorted(names)


def attach_lora(
    model,
    *,
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = 0.0,
    fp32_adapters: bool = True,
):
    """Wrap `model` with LoRA on every in-block linear layer, and freeze everything else.

    Adapters are kept in fp32 while the frozen backbone stays bf16: 168 M trainable parameters at
    4 bytes is 0.67 GB of weights, 0.67 GB of gradients and 1.34 GB of AdamW moments, which is the
    ~2.7 GB the memory budget accounts for. peft casts activations to the adapter dtype at each
    LoRA layer, so the mixed precision is handled inside the adapter rather than by autocast.
    """
    from peft import LoraConfig, get_peft_model

    targets = in_block_linear_names(model)
    n_blocks = len({n.split(".blocks.")[1].split(".")[0] for n in targets})
    expected = n_blocks * len(BLOCK_LINEAR_SUFFIXES)
    if len(targets) != expected:
        raise AssertionError(
            f"expected {expected} LoRA targets ({n_blocks} blocks x "
            f"{len(BLOCK_LINEAR_SUFFIXES)} linears), found {len(targets)}"
        )

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )
    peft_model = get_peft_model(model, cfg)
    if fp32_adapters:
        for param in peft_model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    return peft_model


def enable_activation_checkpointing(model, strategy: str = "whole_layer") -> None:
    """Turn on LLaDA's own activation checkpointing.

    `ActivationCheckpointingStrategy` is a `StrEnum`, so the plain string compares equal to the
    enum member and we avoid importing from the trust_remote_code module.
    """
    target = model
    for attr in ("base_model", "model", "module"):
        # Unwrap peft/HF wrappers until something exposes the setter.
        if hasattr(target, "set_activation_checkpointing"):
            break
        target = getattr(target, attr, target)
    if not hasattr(target, "set_activation_checkpointing"):
        raise AttributeError("could not find set_activation_checkpointing on the model")
    target.set_activation_checkpointing(strategy)


def answer_logits(
    model,
    input_ids: torch.Tensor,
    attention_bias: torch.Tensor | None,
    answer_pos: torch.Tensor | int,
) -> torch.Tensor:
    """Logits at the single answer position. Returns `(B, vocab)`.

    One forward pass, no iterative denoising -- which is what TAG-DLM actually does (E0.0 item 1).

    The full `(B, S, vocab)` logit tensor is materialised and then sliced, costing ~0.5 GB in bf16
    at S=2048. Slicing the head instead would need the `[M]` token to sit at the final index so
    LLaDA's `last_logits_only` could be used; that is a real option if memory becomes binding,
    but it moves padding into the middle of the sequence and `plexus.masks` assumes a contiguous
    padding tail, so it is not free.
    """
    out = model(input_ids=input_ids, attention_bias=attention_bias)
    logits = out.logits
    if isinstance(answer_pos, int):
        return logits[:, answer_pos, :]
    idx = answer_pos.to(logits.device).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
    return logits.gather(1, idx).squeeze(1)


def candidate_logprobs(logits: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Log-probabilities restricted to the answer options. Returns `(B, n_options)`.

    Normalised over the **full** vocabulary before restriction, which is TAG-DLM's rule: each
    option is scored by "the log-likelihood of that digit token at the masked answer position",
    not by a softmax over the options. The two differ, and renormalising over the options would
    hide the model putting most of its mass on tokens that are not options at all -- which is
    exactly what a near-chance frozen model does (E0.0 item 7).
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return logprobs.index_select(-1, candidate_ids.to(logits.device))


def answer_loss(logits: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the answer position over the full vocabulary.

    For a single masked position with clean context, the masked-diffusion objective reduces to
    exactly this, so training matches the MDLM formulation TAG-DLM inherits without needing the
    diffusion loop the paper never runs.
    """
    return F.cross_entropy(logits.float(), target_token_ids.to(logits.device))
