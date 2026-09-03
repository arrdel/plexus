"""Model adapters: LLaDA with topology biases, and the autoregressive control for E0.4."""

from plexus.models.llada import (
    LORA_ALPHA,
    LORA_R,
    answer_logits,
    assert_snapshot_present,
    attach_lora,
    candidate_logprobs,
    enable_activation_checkpointing,
    in_block_linear_names,
    load_backbone,
)

__all__ = [
    "LORA_ALPHA",
    "LORA_R",
    "answer_logits",
    "assert_snapshot_present",
    "attach_lora",
    "candidate_logprobs",
    "enable_activation_checkpointing",
    "in_block_linear_names",
    "load_backbone",
]
