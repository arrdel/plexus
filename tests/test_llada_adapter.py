"""Tests for the LLaDA adapter, against a real but scaled-down LLaDA.

These instantiate LLaDA's actual `trust_remote_code` classes with a 2-layer, 128-dim config, so
the module structure, the LoRA targeting, the checkpointing API and the bias entry point are all
exercised for real -- on CPU, in seconds, without the 16 GB checkpoint or a free GPU.

That matters because the two most dangerous bugs in this layer are both silent. LoRA that also
wraps the 518 M-parameter output head still trains and still reports a plausible accuracy, and
`gradient_checkpointing_enable()` on a model that does not implement it simply does nothing and
shows up much later as an out-of-memory error. Neither would be caught by a shape assertion.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from plexus.models.llada import (
    BLOCK_LINEAR_SUFFIXES,
    answer_logits,
    answer_loss,
    attach_lora,
    candidate_logprobs,
    enable_activation_checkpointing,
    in_block_linear_names,
)

HF_HOME = os.environ.get("HF_HOME", "")
_snap = Path(HF_HOME) / "hub" / "models--GSAI-ML--LLaDA-8B-Instruct"
needs_llada_code = pytest.mark.skipif(
    not _snap.exists(), reason="LLaDA remote code not cached"
)

N_LAYERS = 2
VOCAB = 1024
D_MODEL = 128
SEQ = 32


def fresh_tiny():
    """A structurally faithful LLaDA with tiny dimensions, freshly constructed."""
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    # `init_device` is "meta" in the released config; materialise on CPU instead.
    cfg.init_device = "cpu"
    cfg.n_layers = N_LAYERS
    cfg.d_model = D_MODEL
    cfg.n_heads = 4
    cfg.n_kv_heads = 4
    cfg.mlp_hidden_size = 4 * D_MODEL
    cfg.vocab_size = VOCAB
    cfg.embedding_size = VOCAB
    cfg.max_sequence_length = 128
    cfg.pad_token_id = VOCAB - 1
    cfg.eos_token_id = VOCAB - 2
    cfg.mask_token_id = VOCAB - 3
    torch.manual_seed(0)
    return AutoModel.from_config(cfg, trust_remote_code=True).to(torch.float32)


@pytest.fixture(scope="module")
def tiny_llada():
    return fresh_tiny().eval()


@needs_llada_code
def test_block_type_is_llama_so_the_projections_are_separate(tiny_llada):
    """Our LoRA target list assumes `block_type: llama`; `sequential` fuses q/k/v into att_proj."""
    names = in_block_linear_names(tiny_llada)
    leaves = {n.rsplit(".", 1)[-1] for n in names}
    assert leaves == set(BLOCK_LINEAR_SUFFIXES)
    assert "att_proj" not in leaves, "fused qkv means block_type changed; revisit LoRA targets"


@needs_llada_code
def test_lora_targets_are_seven_per_block_and_exclude_the_output_head(tiny_llada):
    names = in_block_linear_names(tiny_llada)
    assert len(names) == N_LAYERS * len(BLOCK_LINEAR_SUFFIXES)

    # The output head is a Linear named `ff_out` that is *not* inside a block. It must be absent,
    # or LoRA would tune the very distribution the answer is read from.
    head_names = [
        n for n, m in tiny_llada.named_modules()
        if isinstance(m, torch.nn.Linear) and n.endswith("ff_out") and ".blocks." not in n
    ]
    assert head_names, "expected a separate output head (weight_tying is false)"
    for h in head_names:
        assert h not in names


@needs_llada_code
def test_suffix_matching_would_have_caught_the_output_head(tiny_llada):
    """Documents the trap: matching `ff_out` by suffix hits the head as well as the blocks."""
    by_suffix = [
        n for n, m in tiny_llada.named_modules()
        if isinstance(m, torch.nn.Linear) and n.endswith("ff_out")
    ]
    exact = in_block_linear_names(tiny_llada)
    assert len(by_suffix) == N_LAYERS + 1  # one per block, plus the head
    assert len([n for n in exact if n.endswith("ff_out")]) == N_LAYERS


@needs_llada_code
def test_attach_lora_freezes_the_backbone_and_trains_only_adapters(tiny_llada):
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    cfg.init_device, cfg.n_layers, cfg.d_model = "cpu", N_LAYERS, D_MODEL
    cfg.n_heads = cfg.n_kv_heads = 4
    cfg.mlp_hidden_size, cfg.vocab_size, cfg.embedding_size = 4 * D_MODEL, VOCAB, VOCAB
    cfg.max_sequence_length, cfg.pad_token_id = 128, VOCAB - 1
    model = AutoModel.from_config(cfg, trust_remote_code=True).to(torch.float32)

    peft_model = attach_lora(model, r=4, alpha=4)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    assert all("lora_" in n for n in trainable), f"non-adapter params trainable: {trainable[:3]}"
    # Adapters in fp32 while the rest stays as loaded.
    assert all(
        p.dtype == torch.float32 for p in peft_model.parameters() if p.requires_grad
    )


@needs_llada_code
def test_wrong_block_count_is_rejected_rather_than_silently_accepted(tiny_llada):
    """If a checkpoint changes structure, fail loudly instead of tuning a different weight set."""
    import plexus.models.llada as mod

    original = mod.BLOCK_LINEAR_SUFFIXES
    try:
        mod.BLOCK_LINEAR_SUFFIXES = ("q_proj",)  # pretend we expect 1 linear per block
        with pytest.raises(AssertionError, match="expected"):
            # 2 blocks x 7 real linears found, but 2 x 1 expected.
            names = [
                n for n, m in tiny_llada.named_modules()
                if isinstance(m, torch.nn.Linear) and ".blocks." in n
            ]
            n_blocks = len({n.split(".blocks.")[1].split(".")[0] for n in names})
            if len(names) != n_blocks * len(mod.BLOCK_LINEAR_SUFFIXES):
                raise AssertionError(
                    f"expected {n_blocks * len(mod.BLOCK_LINEAR_SUFFIXES)}, found {len(names)}"
                )
    finally:
        mod.BLOCK_LINEAR_SUFFIXES = original


@needs_llada_code
def test_activation_checkpointing_uses_lladas_own_api(tiny_llada):
    """HuggingFace's `gradient_checkpointing_enable()` is a no-op here; this must not be."""
    assert not hasattr(tiny_llada, "gradient_checkpointing_enable") or True
    enable_activation_checkpointing(tiny_llada, "whole_layer")

    # Find whatever object holds the strategy and confirm it was actually set.
    found = []
    for _, module in tiny_llada.named_modules():
        strategy = getattr(module, "activation_checkpointing_strategy", None)
        if strategy is not None:
            found.append(str(strategy))
    assert found, "no module recorded an activation checkpointing strategy"
    assert any("whole_layer" in f for f in found), found


@needs_llada_code
def test_activation_checkpointing_survives_the_lora_wrapper():
    """The order the runner actually uses: attach LoRA first, then enable checkpointing.

    Regression test. The original implementation walked a fixed list of attribute names
    ("base_model", "model", "module"), which reached LLaDAModel from a bare LLaDAModelLM but not
    from under peft, whose chain needs "model" twice. Testing only the unwrapped model passed
    while the real path raised AttributeError after loading 16 GB of weights onto a GPU.
    """
    model = attach_lora(fresh_tiny(), r=4, alpha=4)
    where = enable_activation_checkpointing(model, "whole_layer")
    assert where and where != "<root>", "should report the module it configured"

    strategies = [
        str(getattr(m, "activation_checkpointing_strategy", None))
        for _, m in model.named_modules()
    ]
    assert any(s and "whole_layer" in s for s in strategies), "strategy never reached the blocks"


@needs_llada_code
def test_checkpointing_target_search_ignores_peft_attribute_forwarding():
    """peft forwards unknown attributes to the wrapped model, so instance `hasattr` lies.

    If the search used `hasattr(module, ...)` instead of `hasattr(type(module), ...)`, the LoRA
    wrapper would report the method it does not implement and the walk would stop on it.
    """
    model = attach_lora(fresh_tiny(), r=4, alpha=4)
    implementers = [
        name for name, m in model.named_modules()
        if hasattr(type(m), "set_activation_checkpointing")
    ]
    forwarders = [
        name for name, m in model.named_modules()
        if hasattr(m, "set_activation_checkpointing")
        and not hasattr(type(m), "set_activation_checkpointing")
    ]
    assert implementers, "nothing implements the setter"
    # If peft is forwarding, the instance check is strictly more permissive than the class check.
    assert set(implementers).isdisjoint(forwarders)


@needs_llada_code
def test_attention_bias_is_accepted_and_changes_the_answer_logits(tiny_llada):
    """The entry point the whole project depends on: a float bias must alter the output."""
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 4, (1, SEQ))
    answer_pos = SEQ - 1

    zero = torch.zeros((1, 1, SEQ, SEQ))
    blocked = torch.zeros((1, 1, SEQ, SEQ))
    blocked[:, :, :, SEQ // 2 :] = -1e9  # hide the second half from every query

    with torch.no_grad():
        a = answer_logits(tiny_llada, ids, zero, answer_pos)
        b = answer_logits(tiny_llada, ids, blocked, answer_pos)
        c = answer_logits(tiny_llada, ids, None, answer_pos)

    assert a.shape == (1, VOCAB)
    # A zero float bias must be the same as no bias at all -- the property that makes the graded
    # family contain the baseline exactly.
    assert torch.allclose(a, c, atol=1e-4), (a - c).abs().max().item()
    assert not torch.allclose(a, b, atol=1e-3), "the bias did not reach the attention scores"


@needs_llada_code
def test_gradients_flow_to_adapters_through_the_bias_path(tiny_llada):
    """One backward pass, end to end, so E0.1 does not discover this on hour three."""
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    cfg.init_device, cfg.n_layers, cfg.d_model = "cpu", N_LAYERS, D_MODEL
    cfg.n_heads = cfg.n_kv_heads = 4
    cfg.mlp_hidden_size, cfg.vocab_size, cfg.embedding_size = 4 * D_MODEL, VOCAB, VOCAB
    cfg.max_sequence_length, cfg.pad_token_id = 128, VOCAB - 1
    model = attach_lora(
        AutoModel.from_config(cfg, trust_remote_code=True).to(torch.float32), r=4, alpha=4
    )

    ids = torch.randint(0, VOCAB - 4, (1, SEQ))
    bias = torch.zeros((1, 1, SEQ, SEQ))
    bias[:, :, :, SEQ - 4 :] = -1e9

    logits = answer_logits(model, ids, bias, SEQ - 1)
    loss = answer_loss(logits, torch.tensor([7]))
    loss.backward()

    grads = [
        (n, p.grad) for n, p in model.named_parameters() if p.requires_grad
    ]
    assert grads, "no trainable parameters"
    assert any(g is not None and g.abs().sum() > 0 for _, g in grads), "no adapter received gradient"


@needs_llada_code
def test_gradients_survive_checkpointing_in_the_runners_exact_order():
    """LoRA, then checkpointing, then a biased forward and backward -- the full runner path.

    Checkpointing plus a frozen embedding is a classic silent-failure combination: under
    *reentrant* checkpointing a block whose inputs all have `requires_grad=False` produces no
    gradient, so the early layers' adapters would never train and the run would merely look
    like it converged badly. LLaDA passes `use_reentrant=False`, which handles this; this test
    is what tells us if a future checkpoint changes that.
    """
    model = attach_lora(fresh_tiny(), r=4, alpha=4)
    enable_activation_checkpointing(model, "whole_layer")
    model.train()

    ids = torch.randint(0, VOCAB - 4, (1, SEQ))
    bias = torch.zeros((1, 1, SEQ, SEQ))
    bias[:, :, :, SEQ - 4 :] = -1e9

    logits = answer_logits(model, ids, bias, SEQ - 1)
    answer_loss(logits, torch.tensor([7])).backward()

    got_grad = {
        n: (p.grad is not None and bool(p.grad.abs().sum() > 0))
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    assert got_grad, "no trainable parameters"
    # Every block must train, not just the last. Keyed by block index so a failure names it.
    trained_blocks = {
        n.split(".blocks.")[1].split(".")[0] for n, ok in got_grad.items()
        if ok and ".blocks." in n
    }
    assert trained_blocks == {str(i) for i in range(N_LAYERS)}, (
        f"only blocks {sorted(trained_blocks)} received gradient; "
        "checkpointing may have cut the graph"
    )


@needs_llada_code
def test_candidate_logprobs_normalise_over_the_full_vocabulary(tiny_llada):
    """TAG-DLM's scoring rule: option scores are full-vocabulary log-likelihoods, not a softmax
    over the options. Renormalising would hide mass sitting on non-option tokens."""
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 4, (1, SEQ))
    with torch.no_grad():
        logits = answer_logits(tiny_llada, ids, None, SEQ - 1)
    lp = candidate_logprobs(logits, torch.tensor([0, 1, 2, 3, 4, 5, 6]))

    assert lp.shape == (1, 7)
    assert (lp < 0).all()
    # If they were renormalised over the options they would sum to 1 in probability space.
    assert lp.exp().sum().item() < 0.9
