"""Tests for the training harness, using a stub backbone.

The point is to validate everything around the model -- collation, the bias reaching the forward
pass, gradient accumulation, the LR schedule, single-position scoring, metric bookkeeping -- on
CPU in under a second. Those are the parts most likely to be quietly wrong, and waiting for a free
24 GB GPU to discover a bug in the LR schedule would be a poor trade.

What this deliberately does *not* cover: that LLaDA itself accepts our bias (E0.0c/E0.0e already
measured that) and real memory or throughput (needs the real backbone).
"""

from __future__ import annotations

import torch

from plexus.data.examples import build_nc_examples, collate
from plexus.masks import HopBias
from plexus.metrics import Result, aggregate, overlaps
from plexus.models.llada import answer_logits, answer_loss
from plexus.train import TrainConfig, evaluate, train
from tests.test_data import toy_tag
from tests.test_prompts import FakeTokenizer

VOCAB = 64
SEQ = 96


class StubOutput:
    def __init__(self, logits):
        self.logits = logits


class StubBackbone(torch.nn.Module):
    """Trainable stand-in with LLaDA's calling convention.

    Reads the attention bias so that a mask family which gates everything is distinguishable from
    one that gates nothing -- otherwise a test could pass with the bias silently ignored, which is
    the exact failure mode E0.0c was written to catch in the real model.
    """

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(1_000_000, 16)
        self.head = torch.nn.Linear(16, VOCAB)
        self.bias_sensitivity = torch.nn.Parameter(torch.zeros(1))
        self.graded_sensitivity = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_bias=None):
        h = self.embed(input_ids)
        if attention_bias is not None:
            readable = (attention_bias[:, 0] > -1e8).float().mean(dim=-1, keepdim=True)
            h = h + self.bias_sensitivity * readable

            # A second, *differentiable* read of the bias values. The comparison above is a step
            # function, so it passes no gradient back to a learnable bias -- a stub that only read
            # the bias that way would report "no gradient reached the parameter" for a graded bias
            # that is in fact wired correctly. Gated entries are excluded through a detached
            # predicate so they contribute no gradient and, being exactly 0 at the kept positions
            # of every hard mask, leave the hard-mask tests numerically untouched.
            #
            # Reduced over *both* score axes, not just the last. Reducing only the last axis makes
            # each position read its own attention row, and the answer [M] token's row is the one
            # row a graded bias never writes to: the answer position belongs to no node span, so
            # `node_to_token_bias` leaves it unbiased by construction. A stub that read only that
            # row therefore reports an all-zero gradient for a correctly wired method. The real
            # backbone has 32 layers in which a bias anywhere reaches the answer token, so a
            # whole-matrix reduction is the faithful stand-in.
            b = attention_bias[:, 0]
            kept = (b > -1e8).detach()
            graded = torch.where(b > -1e8, b, torch.zeros_like(b))
            graded = (graded.to(h.dtype) * kept.to(h.dtype)).mean(dim=(-2, -1))
            h = h + self.graded_sensitivity * graded[:, None, None]
        return StubOutput(self.head(h))


def make_examples(n: int = 6, pad_to_multiple_of: int = 1):
    tok = FakeTokenizer()
    tag = toy_tag()
    centers = list(range(n))
    ex = build_nc_examples(tag, tok, centers, k=2, K=10, seed=0, seq_len=SEQ,
                           mask_token_id=5, max_target_tokens=8, max_neighbour_tokens=4,
                           pad_to_multiple_of=pad_to_multiple_of)
    return tag, tok, ex


def test_collate_pads_ragged_examples_to_the_batch_maximum():
    """Examples now differ in length, so collation must align them without corrupting spans."""
    tag, tok, ex = make_examples(6, pad_to_multiple_of=1)
    lengths = {int(e.input_ids.numel()) for e in ex}
    assert len(lengths) > 1, "toy examples should differ in length"

    batch = collate(ex, family="star", device="cpu", bias_dtype=torch.float32)
    seq = max(lengths)
    assert batch["input_ids"].shape == (len(ex), seq)
    assert batch["attention_bias"].shape == (len(ex), 1, seq, seq)

    # Every example's own padding tail must be dead in both directions, whatever its length.
    from plexus.masks import MASK_BIAS

    for i, e in enumerate(ex):
        pf = e.graph.pad_from
        assert (batch["attention_bias"][i, 0, pf:, :] <= MASK_BIAS / 2).all()
        assert (batch["attention_bias"][i, 0, :, pf:] <= MASK_BIAS / 2).all()


def test_ragged_collation_preserves_each_examples_answer_position():
    tag, tok, ex = make_examples(6, pad_to_multiple_of=1)
    batch = collate(ex, family="star", device="cpu", bias_dtype=torch.float32)
    for i, e in enumerate(ex):
        assert batch["answer_pos"][i].item() == e.answer_pos
        assert batch["input_ids"][i, e.answer_pos].item() == 5  # the [M] id used above


def base_cfg(**kw) -> TrainConfig:
    defaults = dict(
        task="nc", mask_family="star", seq_len=SEQ, epochs=1, micro_batch=1,
        effective_batch=2, device="cpu", bias_dtype="float32", log_every_steps=1000,
    )
    defaults.update(kw)
    return TrainConfig(**defaults)


def test_collate_shapes_and_target_tokens():
    tag, tok, ex = make_examples(4)
    batch = collate(ex[:2], family="star", device="cpu", bias_dtype=torch.float32)

    seq = batch["input_ids"].shape[1]
    assert batch["input_ids"].shape == (2, seq)
    assert batch["attention_bias"].shape == (2, 1, seq, seq)
    assert batch["labels"].shape == (2,)
    # The token the model must emit is the digit for the label, not the label index itself.
    for i, e in enumerate(ex[:2]):
        assert batch["target_token_ids"][i].item() == e.candidate_ids[e.label]


def test_train_runs_and_reduces_loss_on_a_memorisable_set():
    """A 6-example set with a 16-dim stub should be memorisable; loss must fall."""
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    cfg = base_cfg(epochs=40, lr=1e-2, log_every_steps=10)
    hist = train(model, ex, cfg)

    losses = [s["loss"] for s in hist["steps"]]
    assert len(losses) >= 2
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_gradient_accumulation_reaches_the_effective_batch():
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    cfg = base_cfg(epochs=1, effective_batch=3, micro_batch=1)
    assert cfg.accum_steps == 3
    hist = train(model, ex, cfg, out_dir=None)
    # 6 micro steps at accum 3 == 2 optimiser steps.
    assert hist["total_opt_steps"] == 2


def test_max_steps_stops_early():
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    hist = train(model, ex, base_cfg(epochs=10), max_steps=3)
    assert hist["wall_clock_s"] >= 0
    assert all(s["opt_step"] <= 2 for s in hist["steps"])


def test_lr_warms_up_then_decays():
    from plexus.train.harness import _lr_at

    cfg = base_cfg(lr=1.0, warmup_frac=0.1)
    total = 100
    assert _lr_at(0, total, cfg) == 0.0
    assert _lr_at(10, total, cfg) == 1.0          # end of warmup
    assert _lr_at(55, total, cfg) < 1.0           # decaying
    assert _lr_at(100, total, cfg) < 1e-6         # cosine floor


def test_first_optimiser_step_has_a_nonzero_lr():
    """A first step at lr=0 is a silently wasted update."""
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    hist = train(model, ex, base_cfg(epochs=2, lr=1e-2, log_every_steps=1), max_steps=4)
    assert hist["steps"][0]["lr"] > 0.0


def test_evaluate_reports_density_alongside_accuracy():
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    res = evaluate(model, ex, base_cfg(mask_family="star"))

    assert isinstance(res, Result)
    assert res.n == len(ex)
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 < res.mask_density <= 1.0, "density must be reported, not NaN"
    assert "density" in res.row() and "acc" in res.row()


def test_evaluate_distinguishes_mask_families():
    """`none` and `ego` must produce different densities through the real code path."""
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    d_none = evaluate(model, ex, base_cfg(mask_family="none")).mask_density
    d_ego = evaluate(model, ex, base_cfg(mask_family="ego")).mask_density
    assert d_none == 1.0
    assert d_ego < d_none


def test_aggregate_and_overlap_logic():
    mk = lambda acc, seed: Result(  # noqa: E731
        accuracy=acc, n=542, mask_family="star", mask_density=0.8, task="nc", seed=seed
    )
    agg = aggregate([mk(0.90, 0), mk(0.91, 1), mk(0.92, 2)])
    assert agg.n_seeds == 3
    assert abs(agg.mean - 0.91) < 1e-9
    assert agg.std > 0

    close = aggregate([mk(0.905, 0), mk(0.915, 1), mk(0.925, 2)])
    assert overlaps(agg, close), "nearly identical means must not be called separable"

    far = aggregate([
        Result(accuracy=a, n=542, mask_family="ego", mask_density=0.8, task="nc", seed=s)
        for s, a in enumerate([0.60, 0.61, 0.62])
    ])
    assert not overlaps(agg, far), "a 30-point gap must be separable"


def test_single_seed_is_never_called_separable():
    """§4's standing rule: one seed cannot support a claim about a difference."""
    a = aggregate([Result(accuracy=0.90, n=542, mask_family="star",
                          mask_density=0.8, task="nc", seed=0)])
    b = aggregate([Result(accuracy=0.60, n=542, mask_family="ego",
                          mask_density=0.8, task="nc", seed=0)])
    assert overlaps(a, b)


def test_learnable_hop_bias_receives_gradient():
    """C3's method must actually train, not merely be representable.

    The graded bias is the proposed contribution, and its whole appeal is that the optimiser can
    *learn* how much topology should matter. If gradients do not reach `HopBias.values` the run
    still completes and still reports an accuracy -- it would just be the unmodified model with a
    constant offset, and we would have reported a null result as a method.

    The path is easy to break: the bias is assembled on CPU by `build_attention_bias` via
    in-place index assignment into a `torch.zeros`, cast to bf16, then moved to the device before
    reaching the attention scores.
    """
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    hb = HopBias(values=torch.nn.Parameter(torch.zeros(4)))

    batch = collate(ex[:1], family="none", hop_bias=hb, device="cpu", bias_dtype=torch.float32)
    assert batch["attention_bias"].requires_grad, "bias is detached from the parameter"

    logits = answer_logits(model, batch["input_ids"], batch["attention_bias"],
                           batch["answer_pos"])
    answer_loss(logits, batch["target_token_ids"]).backward()

    assert hb.values.grad is not None, "no gradient reached HopBias.values"
    assert hb.values.grad.abs().sum() > 0, "gradient reached the parameter but is all zero"


def test_hop_bias_gradient_survives_bf16_bias_construction():
    """The training default is a bf16 bias; the cast must not sever the graph."""
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    hb = HopBias(values=torch.nn.Parameter(torch.zeros(4)))

    batch = collate(ex[:1], family="none", hop_bias=hb, device="cpu",
                    bias_dtype=torch.bfloat16)
    logits = answer_logits(model, batch["input_ids"], batch["attention_bias"],
                           batch["answer_pos"])
    answer_loss(logits, batch["target_token_ids"]).backward()
    assert hb.values.grad is not None and hb.values.grad.abs().sum() > 0


def test_train_optimises_the_hop_bias_when_enabled():
    """End-to-end through the harness: the bias values must move away from their init."""
    tag, tok, ex = make_examples(6)
    model = StubBackbone()
    cfg = base_cfg(epochs=6, lr=1e-1, hop_bias_buckets=4, effective_batch=2,
                   log_every_steps=1000)

    # `_make_hop_bias` is called inside `train`, so capture the parameter it creates. Checking
    # `.grad` afterwards would prove nothing either way -- the harness zeroes with
    # `set_to_none=True`, so `.grad` is None after every optimiser step whether or not the
    # parameter was ever optimised. Displacement from the init is the assertion that bites.
    import plexus.train.harness as harness

    original = harness._make_hop_bias
    created = {}

    def spy(c, device):
        hb = original(c, device)
        created["hb"] = hb
        return hb

    harness._make_hop_bias = spy
    try:
        hist = train(model, ex, cfg)
    finally:
        harness._make_hop_bias = original

    hb = created.get("hb")
    assert hb is not None, "hop bias was never created"
    assert not torch.allclose(hb.values.detach(), torch.zeros(4)), (
        "hop bias never moved from its zero init, so it was not being optimised"
    )
    assert hist["config"]["hop_bias_buckets"] == 4


def test_answer_as_target_grades_the_answer_row_that_is_otherwise_untouched():
    """The default leaves the answer row unbiased; the option is what fills it in."""
    tag, tok, ex = make_examples(4)
    e = ex[0]
    pos = e.answer_pos
    values = torch.nn.Parameter(torch.tensor([0.5, -0.25, 0.75, -1.0]))

    plain = collate([e], family="none", hop_bias=HopBias(values=values),
                    device="cpu", bias_dtype=torch.float32)["attention_bias"][0, 0]
    graded = collate([e], family="none",
                     hop_bias=HopBias(values=values, answer_as_target=True),
                     device="cpu", bias_dtype=torch.float32)["attention_bias"][0, 0]

    live = e.graph.pad_from or plain.shape[0]
    assert torch.allclose(plain[pos, :live], torch.zeros(live)), (
        "answer row was expected to be unbiased by default"
    )
    assert not torch.allclose(graded[pos, :live], torch.zeros(live)), (
        "answer_as_target did not reach the answer row"
    )

    # Node-to-node entries must be identical; only the answer's own row and column may change.
    owner = e.graph.span_of_token()
    node_tok = (owner >= 0).nonzero().flatten()
    assert torch.allclose(plain[node_tok][:, node_tok], graded[node_tok][:, node_tok]), (
        "answer_as_target perturbed node-to-node bias, so the arm is no longer comparable"
    )


def test_answer_as_target_reads_the_target_bucket_for_the_target_span():
    """Concretely: the answer sees the target's tokens at the hop-0 value."""
    tag, tok, ex = make_examples(4)
    e = ex[0]
    values = torch.nn.Parameter(torch.tensor([0.5, -0.25, 0.75, -1.0]))
    bias = collate([e], family="none",
                   hop_bias=HopBias(values=values, answer_as_target=True),
                   device="cpu", bias_dtype=torch.float32)["attention_bias"][0, 0]

    lo, hi = e.graph.spans[0]
    assert torch.allclose(bias[e.answer_pos, lo:hi], torch.full((hi - lo,), 0.5))


def test_answer_as_target_never_alters_a_hard_mask():
    """The option touches the additive term only, so `star` and `ego` are untouched."""
    tag, tok, ex = make_examples(4)
    for family in ("star", "ego", "ego_star"):
        a = collate(ex[:1], family=family, device="cpu",
                    bias_dtype=torch.float32)["attention_bias"]
        b = collate(ex[:1], family=family,
                    hop_bias=HopBias(values=torch.zeros(4), answer_as_target=True),
                    device="cpu", bias_dtype=torch.float32)["attention_bias"]
        assert torch.equal(a, b), f"{family} changed under answer_as_target"


def test_zero_initialised_hop_bias_starts_as_the_unmodified_model():
    """The method's safety property: at init it is exactly the no-structure baseline."""
    tag, tok, ex = make_examples(4)
    hb = HopBias(values=torch.nn.Parameter(torch.zeros(4)))
    with_bias = collate(ex[:1], family="none", hop_bias=hb, device="cpu",
                        bias_dtype=torch.float32)["attention_bias"]
    without = collate(ex[:1], family="none", device="cpu",
                      bias_dtype=torch.float32)["attention_bias"]
    assert torch.equal(with_bias.detach(), without)


def test_history_records_config_and_provenance():
    tag, tok, ex = make_examples(4)
    model = StubBackbone()
    cfg = base_cfg(provenance={"split": "test-provenance"})
    hist = train(model, ex, cfg, max_steps=2)
    assert hist["config"]["provenance"]["split"] == "test-provenance"
    assert hist["config"]["mask_family"] == "star"
    assert hist["n_train_examples"] == 4
