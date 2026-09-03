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
from plexus.metrics import Result, aggregate, overlaps
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

    def forward(self, input_ids, attention_bias=None):
        h = self.embed(input_ids)
        if attention_bias is not None:
            readable = (attention_bias[:, 0] > -1e8).float().mean(dim=-1, keepdim=True)
            h = h + self.bias_sensitivity * readable
        return StubOutput(self.head(h))


def make_examples(n: int = 6):
    tok = FakeTokenizer()
    tag = toy_tag()
    centers = list(range(n))
    ex = build_nc_examples(tag, tok, centers, k=2, K=10, seed=0, seq_len=SEQ,
                           mask_token_id=5, max_target_tokens=8, max_neighbour_tokens=4)
    return tag, tok, ex


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

    assert batch["input_ids"].shape == (2, SEQ)
    assert batch["attention_bias"].shape == (2, 1, SEQ, SEQ)
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


def test_history_records_config_and_provenance():
    tag, tok, ex = make_examples(4)
    model = StubBackbone()
    cfg = base_cfg(provenance={"split": "test-provenance"})
    hist = train(model, ex, cfg, max_steps=2)
    assert hist["config"]["provenance"]["split"] == "test-provenance"
    assert hist["config"]["mask_family"] == "star"
    assert hist["n_train_examples"] == 4
