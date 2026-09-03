"""Tests for linearisation and span bookkeeping.

Spans are the interface between the prompt and every mask family: if `spans[i]` does not cover
exactly node *i*'s tokens, the masks gate the wrong positions and every downstream accuracy
number is meaningless while still looking plausible. So these tests decode spans back to text and
check the content, rather than only checking shapes.

Structural tests use a fake word-level tokenizer so they run without the 16 GB checkpoint. The
tests that must see LLaDA's real vocabulary -- single-token digits, the `[M]` id -- skip unless it
is present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from plexus.data.sampling import sample_ego_graph
from plexus.masks import MASK_BIAS, build_attention_bias
from plexus.prompts.templates import build_lp_example, build_nc_example, candidate_token_ids
from tests.test_data import toy_tag

MASK_ID = 126336


class FakeTokenizer:
    """Word-level tokenizer: one id per whitespace-separated word, digits reserved at 0..9."""

    pad_token_id = 999_999

    def __init__(self):
        self.vocab: dict[str, int] = {str(i): i for i in range(10)}
        self.inv: dict[int, str] = {i: str(i) for i in range(10)}
        self._next = 10

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = self._next
                self.inv[self._next] = w
                self._next += 1
            ids.append(self.vocab[w])
        return {"input_ids": ids}

    def decode(self, ids) -> str:
        vals = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return " ".join(self.inv.get(int(i), "<pad>") for i in vals)


@pytest.fixture
def tok():
    return FakeTokenizer()


def test_nc_spans_decode_to_their_own_node(tok):
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    ex = build_nc_example(tag, s, tok, seq_len=512, mask_token_id=MASK_ID)

    assert len(ex.graph.spans) == s.n_nodes
    for i, (a, b) in enumerate(ex.graph.spans):
        decoded = tok.decode(ex.input_ids[a:b])
        assert f"{i}]" in decoded, f"span {i} decoded to {decoded!r}"
        # The node's own paper text must be inside its span.
        assert f"paper {tag.text[s.node_ids[i]].split()[-1]}" in decoded


def test_spans_are_disjoint_and_ordered(tok):
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    ex = build_nc_example(tag, s, tok, seq_len=512, mask_token_id=MASK_ID)
    prev_end = 0
    for a, b in ex.graph.spans:
        assert a >= prev_end, "spans overlap"
        assert b > a, "empty span"
        prev_end = b


def test_answer_position_holds_the_mask_token_and_is_outside_every_span(tok):
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    ex = build_nc_example(tag, s, tok, seq_len=512, mask_token_id=MASK_ID)

    assert ex.input_ids[ex.answer_pos].item() == MASK_ID
    assert ex.graph.answer_pos == ex.answer_pos
    for a, b in ex.graph.spans:
        assert not (a <= ex.answer_pos < b), "the answer must not live inside a node span"


def test_padding_starts_after_the_answer(tok):
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    ex = build_nc_example(tag, s, tok, seq_len=512, mask_token_id=MASK_ID)

    assert ex.graph.pad_from == ex.answer_pos + 1
    assert (ex.input_ids[ex.graph.pad_from:] == tok.pad_token_id).all()


def test_label_is_the_targets_label(tok):
    tag = toy_tag()
    for center in range(5):
        s = sample_ego_graph(tag.build_adjacency(), center, k=2, K=10, seed=0)
        ex = build_nc_example(tag, s, tok, seq_len=512, mask_token_id=MASK_ID)
        assert ex.label == int(tag.labels[center])


def test_masks_apply_to_real_spans(tok):
    """End-to-end: a real linearised example must produce the mask semantics the plan relies on."""
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    ex = build_nc_example(tag, s, tok, seq_len=512, mask_token_id=MASK_ID)
    g = ex.graph

    star = build_attention_bias([g], family="star")[0, 0]
    ego = build_attention_bias([g], family="ego")[0, 0]

    # Pick a hop-2 node and a token inside the target span.
    hop2 = [i for i, h in enumerate(g.hops) if h == 2]
    assert hop2, "toy graph should yield hop-2 nodes"
    t_start, _ = g.spans[0]
    h_start, h_end = g.spans[hop2[0]]

    assert (star[t_start, h_start:h_end] > MASK_BIAS / 2).all(), "star: target reads hop-2"
    assert (ego[t_start, h_start:h_end] <= MASK_BIAS / 2).all(), "ego: target cannot read hop-2"


def _nc_reserved(tok, tag) -> int:
    """Tokens the NC scaffolding costs, so budget tests need no hardcoded magic numbers."""
    from plexus.prompts.templates import NC_INSTRUCTION, _encode

    options = "\n".join(f"{i}) {n}" for i, n in enumerate(tag.class_prompt_names))
    prefix = f"{NC_INSTRUCTION}Categories:\n{options}\n"
    return len(_encode(tok, prefix)) + len(_encode(tok, "\nThe target paper's category is: ")) + 1


def test_truncation_drops_highest_hop_first_and_reports_it(tok):
    """A tiny budget must drop distant context, keep the target, and say what it did."""
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    # Room for the scaffolding, the target, and exactly one neighbour.
    seq_len = _nc_reserved(tok, tag) + 8 + 4
    ex = build_nc_example(tag, s, tok, seq_len=seq_len, mask_token_id=MASK_ID,
                          max_target_tokens=8, max_neighbour_tokens=4)
    assert ex.meta["n_dropped"] > 0
    assert len(ex.graph.spans) == ex.meta["n_nodes"] < s.n_nodes
    # The target survives, and hops/edges were restricted to the kept nodes.
    assert ex.graph.hops[0] == 0
    assert all(u < ex.meta["n_nodes"] and v < ex.meta["n_nodes"] for u, v in ex.graph.edges)
    # Kept nodes must be a *prefix* of the sample, or spans point at the wrong nodes.
    assert ex.graph.node_ids == s.node_ids[: ex.meta["n_nodes"]]
    assert ex.graph.hops == s.hops[: ex.meta["n_nodes"]]


def test_dropping_the_target_raises_rather_than_corrupting(tok):
    """An example without its target is corrupt, not merely harder."""
    tag = toy_tag()
    s = sample_ego_graph(tag.build_adjacency(), 0, k=2, K=10, seed=0)
    # One token short of fitting the target block on top of the scaffolding.
    seq_len = _nc_reserved(tok, tag) + 2
    with pytest.raises(ValueError, match="required block 0"):
        build_nc_example(tag, s, tok, seq_len=seq_len, mask_token_id=MASK_ID,
                         max_target_tokens=64, max_neighbour_tokens=4)


def test_lp_requires_both_endpoints_to_fit(tok):
    """Link prediction has two required blocks; failing the second must also raise."""
    from plexus.prompts.templates import LP_INSTRUCTION, LP_OPTIONS, _encode, _node_block

    tag = toy_tag()
    adj = tag.build_adjacency()
    s = sample_ego_graph(adj, [0, 3], k=2, K=10, seed=0, exclude_edges={(0, 3)})

    options = "\n".join(f"{i}) {n}" for i, n in enumerate(LP_OPTIONS))
    prefix = f"{LP_INSTRUCTION}Options:\n{options}\n"
    reserved = (len(_encode(tok, prefix))
                + len(_encode(tok, "\nDo the two target papers cite each other? ")) + 1)
    first = len(_encode(tok, _node_block(0, tag.text[0], "First target paper"), 16))
    # Fits the first endpoint but not the second.
    with pytest.raises(ValueError, match="required block 1"):
        build_lp_example(tag, s, tok, 1, seq_len=reserved + first + 1, mask_token_id=MASK_ID,
                         max_target_tokens=16, max_neighbour_tokens=4)


def test_lp_refuses_to_leak_the_queried_edge(tok):
    tag = toy_tag()
    adj = tag.build_adjacency()
    leaky = sample_ego_graph(adj, [0, 1], k=2, K=10, seed=0)  # no exclude_edges
    with pytest.raises(ValueError, match="reveal the label"):
        build_lp_example(tag, leaky, tok, 1, seq_len=512, mask_token_id=MASK_ID)


def test_lp_puts_both_endpoints_first(tok):
    tag = toy_tag()
    adj = tag.build_adjacency()
    s = sample_ego_graph(adj, [0, 3], k=2, K=10, seed=0, exclude_edges={(0, 3)})
    ex = build_lp_example(tag, s, tok, 0, seq_len=512, mask_token_id=MASK_ID)
    assert ex.graph.hops[0] == 0 and ex.graph.hops[1] == 0
    assert ex.meta["pair"] == (0, 3)
    assert ex.label == 0


# ------------------------------------------------------------------- real LLaDA vocabulary

HF_HOME = os.environ.get("HF_HOME", "")
_snap = Path(HF_HOME) / "hub" / "models--GSAI-ML--LLaDA-8B-Instruct"
needs_llada = pytest.mark.skipif(not _snap.exists(), reason="LLaDA tokenizer not present")


@needs_llada
def test_option_digits_are_single_tokens_in_llada():
    """The single-position scoring rule is only well defined if each option is one token."""
    from transformers import AutoTokenizer

    real = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    ids = candidate_token_ids(real, 7)
    assert len(set(ids)) == 7
    for i, tid in enumerate(ids):
        assert real.decode([tid]).strip() == str(i)


@needs_llada
def test_real_cora_example_fits_the_sequence_budget():
    from transformers import AutoTokenizer

    from plexus.data.tags import load_cora
    from tests.test_data import CORA_ROOT

    if not (Path(CORA_ROOT) / "cora_orig" / "cora.content").exists():
        pytest.skip("Cora not present")

    real = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    tag = load_cora(CORA_ROOT, seed=0)
    adj = tag.build_adjacency()
    for center in tag.train_idx[:20].tolist():
        s = sample_ego_graph(adj, center, k=2, K=10, seed=0)
        ex = build_nc_example(tag, s, real, seq_len=2048)
        assert ex.meta["n_tokens"] <= 2048
        assert ex.meta["n_dropped"] == 0, "K=10,k=2 should fit in 2048 without dropping"
        assert ex.input_ids[ex.answer_pos].item() == 126336
