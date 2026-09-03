"""Turning a TAG into batches of linearised examples.

Biases are built **lazily, per batch**. A `(1, 1, 2048, 2048)` fp32 bias is 16.8 MB, so
precomputing them for Cora's 1624 training examples would cost 27 GB -- more than the GPU. The
examples themselves (token ids and spans) are cheap and are built once up front.
"""

from __future__ import annotations

import torch

from plexus.data.sampling import sample_ego_graph, sample_lp_examples
from plexus.data.tags import TAG
from plexus.masks import HopBias, build_attention_bias
from plexus.prompts.templates import LinearisedExample, build_lp_example, build_nc_example


def build_nc_examples(
    tag: TAG,
    tokenizer,
    centers: list[int],
    *,
    k: int = 2,
    K: int = 10,
    seed: int = 0,
    seq_len: int = 2048,
    random_node_control: bool = False,
    **prompt_kw,
) -> list[LinearisedExample]:
    """Node-classification examples, one per centre.

    `random_node_control` implements E0.3: keep the prompt format and the star mask, but replace
    the sampled graph neighbours with uniformly random nodes. If accuracy survives that, the
    reported gain is prompt context rather than topology, and C1 is supported.
    """
    adj = tag.build_adjacency()
    out = []
    for center in centers:
        sample = sample_ego_graph(adj, int(center), k=k, K=K, seed=seed)
        if random_node_control:
            sample = _randomise_neighbours(tag, sample, seed=seed)
        out.append(build_nc_example(tag, sample, tokenizer, seq_len=seq_len, **prompt_kw))
    return out


def _randomise_neighbours(tag: TAG, sample, *, seed: int):
    """Replace every non-target node with a uniformly random node, keeping count and wiring."""
    from plexus.data.sampling import EgoSample, _derive_generator

    gen = _derive_generator(seed, sample.node_ids[0], 0xE03)
    n_replace = sample.n_nodes - 1
    pool = torch.randperm(tag.num_nodes, generator=gen)[: n_replace + 1].tolist()
    pool = [p for p in pool if p != sample.node_ids[0]][:n_replace]
    return EgoSample(
        node_ids=[sample.node_ids[0]] + pool,
        hops=list(sample.hops),
        edges=set(sample.edges),
    )


def build_lp_example_set(
    tag: TAG,
    tokenizer,
    node_idx: torch.Tensor,
    *,
    n_examples: int | None = None,
    k: int = 2,
    K: int = 10,
    seed: int = 0,
    seq_len: int = 2048,
    **prompt_kw,
) -> list[LinearisedExample]:
    """Link-prediction examples, with the queried edge removed from every sample."""
    adj = tag.build_adjacency()
    pairs = sample_lp_examples(tag, node_idx, n_examples=n_examples, seed=seed)
    out = []
    for u, v, label in pairs:
        sample = sample_ego_graph(
            adj, [u, v], k=k, K=K, seed=seed, exclude_edges={(u, v)}
        )
        out.append(build_lp_example(tag, sample, tokenizer, label, seq_len=seq_len, **prompt_kw))
    return out


def collate(
    examples: list[LinearisedExample],
    *,
    family: str = "star",
    hop_bias: HopBias | None = None,
    device: str = "cuda:0",
    bias_dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> dict:
    """Stack examples and build their attention bias.

    `bias_dtype` defaults to bf16: it halves the 16.8 MB-per-sample cost and matches the compute
    dtype, at the cost of coarsening graded biases. `HopBias` values are O(1), so bf16's ~3 decimal
    digits are enough; revisit if E1 needs finer resolution.
    """
    input_ids = torch.stack([e.input_ids for e in examples]).to(device)
    labels = torch.tensor([e.label for e in examples], dtype=torch.long, device=device)
    answer_pos = torch.tensor([e.answer_pos for e in examples], dtype=torch.long, device=device)
    candidate_ids = torch.tensor(examples[0].candidate_ids, dtype=torch.long, device=device)

    bias = build_attention_bias(
        [e.graph for e in examples],
        family=family,
        hop_bias=hop_bias,
        dtype=bias_dtype,
        device=device,
        generator=generator,
    )
    # The label is a class index; the token the model must produce is the corresponding digit.
    target_token_ids = candidate_ids[labels]
    return {
        "input_ids": input_ids,
        "attention_bias": bias,
        "answer_pos": answer_pos,
        "labels": labels,
        "candidate_ids": candidate_ids,
        "target_token_ids": target_token_ids,
    }
