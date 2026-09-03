"""Tests for the topology mask families.

These assert the *semantics* the experiments depend on -- especially that ``star`` and ``ego``
differ in target reach as well as in neighbour-neighbour edges, since the whole E0.2 argument
rests on that being true.
"""

import torch

from plexus.masks import (
    MASK_BIAS,
    TARGET,
    HopBias,
    LinearisedGraph,
    build_attention_bias,
    describe,
    mask_density,
    pairwise_hops,
)


def make_graph(seq_len=24, pad_from=None):
    """4 nodes: target 0, hop-1 neighbours 1 and 2, hop-2 node 3 hanging off node 1.

    Spans are 4 tokens each starting at 4, leaving 0..3 as an unassigned prompt prefix.
    """
    spans = [(4, 8), (8, 12), (12, 16), (16, 20)]
    edges = {(0, 1), (0, 2), (1, 3)}
    return LinearisedGraph(spans=spans, seq_len=seq_len, edges=edges, pad_from=pad_from or seq_len)


def test_hops_from_bfs():
    g = make_graph()
    assert g.hops == [0, 1, 1, 2]


def test_pairwise_hops_exact():
    g = make_graph()
    d = pairwise_hops(g)
    assert d[TARGET, 3] == 2
    assert d[1, 2] == 2  # 1 -> 0 -> 2
    assert d[2, 3] == 3  # 2 -> 0 -> 1 -> 3
    assert (torch.diagonal(d) == 0).all()


def test_star_target_reads_everything():
    g = make_graph()
    bias = build_attention_bias([g], family="star")[0, 0]
    # A target token may read every node span, including the hop-2 node.
    for s, e in g.spans:
        assert (bias[5, s:e] > MASK_BIAS / 2).all()


def test_star_blocks_neighbour_to_neighbour():
    g = make_graph()
    bias = build_attention_bias([g], family="star")[0, 0]
    # Token in node 1 must not read node 2.
    assert (bias[9, 12:16] <= MASK_BIAS / 2).all()
    # But may read the target and itself.
    assert (bias[9, 4:8] > MASK_BIAS / 2).all()
    assert (bias[9, 8:12] > MASK_BIAS / 2).all()


def test_ego_removes_target_reach_to_hop2():
    """The load-bearing asymmetry: under `ego` the target cannot see the hop-2 node at all.

    TAG-DLM attributes the ego mask's collapse to neighbour-neighbour edges. This test documents
    that the ego mask *also* silently removes 2-hop context from the target, which is the
    alternative explanation E0.2 is designed to separate.
    """
    g = make_graph()
    ego = build_attention_bias([g], family="ego")[0, 0]
    star = build_attention_bias([g], family="star")[0, 0]
    assert (ego[5, 16:20] <= MASK_BIAS / 2).all()
    assert (star[5, 16:20] > MASK_BIAS / 2).all()


def test_ego_star_restores_target_reach_and_keeps_edges():
    g = make_graph()
    bias = build_attention_bias([g], family="ego_star")[0, 0]
    assert (bias[5, 16:20] > MASK_BIAS / 2).all()   # target sees hop-2 again
    assert (bias[9, 16:20] > MASK_BIAS / 2).all()   # edge (1,3) is live
    assert (bias[9, 12:16] <= MASK_BIAS / 2).all()  # non-edge (1,2) still blocked


def test_density_ordering():
    g = make_graph()
    d = {f: mask_density(build_attention_bias([g], family=f)) for f in ("none", "star", "ego", "ego_star")}
    assert d["none"] == 1.0
    assert d["ego"] < d["ego_star"] <= d["none"]
    # ego_star is by construction a superset of star
    assert d["star"] <= d["ego_star"]


def test_padding_masked_both_directions():
    g = make_graph(seq_len=24, pad_from=20)
    bias = build_attention_bias([g], family="star")[0, 0]
    assert (bias[20:, :] <= MASK_BIAS / 2).all()
    assert (bias[:, 20:] <= MASK_BIAS / 2).all()


def test_unassigned_prompt_tokens_stay_visible():
    """Prompt/instruction tokens carry the task, not graph content, and must not be gated."""
    g = make_graph()
    bias = build_attention_bias([g], family="ego")[0, 0]
    assert (bias[9, 0:4] > MASK_BIAS / 2).all()
    assert (bias[0:4, 9] > MASK_BIAS / 2).all()


def test_hop_bias_is_graded_not_gating():
    g = make_graph()
    hb = HopBias(values=torch.tensor([0.0, -0.5, -1.0, -2.0]))
    bias = build_attention_bias([g], family="none", hop_bias=hb)[0, 0]
    # Nothing is hard-masked: every non-pad entry remains readable.
    assert mask_density(bias) == 1.0
    # But hop-2 content is downweighted relative to hop-1.
    assert bias[5, 17] < bias[5, 9] <= 0.0


def test_hop_bias_uses_exact_apsp_not_the_hop_difference_approximation():
    """Two distinct hop-1 neighbours are 2 apart, not 0.

    The `|h_i - h_j|` shortcut puts them in the *self* bucket, i.e. it asserts that two different
    neighbours of the target are the same node. E1.1 reads the learned per-distance values
    directly, so that error would not merely add noise -- it would move mass into the wrong
    bucket and change what the method appears to have learned.
    """
    g = make_graph()  # nodes 1 and 2 are both hop-1 and are not adjacent
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])

    exact = HopBias(values=values).node_bias(g)
    approx = HopBias(values=values, exact=False).node_bias(g)

    assert pairwise_hops(g)[1, 2] == 2
    assert exact[1, 2] == 30.0, "exact APSP must place them in the distance-2 bucket"
    assert approx[1, 2] == 10.0, "the approximation collapses them into the self bucket"
    # The two paths must therefore disagree, which is the reason the default changed.
    assert not torch.equal(exact, approx)


def test_hop_bias_exact_is_symmetric_and_self_is_bucket_zero():
    g = make_graph()
    b = HopBias(values=torch.tensor([1.0, 2.0, 3.0, 4.0])).node_bias(g)
    assert torch.equal(b, b.t())
    assert (torch.diagonal(b) == 1.0).all()


def test_hop_bias_unreachable_goes_to_the_catch_all_bucket():
    """An unreachable node must get the catch-all, not distance 0."""
    g = LinearisedGraph(spans=[(0, 2), (2, 4), (4, 6)], seq_len=8, edges={(0, 1)})
    # Node 2 is isolated, so it is unreachable from the target.
    assert g.hops[2] == -1
    b = HopBias(values=torch.tensor([0.0, -1.0, -2.0, -9.0])).node_bias(g)
    assert b[0, 2] == -9.0


def test_zero_hop_bias_is_exactly_the_no_structure_baseline():
    """The graded family must contain the baseline as an exact special case.

    This is the property that makes the learned version safe: if topology does not help, the
    optimiser can return to the unmodified model rather than being stuck with a harmful mask.
    """
    g = make_graph()
    hb = HopBias(values=torch.zeros(4))
    graded = build_attention_bias([g], family="none", hop_bias=hb)[0, 0]
    plain = build_attention_bias([g], family="none")[0, 0]
    assert torch.equal(graded, plain)


def test_random_rewire_matches_ego_star_density_exactly():
    """The control must be density-matched to `ego_star`, or it confounds the thing it controls."""
    g = make_graph()
    ego_star = mask_density(build_attention_bias([g], family="ego_star"))
    for seed in range(8):
        gen = torch.Generator().manual_seed(seed)
        bias = build_attention_bias([g], family="random_rewire", generator=gen)[0, 0]
        assert (bias[5, 16:20] > MASK_BIAS / 2).all()  # star reach kept
        assert abs(mask_density(bias) - ego_star) < 1e-6


def test_batched_shape_and_describe():
    gs = [make_graph(), make_graph()]
    bias = build_attention_bias(gs, family="star")
    assert bias.shape == (2, 1, 24, 24)
    stats = describe(bias)
    assert set(stats) == {
        "density", "n_live_tokens", "min_row_visibility", "mean_row_visibility", "graded_frac",
    }
    assert 0.0 < stats["density"] <= 1.0


def test_density_ignores_padding():
    """Padding must not dominate the density metric, or all families look identical.

    A short sequence in a long buffer is ~1% dense under every mask if padding is counted, which
    destroys exactly the comparison the metric exists to support.
    """
    short = make_graph(seq_len=24, pad_from=20)
    padded = LinearisedGraph(spans=short.spans, seq_len=512, edges=set(short.edges), pad_from=20)
    d_short = mask_density(build_attention_bias([short], family="star"))
    d_padded = mask_density(build_attention_bias([padded], family="star"))
    assert abs(d_short - d_padded) < 1e-6
    assert mask_density(build_attention_bias([padded], family="star"), exclude_padding=False) < 0.02


def test_mask_bias_sums_without_overflow():
    """Two MASK_BIAS values must sum to something finite -- LLaDA adds the padding mask itself."""
    assert torch.isfinite(torch.tensor(MASK_BIAS + MASK_BIAS))
    assert torch.softmax(torch.tensor([0.0, MASK_BIAS + MASK_BIAS]), -1)[1] == 0.0
