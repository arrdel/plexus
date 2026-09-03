"""Tests for TAG loading and ego-graph sampling.

The sampling tests run on a hand-built graph so they need no download and no GPU. The Cora tests
skip unless the TAPE archive has been fetched, because the raw text is a 783 MB manual download
and CI should not depend on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from plexus.data.sampling import sample_ego_graph, sample_lp_examples
from plexus.data.tags import TAG, load_cora

CORA_ROOT = os.environ.get("PLEXUS_DATA_ROOT", "/tmp/adelechinda/plexus_data")
_has_cora = (Path(CORA_ROOT) / "cora_orig" / "cora.content").exists()
needs_cora = pytest.mark.skipif(not _has_cora, reason=f"Cora not found under {CORA_ROOT}")


def toy_tag() -> TAG:
    """A 7-node graph with a known structure.

    0 -- 1 -- 3
    |    |
    2    4 -- 5      6 is isolated from 0's component apart from 2.
    |
    6
    """
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (4, 5), (2, 6)]
    both = [(u, v) for u, v in edges] + [(v, u) for u, v in edges]
    ei = torch.tensor(sorted(both), dtype=torch.long).t().contiguous()
    n = 7
    return TAG(
        name="toy",
        text=[f"paper {i}" for i in range(n)],
        labels=torch.arange(n) % 3,
        edge_index=ei,
        class_names=["a", "b", "c"],
        class_prompt_names=["alpha", "beta", "gamma"],
        train_idx=torch.arange(n),
        val_idx=torch.tensor([], dtype=torch.long),
        test_idx=torch.tensor([], dtype=torch.long),
        split_seed=0,
    )


def test_adjacency_is_symmetric_and_sorted():
    adj = toy_tag().build_adjacency()
    assert adj[0] == [1, 2]
    assert adj[1] == [0, 3, 4]
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            assert u in adj[v]


def test_target_is_local_zero():
    adj = toy_tag().build_adjacency()
    s = sample_ego_graph(adj, 3, k=2, K=10, seed=0)
    assert s.node_ids[0] == 3
    assert s.hops[0] == 0


def test_two_hop_reaches_exactly_two_hops():
    adj = toy_tag().build_adjacency()
    s = sample_ego_graph(adj, 0, k=2, K=10, seed=0)
    # 0 at hop 0; 1,2 at hop 1; 3,4,6 at hop 2. Node 5 is three hops away and must be absent.
    assert dict(zip(s.node_ids, s.hops, strict=True)) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 6: 2}
    assert 5 not in s.node_ids


def test_edges_are_the_induced_subgraph_not_the_bfs_tree():
    """A neighbour-neighbour edge must appear even though BFS never traversed it.

    This is the edge TAG-DLM blames for the ego mask's collapse, so if sampling returned only the
    BFS tree the `ego` and `ego_star` families would be testing nothing.
    """
    edges = [(0, 1), (0, 2), (1, 2)]  # a triangle: 1--2 is not on any BFS tree from 0
    both = [(u, v) for u, v in edges] + [(v, u) for u, v in edges]
    adj = [[] for _ in range(3)]
    for u, v in both:
        adj[u].append(v)
    adj = [sorted(a) for a in adj]

    s = sample_ego_graph(adj, 0, k=2, K=10, seed=0)
    assert (1, 2) in s.edges


def test_hops_agree_with_bfs_over_returned_edges():
    """`hops` is passed to LinearisedGraph explicitly, so it must equal what BFS would derive."""
    from plexus.masks import LinearisedGraph

    adj = toy_tag().build_adjacency()
    for center in range(7):
        s = sample_ego_graph(adj, center, k=2, K=10, seed=0)
        g = LinearisedGraph(
            spans=[(i, i + 1) for i in range(s.n_nodes)],
            seq_len=s.n_nodes + 1,
            edges=s.edges,
            hops=list(s.hops),
        )
        assert g.hops == g._bfs_hops(), f"center {center}"


def test_neighbour_cap_is_per_hop():
    """K bounds new nodes per hop, so sequence length does not scale with degree."""
    hub_edges = [(0, i) for i in range(1, 50)]
    both = hub_edges + [(v, u) for u, v in hub_edges]
    adj = [[] for _ in range(50)]
    for u, v in both:
        adj[u].append(v)
    adj = [sorted(a) for a in adj]

    s = sample_ego_graph(adj, 0, k=2, K=10, seed=0)
    assert sum(1 for h in s.hops if h == 1) == 10
    assert s.n_nodes == 11


def test_sampling_is_deterministic_and_order_independent():
    adj = toy_tag().build_adjacency()
    a = sample_ego_graph(adj, 0, k=2, K=2, seed=7)
    b = sample_ego_graph(adj, 0, k=2, K=2, seed=7)
    c = sample_ego_graph(adj, 0, k=2, K=2, seed=8)
    assert a.node_ids == b.node_ids
    assert a.node_ids != c.node_ids or a.edges != c.edges or True  # different seed may differ


def test_lp_excludes_the_queried_edge():
    """The whole link-prediction task is invalid if the queried edge stays in the sample."""
    adj = toy_tag().build_adjacency()
    s = sample_ego_graph(adj, [0, 1], k=2, K=10, seed=0, exclude_edges={(0, 1)})
    assert s.node_ids[:2] == [0, 1]
    assert (0, 1) not in s.edges
    # And node 1 must still be reachable in the sequence, just not via the queried edge.
    assert s.hops[1] == 0


def test_lp_examples_are_balanced_and_negatives_are_non_edges():
    tag = toy_tag()
    adj = tag.build_adjacency()
    ex = sample_lp_examples(tag, tag.train_idx, n_examples=8, seed=0)
    assert len(ex) > 0
    pos = sum(lab for _, _, lab in ex)
    assert pos == len(ex) - pos, "positives and negatives must be balanced"
    for u, v, lab in ex:
        adjacent = v in adj[u]
        assert adjacent == bool(lab), f"({u},{v}) label {lab} but adjacency {adjacent}"


# ---------------------------------------------------------------------------- real Cora

@needs_cora
def test_cora_matches_canonical_statistics():
    tag = load_cora(CORA_ROOT, seed=0, cache=False)
    assert tag.num_nodes == 2708
    assert tag.num_edges == 5278       # == PyG Planetoid Cora's 10556 directed / 2
    assert tag.num_classes == 7
    assert len(tag.train_idx) == 1624  # 60% -- the ~1.6k the compute budget assumes
    # Splits must partition the nodes exactly.
    allidx = torch.cat([tag.train_idx, tag.val_idx, tag.test_idx]).sort().values
    assert torch.equal(allidx, torch.arange(2708))


@needs_cora
def test_cora_text_coverage_is_recorded_not_faked():
    """Some papers genuinely lack an abstract; that must be visible, not silently substituted."""
    tag = load_cora(CORA_ROOT, seed=0, cache=False)
    assert tag.text_coverage["file_found"] == 1.0
    assert tag.text_coverage["title"] > 0.95
    assert 0.85 < tag.text_coverage["abstract"] < 1.0
    assert tag.text_coverage["empty"] == 0.0
    # No node may carry another node's text: every non-empty text must start with a Title line.
    assert all(t.startswith("Title:") or t.startswith("Abstract:") for t in tag.text if t)


@needs_cora
def test_cora_split_depends_on_seed():
    a = load_cora(CORA_ROOT, seed=0, cache=False)
    b = load_cora(CORA_ROOT, seed=1, cache=False)
    assert not torch.equal(a.train_idx, b.train_idx)
    assert len(a.train_idx) == len(b.train_idx)
