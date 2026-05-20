"""Tests for IndexLayer and IndexGroups."""

import torch

from tb.indices import IndexGroups, IndexLayer, masked_softmax


def test_index_layer_weight_tying() -> None:
    """Forward gives A^T q logits; embed(k) returns A[:, k]."""
    layer = IndexLayer(num_indices=5, dim=3, use_bias=False)
    q = torch.zeros(2, 3)
    logits = layer(q)
    assert logits.shape == (2, 5)
    # With zero q and no bias, logits are zero.
    assert torch.allclose(logits, torch.zeros_like(logits))

    # Embed lookup is just row indexing.
    k = torch.tensor([2, 4])
    emb = layer.embed(k)
    assert emb.shape == (2, 3)
    assert torch.allclose(emb, layer.weight[k])


def test_index_layer_bias() -> None:
    layer = IndexLayer(num_indices=4, dim=2, use_bias=True)
    with torch.no_grad():
        layer.bias.fill_(7.0)
    logits = layer(torch.zeros(1, 2))
    assert torch.allclose(logits, torch.full((1, 4), 7.0))


def test_index_groups_flat() -> None:
    g = IndexGroups.flat(10)
    assert g.num_indices == 10
    assert g.names == ("all",)
    assert g.slice_of("all") == slice(0, 10)
    mask = g.mask("all")
    assert mask.all()


def test_index_groups_named() -> None:
    g = IndexGroups.from_sizes([("entity", 5), ("predicate", 3), ("instance", 2)])
    assert g.num_indices == 10
    assert g.offsets == (0, 5, 8)
    assert g.slice_of("entity") == slice(0, 5)
    assert g.slice_of("predicate") == slice(5, 8)
    assert g.slice_of("instance") == slice(8, 10)
    m = g.mask("predicate")
    assert m.sum() == 3
    assert m[5] and m[6] and m[7]
    assert not m[4] and not m[8]


def test_masked_softmax() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([True, True, False, False])
    p = masked_softmax(logits, mask)
    # Probability mass only on first two indices.
    assert torch.allclose(p[0, 2:], torch.zeros(2))
    assert torch.isclose(p.sum(), torch.tensor(1.0))


def test_masked_softmax_none() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    p = masked_softmax(logits, None)
    assert torch.isclose(p.sum(), torch.tensor(1.0))
    assert (p > 0).all()
