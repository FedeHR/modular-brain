"""Tests for the decode_triple composer."""

import pytest
import torch

from tb.composers import decode_triple
from tb.evolve import QTBEvolve, TBEvolve
from tb.indices import IndexGroups, IndexLayer
from tb.primitives import TB


def make_tb(num_indices: int = 12, *, persistent: bool = False) -> TB:
    dim = 8
    evolve = TBEvolve(dim=dim, hidden=4) if persistent else QTBEvolve(dim=dim, hidden=4)
    return TB(
        dim=dim,
        index_layer=IndexLayer(num_indices=num_indices, dim=dim),
        evolve_module=evolve,
    )


def test_decode_triple_shapes() -> None:
    tb = make_tb()
    out = decode_triple(tb, commit="argmax", batch_size=4, device=torch.device("cpu"))
    assert out.subject.k.shape == (4,)
    assert out.object.k.shape == (4,)
    assert out.predicate.k.shape == (4,)
    assert out.subject.logits.shape == (4, 12)


def test_decode_triple_requires_init_or_batch() -> None:
    tb = make_tb()
    with pytest.raises(ValueError, match="initial_q"):
        decode_triple(tb, commit="argmax")


def test_decode_triple_initial_q_passthrough() -> None:
    tb = make_tb()
    q0 = torch.randn(3, 8)
    out = decode_triple(tb, commit="argmax", initial_q=q0)
    assert out.subject.k.shape == (3,)


def test_decode_triple_persistent_state() -> None:
    """With TBEvolve, decoding three positions in a row uses persistent h."""
    tb = make_tb(persistent=True)
    out = decode_triple(tb, commit="argmax", batch_size=2, device=torch.device("cpu"))
    # Just check it runs and produces valid output.
    assert out.predicate.k.shape == (2,)


def test_decode_triple_masks_restrict_per_position() -> None:
    """Subject/object → entities, predicate → predicates."""
    groups = IndexGroups.from_sizes([("ent", 6), ("pred", 4)])
    tb = make_tb(num_indices=groups.num_indices)
    ent_mask = groups.mask("ent")
    pred_mask = groups.mask("pred")

    out = decode_triple(
        tb,
        subject_mask=ent_mask,
        object_mask=ent_mask,
        predicate_mask=pred_mask,
        commit="argmax",
        batch_size=20,
        device=torch.device("cpu"),
    )
    assert (out.subject.k < 6).all()
    assert (out.object.k < 6).all()
    assert (out.predicate.k >= 6).all() and (out.predicate.k < 10).all()


def test_decode_triple_grad_flow() -> None:
    tb = make_tb()
    out = decode_triple(tb, commit="expectation", batch_size=2, device=torch.device("cpu"))
    loss = out.subject.logits.sum() + out.object.logits.sum() + out.predicate.logits.sum()
    loss.backward()
    # Some parameter should have a nonzero grad.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tb.parameters())
