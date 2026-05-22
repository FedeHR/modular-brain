"""Tests for decode_chain composer and Position dataclass."""

import pytest
import torch

from tb.composers import ChainOutput, Position, decode_chain
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


def test_decode_chain_empty_raises() -> None:
    tb = make_tb()
    with pytest.raises(ValueError, match="at least one"):
        decode_chain(tb, [], batch_size=1, device=torch.device("cpu"))


def test_decode_chain_requires_init_or_batch() -> None:
    tb = make_tb()
    with pytest.raises(ValueError, match="initial_q"):
        decode_chain(tb, [Position()])


def test_decode_chain_returns_chain_output() -> None:
    tb = make_tb()
    out = decode_chain(
        tb,
        [Position(), Position(), Position()],
        commit="argmax",
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert isinstance(out, ChainOutput)
    assert len(out.positions) == 3
    assert out.q_final.shape == (2, 8)
    for pos in out.positions:
        assert pos.typed.k is not None
        assert pos.typed.k.shape == (2,)
        assert pos.concept is None


def test_decode_chain_single_position() -> None:
    """A chain of length 1 is valid (e.g. for episodic-only decoding)."""
    tb = make_tb()
    out = decode_chain(tb, [Position()], commit="argmax", batch_size=3, device=torch.device("cpu"))
    assert len(out.positions) == 1


def test_decode_chain_concept_measurement() -> None:
    """When concept_mask is set, a second measurement is performed."""
    groups = IndexGroups.from_sizes([("entity", 6), ("attribute", 4)])
    tb = make_tb(num_indices=groups.num_indices)
    ent_mask = groups.mask("entity")
    attr_mask = groups.mask("attribute")

    out = decode_chain(
        tb,
        [Position(typed_mask=ent_mask, concept_mask=attr_mask)],
        commit="argmax",
        batch_size=5,
        device=torch.device("cpu"),
    )
    pos = out.positions[0]
    assert pos.typed.k is not None and (pos.typed.k < 6).all()
    assert pos.concept is not None
    assert pos.concept.k is not None and (pos.concept.k >= 6).all()
    assert (pos.concept.k < 10).all()


def test_decode_chain_no_concept_when_mask_none() -> None:
    tb = make_tb()
    out = decode_chain(tb, [Position()], commit="argmax", batch_size=1, device=torch.device("cpu"))
    assert out.positions[0].concept is None


def test_decode_chain_teacher_forced_per_position() -> None:
    """A Position with `teacher` set commits to that index regardless of `commit`."""
    tb = make_tb(num_indices=10)
    teacher_a = torch.tensor([2, 3])
    teacher_b = torch.tensor([7, 8])
    out = decode_chain(
        tb,
        [Position(teacher=teacher_a), Position(teacher=teacher_b)],
        commit="argmax",  # would be ignored at teacher-forced positions
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert (out.positions[0].typed.k == teacher_a).all()
    assert (out.positions[1].typed.k == teacher_b).all()


def test_decode_chain_mixed_teacher_and_sample() -> None:
    """First position teacher-forced, second sampled — both work in one chain."""
    tb = make_tb(num_indices=10)
    teacher = torch.tensor([4])
    out = decode_chain(
        tb,
        [Position(teacher=teacher), Position()],
        commit="argmax",
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert (out.positions[0].typed.k == teacher).all()
    # Second position's k is whatever the argmax says — just check it's there.
    assert out.positions[1].typed.k is not None


def test_decode_chain_with_sensory_input() -> None:
    """nu in a Position is passed to attend at that step."""
    tb = make_tb()
    nu = torch.randn(2, 8)
    out = decode_chain(
        tb,
        [Position(nu=nu)],
        commit="argmax",
        batch_size=2,
        device=torch.device("cpu"),
    )
    # Without checking exact dynamics, ensure it runs and produces output.
    assert out.positions[0].typed.k.shape == (2,)


def test_decode_chain_persistent_state() -> None:
    """With TBEvolve, the state threads across positions."""
    tb = make_tb(persistent=True)
    out = decode_chain(
        tb,
        [Position(), Position(), Position()],
        commit="argmax",
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert len(out.positions) == 3


def test_decode_chain_initial_q_carries_through() -> None:
    """A non-zero initial_q changes the first position's logits."""
    tb = make_tb()
    q0_zero = torch.zeros(1, 8)
    q0_seed = torch.randn(1, 8)
    out_zero = decode_chain(tb, [Position()], commit="argmax", initial_q=q0_zero)
    out_seed = decode_chain(tb, [Position()], commit="argmax", initial_q=q0_seed)
    # Logits at first position should differ.
    assert not torch.allclose(
        out_zero.positions[0].typed.logits, out_seed.positions[0].typed.logits
    )


def test_decode_chain_concept_teacher_forced() -> None:
    """concept_teacher forces the concept measurement to the given index."""
    groups = IndexGroups.from_sizes([("entity", 6), ("concept", 4)])
    tb = make_tb(num_indices=groups.num_indices)
    ent_mask = groups.mask("entity")
    con_mask = groups.mask("concept")
    concept_teacher = torch.tensor([7, 8])  # global indices inside concept group [6,10)

    out = decode_chain(
        tb,
        [Position(typed_mask=ent_mask, concept_mask=con_mask, concept_teacher=concept_teacher)],
        commit="argmax",
        batch_size=2,
        device=torch.device("cpu"),
    )
    pos = out.positions[0]
    assert pos.concept is not None
    assert (pos.concept.k == concept_teacher).all()
    # Typed measurement is not teacher-forced — it's argmax, should be in entity range.
    assert (pos.typed.k < 6).all()


def test_decode_chain_concept_teacher_logits_still_masked() -> None:
    """Concept logits respect concept_mask even when teacher-forced."""
    groups = IndexGroups.from_sizes([("entity", 6), ("concept", 4)])
    tb = make_tb(num_indices=groups.num_indices)
    ent_mask = groups.mask("entity")
    con_mask = groups.mask("concept")
    concept_teacher = torch.tensor([7])  # inside concept range

    out = decode_chain(
        tb,
        [Position(typed_mask=ent_mask, concept_mask=con_mask, concept_teacher=concept_teacher)],
        commit="argmax",
        batch_size=1,
        device=torch.device("cpu"),
    )
    logits = out.positions[0].concept.logits[0]
    # Entity positions (0-5) should be masked to -inf.
    assert torch.isinf(logits[:6]).all()
    # Concept positions should be finite.
    assert logits[6:10].isfinite().all()


def test_decode_chain_grad_flow_through_concept() -> None:
    """Gradients flow back to TB parameters through both typed and concept measurements."""
    groups = IndexGroups.from_sizes([("a", 5), ("b", 5)])
    tb = make_tb(num_indices=groups.num_indices)
    a_mask = groups.mask("a")
    b_mask = groups.mask("b")
    out = decode_chain(
        tb,
        [Position(typed_mask=a_mask, concept_mask=b_mask)],
        commit="expectation",
        batch_size=2,
        device=torch.device("cpu"),
    )
    (out.positions[0].typed.logits.sum() + out.positions[0].concept.logits.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tb.parameters())
