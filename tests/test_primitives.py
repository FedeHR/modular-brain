"""Tests for the TB primitives: evolve, attend, measure."""

import pytest
import torch
import torch.nn as nn

from tb.evolve import QTBEvolve, TBEvolve
from tb.indices import IndexGroups, IndexLayer
from tb.primitives import TB, learnable_alpha


def make_tb(dim: int = 8, num_indices: int = 10, *, learn_alpha: bool = False) -> TB:
    alpha = learnable_alpha(1.0) if learn_alpha else 1.0
    return TB(
        dim=dim,
        index_layer=IndexLayer(num_indices=num_indices, dim=dim),
        evolve_module=QTBEvolve(dim=dim, hidden=4),
        alpha=alpha,
        beta=1.0,
    )


def test_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dim"):
        TB(
            dim=8,
            index_layer=IndexLayer(num_indices=5, dim=6),
            evolve_module=QTBEvolve(dim=8, hidden=4),
        )


def test_evolve_threads_state() -> None:
    """TB.evolve delegates to the evolve module; state threads through."""
    tb = TB(
        dim=4,
        index_layer=IndexLayer(num_indices=5, dim=4),
        evolve_module=TBEvolve(dim=4, hidden=3),
    )
    q = torch.randn(2, 4)
    state = tb.init_state(2, torch.device("cpu"))
    q1, state1 = tb.evolve(q, state)
    assert state1.shape == (2, 3)
    q2, state2 = tb.evolve(q1, state1)
    assert q2.shape == (2, 4)


def test_attend_no_input_no_change() -> None:
    """attend with nu=None is a no-op."""
    tb = make_tb()
    q = torch.randn(2, 8)
    q_new = tb.attend(q)
    assert torch.allclose(q_new, q)


def test_attend_adds_sensory() -> None:
    """attend with nu adds μ · nu to q (pure injection, no readout)."""
    tb = make_tb()
    q = torch.zeros(2, 8)
    nu = torch.randn(2, 8)
    q_new = tb.attend(q, nu=nu, mu=1.0)
    assert torch.allclose(q_new, nu)


def test_attend_mu_scales() -> None:
    """μ scales the sensory contribution linearly."""
    tb = make_tb()
    q = torch.zeros(1, 8)
    nu = torch.randn(1, 8)
    q_half = tb.attend(q, nu=nu, mu=0.5)
    q_full = tb.attend(q, nu=nu, mu=1.0)
    assert torch.allclose(q_full, 2 * q_half)


def test_attend_is_softmax_independent() -> None:
    """attend no longer does a readout, so its output is purely linear in nu."""
    tb = make_tb()
    q = torch.zeros(1, 8)
    nu1 = torch.randn(1, 8)
    nu2 = torch.randn(1, 8)
    # Linearity: attend(q, nu1+nu2) == attend(q, nu1) + attend(q, nu2) - q.
    lhs = tb.attend(q, nu=nu1 + nu2)
    rhs = tb.attend(q, nu=nu1) + tb.attend(q, nu=nu2) - q
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_measure_commit_modes() -> None:
    """All four commit modes produce a valid q update and logits."""
    tb = make_tb()
    q = torch.randn(3, 8)

    out_sample = tb.measure(q, commit="sample")
    assert out_sample.k is not None and out_sample.k.shape == (3,)
    assert out_sample.q.shape == (3, 8)
    assert out_sample.logits.shape == (3, 10)

    out_argmax = tb.measure(q, commit="argmax")
    assert out_argmax.k is not None
    # Argmax must equal logits.argmax.
    assert (out_argmax.k == out_argmax.logits.argmax(-1)).all()

    out_exp = tb.measure(q, commit="expectation")
    assert out_exp.k is None
    assert out_exp.q.shape == (3, 8)

    out_gumbel = tb.measure(q, commit="gumbel")
    assert out_gumbel.k is not None


def test_measure_unknown_commit_raises() -> None:
    tb = make_tb()
    with pytest.raises(ValueError, match="commit mode"):
        tb.measure(torch.zeros(1, 8), commit="nonsense")  # type: ignore[arg-type]


def test_measure_teacher_commit() -> None:
    """teacher mode forces commit to the supplied teacher_k."""
    tb = make_tb(num_indices=10)
    q = torch.randn(4, 8)
    teacher = torch.tensor([2, 7, 0, 5])
    out = tb.measure(q, commit="teacher", teacher_k=teacher)
    assert out.k is not None
    assert (out.k == teacher).all()
    # q update uses teacher's embedding regardless of what logits prefer.
    expected = q + tb.index_layer.embed(teacher)  # α=β=1
    assert torch.allclose(out.q, expected, atol=1e-6)


def test_measure_teacher_requires_teacher_k() -> None:
    tb = make_tb()
    with pytest.raises(ValueError, match="teacher_k"):
        tb.measure(torch.zeros(1, 8), commit="teacher")


def test_measure_teacher_logits_still_masked() -> None:
    """Even in teacher mode, returned logits respect the mask (for the loss)."""
    tb = make_tb(num_indices=10)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[3:7] = True
    teacher = torch.tensor([4])  # inside the mask
    out = tb.measure(torch.zeros(1, 8), mask=mask, commit="teacher", teacher_k=teacher)
    # Masked positions should be -inf so cross-entropy works.
    assert torch.isinf(out.logits[0, 0]).item()
    assert out.logits[0, 4].isfinite()


def test_measure_argmax_update_formula() -> None:
    """q ← α q + β a_k for argmax with α=β=1."""
    tb = make_tb()
    q = torch.randn(2, 8)
    out = tb.measure(q, commit="argmax")
    expected = q + tb.index_layer.embed(out.k)
    assert torch.allclose(out.q, expected, atol=1e-6)


def test_measure_expectation_update() -> None:
    """Expectation update: q ← α q + β · Σ p_k a_k."""
    tb = make_tb()
    q = torch.randn(2, 8)
    out = tb.measure(q, commit="expectation")
    probs = out.logits.softmax(-1)
    expected = q + probs @ tb.index_layer.weight
    assert torch.allclose(out.q, expected, atol=1e-6)


def test_measure_alpha_zero_drops_skip() -> None:
    """α=0 (PVM mode): q ← β a_k, dropping the prior entirely."""
    tb = make_tb()
    with torch.no_grad():
        tb.alpha.fill_(0.0)
    q = torch.randn(1, 8)
    out = tb.measure(q, commit="argmax")
    expected = tb.index_layer.embed(out.k)
    assert torch.allclose(out.q, expected, atol=1e-6)


def test_measure_beta_zero_drops_outcome() -> None:
    """β=0 (gRNN mode): q ← α q, outcome ignored."""
    tb = make_tb()
    with torch.no_grad():
        tb.beta.fill_(0.0)
    q = torch.randn(1, 8)
    out = tb.measure(q, commit="argmax")
    assert torch.allclose(out.q, q, atol=1e-6)


def test_measure_gradient_through_expectation() -> None:
    """Gradients flow back through expectation commit."""
    tb = make_tb()
    q = torch.randn(2, 8, requires_grad=True)
    out = tb.measure(q, commit="expectation")
    out.q.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_measure_gradient_through_gumbel() -> None:
    """Gradients flow back through straight-through Gumbel."""
    tb = make_tb()
    q = torch.randn(2, 8, requires_grad=True)
    out = tb.measure(q, commit="gumbel")
    out.q.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_measure_mask_restricts_sampled_index() -> None:
    """Sampled / argmax k must lie inside the allowed mask."""
    tb = make_tb(num_indices=10)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[2:5] = True  # only [2, 3, 4]
    q = torch.randn(20, 8)
    for commit in ("sample", "argmax", "gumbel"):
        out = tb.measure(q, mask=mask, commit=commit)
        assert ((out.k >= 2) & (out.k < 5)).all(), commit


def test_learnable_alpha_is_parameter() -> None:
    tb = make_tb(learn_alpha=True)
    assert isinstance(tb.alpha, nn.Parameter)
    assert tb.alpha.requires_grad
    # Should appear in parameters() iteration.
    assert any(p is tb.alpha for p in tb.parameters())


def test_fixed_alpha_is_buffer() -> None:
    tb = make_tb(learn_alpha=False)
    assert not isinstance(tb.alpha, nn.Parameter)
    assert tb.alpha in dict(tb.named_buffers()).values()


def test_groups_with_measure_mask() -> None:
    """IndexGroups.mask works as a measure mask."""
    groups = IndexGroups.from_sizes([("ent", 6), ("pred", 4)])
    tb = make_tb(num_indices=groups.num_indices)
    pred_mask = groups.mask("pred")
    q = torch.randn(5, 8)
    out = tb.measure(q, mask=pred_mask, commit="argmax")
    assert (out.k >= 6).all() and (out.k < 10).all()
