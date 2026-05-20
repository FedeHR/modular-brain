"""Tests for EvolveModule implementations."""

import torch

from tb.evolve import EvolveModule, QTBEvolve, TBEvolve


def test_qtb_evolve_protocol() -> None:
    e = QTBEvolve(dim=8, hidden=4)
    assert isinstance(e, EvolveModule)


def test_qtb_evolve_stateless() -> None:
    """QTB evolve returns None state and is deterministic given inputs."""
    e = QTBEvolve(dim=8, hidden=4)
    q = torch.randn(2, 8)
    q1, s1 = e(q, None)
    q2, s2 = e(q, None)
    assert q1.shape == (2, 8)
    assert s1 is None and s2 is None
    assert torch.allclose(q1, q2)


def test_qtb_evolve_init_state() -> None:
    e = QTBEvolve(dim=8, hidden=4)
    assert e.init_state(batch_size=3, device=torch.device("cpu")) is None


def test_tb_evolve_protocol() -> None:
    e = TBEvolve(dim=8, hidden=4)
    assert isinstance(e, EvolveModule)


def test_tb_evolve_persistent_state() -> None:
    """Two TBEvolve calls produce different q outputs because h is carried.

    The deterministic check is: feeding the same q twice with state carried
    forward must differ from feeding it twice with state reset (None).
    """
    e = TBEvolve(dim=8, hidden=4)
    q = torch.randn(2, 8)
    # Carried state.
    q_a, state_a = e(q, None)
    q_b_carried, _ = e(q, state_a)
    # Reset state.
    q_b_fresh, _ = e(q, None)
    # The two must differ (h has accumulated information).
    assert not torch.allclose(q_b_carried, q_b_fresh)


def test_tb_evolve_init_state_shape() -> None:
    e = TBEvolve(dim=8, hidden=5)
    s = e.init_state(batch_size=3, device=torch.device("cpu"))
    assert s.shape == (3, 5)
    assert (s == 0).all()


def test_evolve_grad_flow() -> None:
    """Gradients should flow through both implementations."""
    for cls in (QTBEvolve, TBEvolve):
        e = cls(dim=4, hidden=4)
        q = torch.randn(1, 4, requires_grad=True)
        q_out, _ = e(q, None)
        q_out.sum().backward()
        assert q.grad is not None
        assert q.grad.abs().sum() > 0
