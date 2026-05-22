"""The TB module: three primitives over the cognitive brain state `q`.

Following QTB Algorithms 1–3, every operation has signature `q -> q`
(possibly returning a sampled index and/or updating an opaque evolve state):

    evolve : q ← W·h,  h ← f(h, q)                 (recurrent dynamics)
    attend : q ← q + μ · g(ν)                      (sensory injection)
    measure: q ← α·q + β·a_k                       (commit on sampled k)

The QTB §7.4 "ignorant Y-measurement" (attention readout) is recovered
as `measure(commit="expectation")` — a soft, no-commit measurement that
adds the full softmax-weighted readout instead of one committed `a_k`.

The commit modes match the official BTN code (integral/max/sample/teacher)
plus a Gumbel-softmax mode for differentiable hard sampling during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tb.evolve import EvolveModule
from tb.indices import IndexLayer

Commit = Literal["sample", "argmax", "expectation", "gumbel", "teacher"]


@dataclass
class MeasureOutput:
    """Result of one measurement.

    Attributes
    ----------
    k : Tensor | None
        Sampled index, shape (B,). None for `commit="expectation"`.
    q : Tensor
        Updated representation vector, shape (B, n).
    logits : Tensor
        Raw pre-softmax scores, shape (B, N). Kept for downstream losses.
    """

    k: Tensor | None
    q: Tensor
    logits: Tensor


class TB(nn.Module):
    """The Tensor Brain core.

    Holds the index layer, an evolve module, and the (α, β, μ) knobs that
    select between operational modes (perception / memory / PVM / gRNN).

    Parameters
    ----------
    dim : int
        n — dimensionality of the representation layer.
    index_layer : IndexLayer
        Holds the embedding matrix A and per-index bias a_0.
    evolve_module : EvolveModule
        Recurrence on q. Carries any per-episode state itself.
    alpha, beta : float | nn.Parameter
        Mix weights in the measurement update `q ← α q + β a_k`.
        Either fixed floats or `nn.Parameter` for learning (the official
        BTN code learns α; see `learnable_alpha` factory below).
    """

    def __init__(
        self,
        dim: int,
        index_layer: IndexLayer,
        evolve_module: EvolveModule,
        *,
        alpha: float | nn.Parameter = 1.0,
        beta: float | nn.Parameter = 1.0,
    ) -> None:
        super().__init__()
        if index_layer.dim != dim:
            raise ValueError(f"IndexLayer dim {index_layer.dim} != TB dim {dim}")
        self.dim = dim
        self.index_layer = index_layer
        self.evolve_module = evolve_module
        # Register alpha/beta as buffers if plain floats, params if nn.Parameter.
        if isinstance(alpha, nn.Parameter):
            self.alpha = alpha
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        if isinstance(beta, nn.Parameter):
            self.beta = beta
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    # --- Primitive 1: evolve --------------------------------------------------

    def evolve(self, q: Tensor, state: object | None = None) -> tuple[Tensor, object]:
        """Apply one step of the recurrence. Returns (q_new, state_new)."""
        return self.evolve_module.forward(q, state)

    def init_state(self, batch_size: int, device: torch.device) -> object | None:
        """Get an initial evolve state for a fresh episode."""
        return self.evolve_module.init_state(batch_size, device)

    # --- Primitive 2: attend --------------------------------------------------

    def attend(
        self,
        q: Tensor,
        *,
        nu: Tensor | None = None,
        mu: float = 1.0,
    ) -> Tensor:
        """Sensory injection: q ← q + μ · g(ν).

        The encoder g(·) is external to the TB core; pass already-encoded
        input as `nu`. With `nu=None`, this is a no-op (pure memory mode).

        Notes
        -----
        The QTB §7.4 "ignorant Y-measurement" (the soft readout you might
        also call "attention over indices") is `measure(commit="expectation")`,
        not part of `attend`. Keeping sensory injection and soft observation
        separate avoids the double-readout that arises when both are chained.
        """
        if nu is None:
            return q
        return q + mu * nu

    # --- Primitive 3: measure -------------------------------------------------

    def measure(
        self,
        q: Tensor,
        *,
        mask: Tensor | None = None,
        commit: Commit = "sample",
        teacher_k: Tensor | None = None,
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
    ) -> MeasureOutput:
        """Commit to an index k and update q via the (α, β) skip/outcome mix.

        `q ← α q + β a_k`, where `a_k` is the embedding for the committed
        outcome. Different `commit` modes implement different choices of
        "committed outcome":

        - "sample"      : Categorical sample (non-differentiable forward).
        - "argmax"      : winner-take-all; β→∞ limit of softmax sampling.
        - "expectation" : a_k ← Σ_k p_k a_k (no actual commit; differentiable).
                          This is the QTB §7.4 ignorant Y-measurement /
                          attention readout.
        - "gumbel"      : straight-through Gumbel-softmax (differentiable).
        - "teacher"     : commit to a given `teacher_k`; for ablations,
                          conditional decoding, and memory recall tasks
                          where an index is provided rather than sampled.

        Returns logits regardless of mode so that a cross-entropy / NLL loss
        can be computed at the call site.
        """
        logits = temperature * self.index_layer(torch.sigmoid(q))
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        k: Tensor | None
        if commit == "sample":
            probs = logits.softmax(dim=-1)
            k = torch.distributions.Categorical(probs=probs).sample()
            a_k = self.index_layer.embed(k)
        elif commit == "argmax":
            k = logits.argmax(dim=-1)
            a_k = self.index_layer.embed(k)
        elif commit == "expectation":
            probs = logits.softmax(dim=-1)
            a_k = probs @ self.index_layer.weight
            k = None
        elif commit == "gumbel":
            onehot = F.gumbel_softmax(logits, tau=gumbel_tau, hard=gumbel_hard, dim=-1)
            a_k = onehot @ self.index_layer.weight
            k = onehot.argmax(dim=-1)
        elif commit == "teacher":
            if teacher_k is None:
                raise ValueError("commit='teacher' requires `teacher_k`.")
            k = teacher_k
            a_k = self.index_layer.embed(k)
        else:
            raise ValueError(f"Unknown commit mode: {commit!r}")

        q = self.alpha * q + self.beta * a_k
        return MeasureOutput(k=k, q=q, logits=logits)


def learnable_alpha(initial: float = 1.0) -> nn.Parameter:
    """Convenience: a learnable scalar α, matching the official BTN code.

    The published BTN numbers depend on a learned α (see `train_scale=True`
    in their config). Use this to pass into `TB(..., alpha=learnable_alpha())`.
    """
    return nn.Parameter(torch.tensor(float(initial)))


def learnable_beta(initial: float = 1.0) -> nn.Parameter:
    """Convenience: a learnable scalar β for the outcome contribution.

    Symmetric to `learnable_alpha`. Use when you want the model to discover
    the right prior/outcome balance on both sides of `q ← α q + β a_k`.
    """
    return nn.Parameter(torch.tensor(float(initial)))
