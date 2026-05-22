"""Composers: higher-level decoding patterns built from the primitives.

The canonical loop at each "position" (one column in Figure 2 of
Tensor_Brain.pdf) is:

    evolve  : update h and produce q ← W·h
    attend  : q ← q + μ · g(ν)                              (sensory injection)
    measure : commit to a typed index k_typed (T, S, O, P)
              q ← α·q + β·a_{k_typed}
    measure : (optional) commit to a concept index k_C      (n_C, second box)
              q ← α·q + β·a_{k_C}

`Position` packages the per-column inputs; `decode_chain` runs a list of
them with shared (α, β) and a single evolve state threaded through.

Modes (perception / episodic recall / semantic recall) are different
*input patterns* to this same loop — see the `tb.examples.*` scripts
for runnable demonstrations, and `tb.examples.*_explicit` for the
same demos written with direct primitive calls (no composer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from tb.primitives import TB, Commit, MeasureOutput


@dataclass
class Position:
    """Per-column configuration for `decode_chain`.

    Attributes
    ----------
    nu : Tensor | None
        Encoded sensory input g(ν) for this position. Shape (B, n).
        None ⇒ no sensory drive at this position (pure memory mode).
    typed_mask : Tensor | None
        Restricts the typed measurement to a subset of indices
        (e.g. only entity indices, only predicate indices). Shape (N,).
    teacher : Tensor | None
        If set, the typed measurement is forced to commit to this index
        (commit="teacher" mode). Shape (B,). Useful for memory recall and
        for posterior-shape studies where you condition on a known outcome.
    concept_mask : Tensor | None
        If set, a *second* measurement is performed after the typed one,
        restricted to this mask. This is the n_C box in Figure 2 — the
        per-position unary/concept label.
    concept_teacher : Tensor | None
        If set (and `concept_mask` is set), the concept measurement is
        teacher-forced to this index. Shape (B,). Mirrors `teacher` for
        the concept (class) measurement.
    """

    nu: Tensor | None = None
    typed_mask: Tensor | None = None
    teacher: Tensor | None = None
    concept_mask: Tensor | None = None
    concept_teacher: Tensor | None = None


@dataclass
class PositionOutput:
    """Output of one decoded position: the typed measurement plus an
    optional concept measurement (n_C).
    """

    typed: MeasureOutput
    concept: MeasureOutput | None = None


@dataclass
class ChainOutput:
    """Output of `decode_chain`. Holds the per-position outputs and the
    final q so callers can continue decoding (e.g. for chain reasoning).
    """

    positions: list[PositionOutput] = field(default_factory=list)
    q_final: Tensor | None = None


def decode_chain(
    tb: TB,
    positions: list[Position],
    *,
    mu: float = 1.0,
    commit: Commit = "sample",
    initial_q: Tensor | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> ChainOutput:
    """Decode a chain of positions by composing the three primitives.

    For each Position in order:
        q, state = evolve(q, state)
        q        = attend(q, nu, mu)
        typed    = measure(q, mask=typed_mask, commit=...)
        q        = typed.q
        if concept_mask:
            concept = measure(q, mask=concept_mask, commit=commit)
            q       = concept.q

    The `commit` argument is the default for non-teacher-forced positions.
    A `Position` with `teacher` set always uses commit="teacher" at the
    typed step regardless of this default. A `Position` with `concept_teacher`
    set always uses commit="teacher" at the concept step.

    Parameters
    ----------
    tb : TB
    positions : list[Position]
        At least one Position. Empty list ⇒ ValueError.
    mu : float
        Sensory drive strength applied uniformly across positions.
    commit : Commit
        Default sampling mode at the measurement step.
    initial_q : Tensor or None
        Starting q. If None, falls back to zeros of shape (batch_size, dim);
        in that case `batch_size` and `device` must be provided.
    """
    if not positions:
        raise ValueError("decode_chain requires at least one Position.")

    if initial_q is None:
        if batch_size is None or device is None:
            raise ValueError("Provide either initial_q or (batch_size, device).")
        q = torch.zeros(batch_size, tb.dim, device=device)
    else:
        q = initial_q
        batch_size = q.shape[0]
        device = q.device

    state = tb.init_state(batch_size, device)

    outputs: list[PositionOutput] = []
    for pos in positions:
        q, state = tb.evolve(q, state)
        q = tb.attend(q, nu=pos.nu, mu=mu)

        if pos.teacher is not None:
            typed = tb.measure(q, mask=pos.typed_mask, commit="teacher", teacher_k=pos.teacher)
        else:
            typed = tb.measure(q, mask=pos.typed_mask, commit=commit)
        q = typed.q

        concept: MeasureOutput | None = None
        if pos.concept_mask is not None:
            if pos.concept_teacher is not None:
                concept = tb.measure(
                    q, mask=pos.concept_mask, commit="teacher", teacher_k=pos.concept_teacher
                )
            else:
                concept = tb.measure(q, mask=pos.concept_mask, commit=commit)
            q = concept.q

        outputs.append(PositionOutput(typed=typed, concept=concept))

    return ChainOutput(positions=outputs, q_final=q)


# -- Convenience wrapper for the simple S-O-P case ----------------------------


@dataclass
class TripleOutput:
    """Convenience holder for a 3-position decode (S, O, P)."""

    subject: MeasureOutput
    object: MeasureOutput
    predicate: MeasureOutput


def decode_triple(
    tb: TB,
    *,
    nu_subject: Tensor | None = None,
    nu_object: Tensor | None = None,
    nu_predicate: Tensor | None = None,
    subject_mask: Tensor | None = None,
    object_mask: Tensor | None = None,
    predicate_mask: Tensor | None = None,
    mu: float = 1.0,
    commit: Commit = "sample",
    initial_q: Tensor | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> TripleOutput:
    """Decode one (subject, object, predicate) triple.

    Thin wrapper around `decode_chain` for the simple case — three
    positions, no concept measurements, no teacher forcing. Use
    `decode_chain` directly for the full T+S+O+P pattern, concept
    measurements, teacher-forcing, or any other variation.
    """
    chain = decode_chain(
        tb,
        positions=[
            Position(nu=nu_subject, typed_mask=subject_mask),
            Position(nu=nu_object, typed_mask=object_mask),
            Position(nu=nu_predicate, typed_mask=predicate_mask),
        ],
        mu=mu,
        commit=commit,
        initial_q=initial_q,
        batch_size=batch_size,
        device=device,
    )
    return TripleOutput(
        subject=chain.positions[0].typed,
        object=chain.positions[1].typed,
        predicate=chain.positions[2].typed,
    )
