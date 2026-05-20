"""Evolution operator: the recurrent dynamics on q.

QTB §6.2 derives a single-hidden-layer form from Jensen's approximation:

    h ← sig(v_0 + V q)
    q ← W h

The original TB (TB §4.4, Algorithm 1, lines 16/23/28) uses a deeper form
with persistent h across decoding positions:

    h ← B · sig[sig(h) + V sig(q)]
    q ← W sig(h)

Both are implementations of the same protocol. The state (`h`, or whatever
a future SSM/xLSTM wants to keep) is opaque to the rest of the system.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class EvolveModule(Protocol):
    """Protocol for the dynamic-context-layer recurrence.

    Implementations may carry private state of any shape (h-vector, SSM
    hidden state, mLSTM matrix memory, ...). Callers thread the state
    through but never inspect it.
    """

    def forward(self, q: Tensor, state: object | None) -> tuple[Tensor, object]:
        """One step of the recurrence. Returns (q_new, state_new)."""
        ...

    def init_state(self, batch_size: int, device: torch.device) -> object | None:
        """Initial state for a fresh episode. May return None if stateless."""
        ...


class QTBEvolve(nn.Module):
    """Single-hidden-layer recurrence (QTB Eq. 27).

    h = sig(v_0 + V q)
    q = W h               (with `skip=False`, the bare QTB Algorithm 1 form)
    q = q + W h           (with `skip=True`, the QTB §7.5 "ResNet-style"
                           refinement; recommended for non-trivial tasks
                           because the bare form contracts information
                           through the sigmoid bottleneck)
    """

    def __init__(self, dim: int, hidden: int, *, skip: bool = True) -> None:
        super().__init__()
        self.V = nn.Linear(dim, hidden)  # includes v_0
        self.W = nn.Linear(hidden, dim, bias=False)
        self.skip = skip

    def forward(self, q: Tensor, state: object | None = None) -> tuple[Tensor, None]:
        del state
        h = torch.sigmoid(self.V(q))
        delta = self.W(h)
        return (q + delta if self.skip else delta), None

    def init_state(self, batch_size: int, device: torch.device) -> None:
        del batch_size, device
        return None


class TBEvolve(nn.Module):
    """Two-hidden-layer recurrence with persistent h across calls.

    h = B · sig[ sig(h_prev) + V · sig(q) ]
    q = W · sig(h)

    Matches the original TB g(·) (TB §4.4) and the persistent-h pattern
    in Algorithm 1 lines 16/23/28.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.V = nn.Linear(dim, hidden, bias=False)
        self.B = nn.Linear(hidden, hidden, bias=False)
        self.W = nn.Linear(hidden, dim, bias=False)
        self.hidden = hidden

    def forward(self, q: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        if state is None:
            state = q.new_zeros(q.shape[0], self.hidden)
        h = self.B(torch.sigmoid(torch.sigmoid(state) + self.V(torch.sigmoid(q))))
        q_new = self.W(torch.sigmoid(h))
        return q_new, h

    def init_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden, device=device)
