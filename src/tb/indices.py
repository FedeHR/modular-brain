"""Index layer and grouped-softmax masking.

The index layer holds the embedding matrix `A ∈ R^{n × N}`, which serves
*both* as the encoding weights (`A^T q` produces logits) and as the source
of decoded embeddings (`A[:, k]` is the embedding of outcome `k`).

This weight tying is the central architectural commitment of the BTN
(see Tensor_Brain.pdf §4.3, §7.6): embeddings *are* connection weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class IndexLayer(nn.Module):
    """Weight-tied embedding/scoring layer.

    `forward(q)` computes logits `A^T q + a_0` for all indices.
    `embed(k)` returns embedding column `A[:, k]` for sampled indices.

    Parameters
    ----------
    num_indices : int
        N — number of discrete indices.
    dim : int
        n — embedding/representation dimensionality.
    use_bias : bool
        If True, include a learnable per-index bias `a_0`.
    """

    def __init__(self, num_indices: int, dim: int, *, use_bias: bool = True) -> None:
        super().__init__()
        # Store as (N, n) so that `weight[k]` is the k-th embedding column a_k.
        # Equivalent to QTB's A ∈ R^{n × N} with A[:, k] = a_k.
        self.weight = nn.Parameter(torch.empty(num_indices, dim))
        self.bias = nn.Parameter(torch.zeros(num_indices)) if use_bias else None
        self.num_indices = num_indices
        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")

    def forward(self, q: Tensor) -> Tensor:
        """Compute logits over all indices: (B, n) -> (B, N)."""
        logits = q @ self.weight.t()
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def embed(self, k: Tensor) -> Tensor:
        """Look up embedding vectors for sampled indices: (B,) -> (B, n)."""
        return self.weight[k]


@dataclass(frozen=True)
class IndexGroups:
    """Disjoint partitioning of the index set for typed/grouped operations.

    Each group is a contiguous slice `[start, end)` over the global index range.
    Groups are used both for masking (restrict softmax to one group) and for
    grouped softmax (independent softmax per group).

    The default mode is `flat`: one group spanning all indices. This is the
    QTB-style behavior and recovers an ordinary softmax.

    Examples
    --------
    >>> g = IndexGroups.flat(100)
    >>> g = IndexGroups.from_sizes([("entity", 100), ("predicate", 20)])
    """

    sizes: tuple[int, ...]
    names: tuple[str, ...]

    @classmethod
    def flat(cls, num_indices: int) -> IndexGroups:
        return cls(sizes=(num_indices,), names=("all",))

    @classmethod
    def from_sizes(cls, named_sizes: list[tuple[str, int]]) -> IndexGroups:
        names, sizes = zip(*named_sizes, strict=True)
        return cls(sizes=tuple(sizes), names=tuple(names))

    @property
    def num_indices(self) -> int:
        return sum(self.sizes)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Starting offset of each group in the global index range."""
        offs: list[int] = [0]
        for s in self.sizes[:-1]:
            offs.append(offs[-1] + s)
        return tuple(offs)

    def slice_of(self, name: str) -> slice:
        """Global slice [start, end) for the named group."""
        i = self.names.index(name)
        start = self.offsets[i]
        return slice(start, start + self.sizes[i])

    def mask(self, name: str, *, device: torch.device | None = None) -> Tensor:
        """Boolean mask selecting only the named group's indices.

        Returns shape (N,). True where index belongs to the group.
        """
        m = torch.zeros(self.num_indices, dtype=torch.bool, device=device)
        m[self.slice_of(name)] = True
        return m


def masked_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
    """Softmax over `logits`, with -inf at positions where `mask` is False.

    `mask` may be shape (N,) (broadcast over batch) or (B, N).
    """
    if mask is not None:
        # Broadcast (N,) to (B, N) implicitly via masked_fill.
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits.softmax(dim=-1)
