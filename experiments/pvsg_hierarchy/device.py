"""Device selection shared by the experiment runners.

The runners are matmul-bound (measure logits over all indices), so any GPU
helps. Data wiring (splits, distances, grouping) stays on CPU; only masks,
models, and training/eval tensors move to the picked device.
"""

from __future__ import annotations

import torch


def pick_device(name: str | None = None) -> torch.device:
    """Resolve a device: an explicit name wins; None/'auto' prefers Apple's
    MPS, then CUDA, then CPU."""
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
