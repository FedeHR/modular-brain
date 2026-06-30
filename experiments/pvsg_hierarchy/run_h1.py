"""H1 — hierarchical entity grounding under perceptual uncertainty (PVSG).

Claim: because the TB activates *indices* into a shared representation layer and
carries the committed outcome forward through the skip update
(`q ← alpha q + beta a_k`), its leaf→mid→coarse decode stays *hierarchically
consistent* even when the input is corrupted — whereas a flat decoder with
independent per-level heads drifts into invalid (leaf, mid, coarse) combinations.

This is the semantic-memory-as-prior story on real PVSG classes: an ambiguous /
occluded object whose fine class is uncertain should still decode to a coherent
coarser class, not a contradictory one. We defer SAM2/DINO and use one
deterministic feature per leaf class (perception stand-in); the question here is
architectural, not perceptual.

Setup
-----
- Concept space = WordNet taxonomy over PVSG's 126 classes → leaf(126) /
  mid(62) / coarse(11) groups (see `pvsg_data.py`).
- Train both models on the clean leaf prototypes (CE on all three levels).
- Evaluate on the real 7358 PVSG object instances with feature noise of growing
  sigma; report per-level accuracy and the **hierarchical-consistency rate**
  (decoded mid & coarse agree with the taxonomy path of the decoded leaf).

Run:
    uv run python -m experiments.pvsg_hierarchy.run_h1
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.pvsg_hierarchy.pvsg_data import build_concept_space, load_instances
from tb.evolve import QTBEvolve
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha

DIM = 64
LEVELS = ("leaf", "mid", "coarse")


def fixed_features(n: int, dim: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n, dim, generator=g)
    return F.normalize(f, dim=-1) * (dim**0.5)


class FlatHeads(nn.Module):
    """Baseline: independent linear head per level, no shared index, no skip."""

    def __init__(self, dim: int, sizes: dict[str, int]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({lvl: nn.Linear(dim, n) for lvl, n in sizes.items()})

    def forward(self, f: Tensor) -> dict[str, Tensor]:
        return {lvl: head(f) for lvl, head in self.heads.items()}


def tb_decode(tb: TB, masks: dict[str, Tensor], feat: Tensor, commit: str):
    """Decode leaf→mid→coarse, threading the committed q (shared index + skip)."""
    q = tb.attend(torch.zeros(feat.shape[0], tb.dim, device=feat.device), nu=feat)
    logits: dict[str, Tensor] = {}
    preds: dict[str, Tensor] = {}
    for lvl in LEVELS:
        out = tb.measure(q, mask=masks[lvl], commit=commit)
        q = out.q
        logits[lvl] = out.logits
        preds[lvl] = out.logits.argmax(-1)
    return logits, preds


def main() -> None:
    torch.manual_seed(0)
    cs = build_concept_space()
    groups = cs.groups
    tax = cs.tax
    sizes = {lvl: len(tax.vocab(lvl)) for lvl in LEVELS}
    masks = {lvl: groups.mask(lvl) for lvl in LEVELS}

    # taxonomy path per leaf (fully-mapped classes only), in GLOBAL indices.
    mapped = [e for e in tax.entries if e.coarse]  # classes with a full leaf→mid→coarse path
    leaf_vocab = [e.name for e in mapped]
    leaf_g = torch.tensor([cs.gindex("leaf", e.name) for e in mapped])
    mid_g = torch.tensor([cs.gindex("mid", e.mid) for e in mapped])
    coarse_g = torch.tensor([cs.gindex("coarse", e.coarse) for e in mapped])
    targets = {"leaf": leaf_g, "mid": mid_g, "coarse": coarse_g}
    # true mid/coarse global index implied by each leaf (for the consistency check)
    leaf_to_mid_g = dict(zip(leaf_g.tolist(), mid_g.tolist()))
    leaf_to_coarse_g = dict(zip(leaf_g.tolist(), coarse_g.tolist()))

    feats = fixed_features(len(mapped), DIM)  # one prototype per mapped leaf class

    # --- models -------------------------------------------------------------
    tb = TB(
        dim=DIM,
        index_layer=IndexLayer(num_indices=groups.num_indices, dim=DIM),
        evolve_module=QTBEvolve(dim=DIM, hidden=64, skip=True),
        alpha=learnable_alpha(1.0),
        beta=1.0,
    )
    flat = FlatHeads(DIM, sizes)

    def train(step_fn, params, steps=400, lr=5e-2):
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = step_fn()
            loss.backward()
            opt.step()

    def tb_step():
        logits, _ = tb_decode(tb, masks, feats, commit="expectation")
        return sum(F.cross_entropy(logits[l], targets[l]) for l in LEVELS)

    def flat_step():
        out = flat(feats)
        # flat heads are local per group; targets are local indices (0..size-1)
        local = {l: targets[l] - groups.offsets[groups.names.index(l)] for l in LEVELS}
        return sum(F.cross_entropy(out[l], local[l]) for l in LEVELS)

    train(tb_step, tb.parameters())
    train(flat_step, flat.parameters())

    # --- evaluate on the real PVSG instances under feature noise ------------
    instances, _ = load_instances(cs=cs)
    leaf_idx = torch.tensor([leaf_vocab.index(i.leaf) for i in instances])  # row in feats
    base = feats[leaf_idx]
    true_leaf_g = leaf_g[leaf_idx]

    print("PVSG H1 — hierarchical grounding under perceptual noise")
    print(f"  concept space: leaf {sizes['leaf']} / mid {sizes['mid']} / coarse "
          f"{sizes['coarse']}   |  eval instances: {len(instances)}\n")
    print(f"  {'noise':>6} | {'model':4} | {'leaf':>5} {'mid':>5} {'coarse':>6} | "
          f"{'consistent':>10}")
    print("  " + "-" * 52)

    off = {l: groups.offsets[groups.names.index(l)] for l in LEVELS}
    for sigma in (0.0, 0.5, 1.0, 2.0):
        torch.manual_seed(1)
        noisy = base + sigma * torch.randn_like(base)
        with torch.no_grad():
            # TB
            _, tb_pred = tb_decode(tb, masks, noisy, commit="argmax")
            # Flat
            flat_out = flat(noisy)
            flat_pred = {l: flat_out[l].argmax(-1) + off[l] for l in LEVELS}

            for name, pred in (("TB", tb_pred), ("flat", flat_pred)):
                accs = {l: (pred[l] == (true_leaf_g if l == "leaf" else targets[l][leaf_idx])).float().mean().item()
                        for l in LEVELS}
                # consistency: decoded mid/coarse match the taxonomy path of decoded leaf
                cons = torch.tensor([
                    leaf_to_mid_g.get(int(pl)) == int(pm) and leaf_to_coarse_g.get(int(pl)) == int(pc)
                    for pl, pm, pc in zip(pred["leaf"], pred["mid"], pred["coarse"])
                ]).float().mean().item()
                tag = f"σ={sigma:>3}" if name == "TB" else "    "
                print(f"  {tag:>6} | {name:4} | {accs['leaf']:.2f}  {accs['mid']:.2f}  "
                      f"{accs['coarse']:.2f}  |   {cons:.2f}")
        print()


if __name__ == "__main__":
    main()
