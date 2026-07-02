"""H1 on real DINO features — prototype-train / real-eval.

The mock `run_h1.py` trained on random per-class prototypes and simulated
perception as `σ·randn` noise. This runs the same claim on real perception:

- **Train (semantic memory):** one prototype per leaf class, built
  *per-instance-then-per-class* — average each object's frames, then average the
  objects of a class (so a 692-frame object doesn't outweigh a 3-frame one). This
  is the centroid summary the TB's similarity readout `⟨sigmoid(q), A_c⟩` warrants.
- **Eval (perception):** held-out object instances' *individual frames* — real
  occlusion / blur / viewpoint, no synthetic noise. Split is **by instance**
  (`instance_id`), so no frame of a training object leaks into eval.

Reports per-level accuracy and the hierarchical-consistency rate (decoded mid &
coarse lie on the taxonomy path of the decoded leaf) for TB vs the flat baseline —
the same table as `run_h1`, now on real features.

    python -m experiments.pvsg_hierarchy.run_h1_real --cache $WORK/cache
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.pvsg_hierarchy.pvsg_features import LEVELS, FeatureTable, load_feature_table
from experiments.pvsg_hierarchy.run_h1 import FlatHeads, tb_decode
from tb.evolve import QTBEvolve
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha


def _norm(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1) * (x.shape[-1] ** 0.5)


def split_instances(inst: FeatureTable, train_frac: float, seed: int) -> tuple[Tensor, Tensor]:
    """Per-leaf-class train/eval split over instances. A class's instances are
    shuffled and cut at `train_frac`; singleton classes go to train (so a
    prototype exists) and simply have no eval instance. Returns boolean masks
    (train, eval) over the per-instance rows."""
    g = torch.Generator().manual_seed(seed)
    train = torch.zeros(inst.feats.shape[0], dtype=torch.bool)
    for c in inst.leaf_g.unique():
        idx = (inst.leaf_g == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_train = max(1, int(round(train_frac * len(idx))))
        train[idx[:n_train]] = True
    return train, ~train


def build_prototypes(inst: FeatureTable, train_mask: Tensor):
    """Per-class mean of the (already per-instance) train vectors → one prototype
    per leaf class, with its (leaf, mid, coarse) global-index triple."""
    feats, leaf_g, mid_g, coarse_g = [], [], [], []
    for c in inst.leaf_g[train_mask].unique():
        rows = (inst.leaf_g == c) & train_mask
        feats.append(_norm(inst.feats[rows].mean(0)))
        j = rows.nonzero(as_tuple=True)[0][0]  # any instance of the class carries the triple
        leaf_g.append(int(inst.leaf_g[j]))
        mid_g.append(int(inst.mid_g[j]))
        coarse_g.append(int(inst.coarse_g[j]))
    return (torch.stack(feats), torch.tensor(leaf_g), torch.tensor(mid_g), torch.tensor(coarse_g))


@torch.no_grad()
def evaluate(models, masks, feats: Tensor, truth: dict[str, Tensor],
             leaf_to_mid, leaf_to_coarse, off, batch: int = 8192):
    """Per-level accuracy + hierarchical-consistency for each model, batched."""
    tb, flat = models
    acc = {name: {lvl: 0 for lvl in LEVELS} for name in ("TB", "flat")}
    cons = {name: 0 for name in ("TB", "flat")}
    n = feats.shape[0]
    for s in range(0, n, batch):
        fb = feats[s:s + batch]
        tb_pred = tb_decode(tb, masks, fb, commit="argmax")[1]
        fout = flat(fb)
        flat_pred = {lvl: fout[lvl].argmax(-1) + off[lvl] for lvl in LEVELS}
        for name, pred in (("TB", tb_pred), ("flat", flat_pred)):
            for lvl in LEVELS:
                acc[name][lvl] += int((pred[lvl] == truth[lvl][s:s + batch]).sum())
            cons[name] += int(sum(
                leaf_to_mid.get(int(pl)) == int(pm) and leaf_to_coarse.get(int(pl)) == int(pc)
                for pl, pm, pc in zip(pred["leaf"], pred["mid"], pred["coarse"])
            ))
    for name in acc:
        for lvl in LEVELS:
            acc[name][lvl] /= n
        cons[name] /= n
    return acc, cons


def run(table: FeatureTable, *, train_frac: float = 0.5, seed: int = 0, steps: int = 400):
    torch.manual_seed(seed)
    cs = table.cs
    groups = cs.groups
    dim = table.dim
    masks = {lvl: groups.mask(lvl) for lvl in LEVELS}
    off = {lvl: groups.offsets[groups.names.index(lvl)] for lvl in LEVELS}
    sizes = {lvl: groups.sizes[groups.names.index(lvl)] for lvl in LEVELS}

    # taxonomy path per leaf (global indices) for the consistency check
    leaf_to_mid = {cs.gindex("leaf", e.name): cs.gindex("mid", e.mid)
                   for e in cs.tax.entries if e.coarse}
    leaf_to_coarse = {cs.gindex("leaf", e.name): cs.gindex("coarse", e.coarse)
                      for e in cs.tax.entries if e.coarse}

    # per-instance means, then split by instance, then per-class prototypes
    inst = table.pool_by_instance(normalize=True)
    train_mask, eval_mask = split_instances(inst, train_frac, seed)
    proto_f, proto_leaf, proto_mid, proto_coarse = build_prototypes(inst, train_mask)
    proto_tgt = {"leaf": proto_leaf, "mid": proto_mid, "coarse": proto_coarse}

    # eval = individual frames of the held-out instances
    eval_inst = eval_mask.nonzero(as_tuple=True)[0]
    frame_eval = torch.isin(table.instance_id, eval_inst)
    feats_eval = table.feats[frame_eval]
    truth = {lvl: table.target(lvl)[frame_eval] for lvl in LEVELS}

    # --- models (same construction as run_h1, dim = feature dim) -------------
    tb = TB(dim=dim,
            index_layer=IndexLayer(num_indices=groups.num_indices, dim=dim),
            evolve_module=QTBEvolve(dim=dim, hidden=64, skip=True),
            alpha=learnable_alpha(1.0), beta=1.0)
    flat = FlatHeads(dim, sizes)

    def train(step_fn, params):
        opt = torch.optim.Adam(params, lr=5e-2)
        for _ in range(steps):
            opt.zero_grad()
            step_fn().backward()
            opt.step()

    def tb_step():
        logits, _ = tb_decode(tb, masks, proto_f, commit="expectation")
        return sum(F.cross_entropy(logits[l], proto_tgt[l]) for l in LEVELS)

    def flat_step():
        out = flat(proto_f)
        local = {l: proto_tgt[l] - off[l] for l in LEVELS}
        return sum(F.cross_entropy(out[l], local[l]) for l in LEVELS)

    train(tb_step, tb.parameters())
    train(flat_step, flat.parameters())

    acc, cons = evaluate((tb, flat), masks, feats_eval, truth,
                         leaf_to_mid, leaf_to_coarse, off)
    return {
        "acc": acc, "cons": cons,
        "n_proto": proto_f.shape[0],
        "n_train_inst": int(train_mask.sum()), "n_eval_inst": int(eval_mask.sum()),
        "n_eval_frames": int(frame_eval.sum()),
        "sizes": sizes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dir with <video_id>.pt files")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    table = load_feature_table(args.cache)
    print(table.stats.summary())
    r = run(table, train_frac=args.train_frac, seed=args.seed, steps=args.steps)

    print("\nH1 (real DINO features) — prototype-train / real-eval")
    print(f"  concept: leaf {r['sizes']['leaf']} / mid {r['sizes']['mid']} / "
          f"coarse {r['sizes']['coarse']}")
    print(f"  prototypes {r['n_proto']} | train inst {r['n_train_inst']} | "
          f"eval inst {r['n_eval_inst']} | eval frames {r['n_eval_frames']}\n")
    print(f"  {'model':4} | {'leaf':>5} {'mid':>5} {'coarse':>6} | {'consistent':>10}")
    print("  " + "-" * 44)
    for name in ("TB", "flat"):
        a = r["acc"][name]
        print(f"  {name:4} | {a['leaf']:.2f}  {a['mid']:.2f}  {a['coarse']:.2f}  "
              f"|   {r['cons'][name]:.2f}")


if __name__ == "__main__":
    main()
