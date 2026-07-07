"""V2 — per-moment triple decoding with the dynamic context layer (Tables 4/6).

The TB paper decoded VRD triples whose predicate is a static co-occurrence
fact; PVSG predicates hold over frame intervals, so here the model decodes
`(subject, predicate, object)` from the two participants' DINO features *at one
frame* — the relation label is per-moment, which is the setting the dynamic
context layer h was designed for.

Decode: S → O → P through `decode_chain` (evolve → attend → measure per
position). With `TBEvolve`, q is regenerated from h at every position
(Algorithm 1), so by the P position the evidence for the predicate lives in h
plus the two committed entity embeddings mixed into q — the paper's claim that
binary labels need the context layer, retested on honest data.

Conditions (each trained separately, identical data/loss/steps):
- TB        : TBEvolve (persistent h) + committed entities (teacher-forced
              training, argmax eval).
- TB-noH    : evolve replaced by identity — information reaches P only through
              the committed q (skip mix). The context-layer ablation.
- TB-noCommit: TBEvolve but expectation (soft) readout at S and O — commitment
              ablation, h intact.
- flat      : linear heads — subject entity from feat_s, object entity from
              feat_o, predicate from [feat_s ‖ feat_o]. Sees the concatenated
              evidence directly; a deliberately strong baseline.

Order-invariance (QTB): the trained TB decodes S→O→P and O→S→P; we report
predicate agreement between the two orders and each order's accuracy.

Rows whose participants have no entity index (unmapped category) are skipped
and counted — the entity targets must exist for the S/O measurements.

Multi-label predicates: overlapping spans of one (subject, object) pair make
the per-frame predicate target a *set* — `build_pair_table` emits one row per
span, so the same features appear in several rows with different labels.
Training stays plain CE (the model learns the conditional label frequencies —
the TB's probabilistic semantics); evaluation dedupes rows into
(video, frame, subj, obj) groups and scores Hits@k against the group's
true-predicate set, as in the paper's Table 4. Naive per-row top-1 accuracy
has a <100% ceiling on multi-label frames that differs per condition through
calibration alone, so it is not reported.

    python -m experiments.pvsg_hierarchy.run_v2 --cache $WORK/cache --timelines $WORK/timelines
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.pvsg_hierarchy.device import pick_device
from experiments.pvsg_hierarchy.pvsg_data import (
    VideoSpace,
    build_concept_space,
    build_video_space,
    load_relation_spans,
)
from experiments.pvsg_hierarchy.pvsg_pairs import PairTable, build_pair_table
from experiments.pvsg_hierarchy.tracking import track
from tb.composers import Position, decode_chain
from tb.evolve import IdentityEvolve, TBEvolve
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha

CONDITIONS = ("TB", "TB-noH", "TB-noCommit", "flat")


class FlatTriple(nn.Module):
    """Baseline: independent linear heads on the raw features."""

    def __init__(self, dim: int, n_entity: int, n_pred: int) -> None:
        super().__init__()
        self.subj = nn.Linear(dim, n_entity)
        self.obj = nn.Linear(dim, n_entity)
        self.pred = nn.Linear(2 * dim, n_pred)

    def forward(self, fs: Tensor, fo: Tensor) -> dict[str, Tensor]:
        return {"subject": self.subj(fs), "object": self.obj(fo),
                "predicate": self.pred(torch.cat([fs, fo], dim=-1))}


# --- data wiring -----------------------------------------------------------------

def triple_targets(table: PairTable, vs: VideoSpace):
    """(subj_g, obj_g, pred_g) per row + keep mask (participants must have an
    entity index; unmapped-category participants are skipped, counted by caller)."""
    subj, obj, keep = [], [], []
    for v, s, o in zip(table.videos, table.subj_ids, table.obj_ids):
        ks, ko = (v, s) in vs.entity_to_global, (v, o) in vs.entity_to_global
        keep.append(ks and ko)
        subj.append(vs.entity_to_global.get((v, s), -1))
        obj.append(vs.entity_to_global.get((v, o), -1))
    return torch.tensor(subj), torch.tensor(obj), torch.tensor(keep, dtype=torch.bool)


def pair_frame_keys(table: PairTable, keep: Tensor) -> list[tuple[str, int, int, int]]:
    """Group key per kept row: one (subject, object) pair at one frame. Rows
    sharing a key carry byte-identical features (same cached rows) — their
    spans overlap here, so the frame's predicate target is a set."""
    return [k for k, ok in zip(zip(table.videos, table.frames, table.subj_ids,
                                   table.obj_ids), keep.tolist()) if ok]


def group_eval(keys: list, pred_g: Tensor, eval_: Tensor
               ) -> tuple[list[int], list[set[int]]]:
    """One representative row per unique (pair, frame) group in the eval split,
    plus each group's true-predicate set. The set is the union over ALL rows
    (train and eval): it is a property of the annotation, not of the split."""
    true_sets: dict[tuple, set[int]] = defaultdict(set)
    for k, p in zip(keys, pred_g.tolist()):
        true_sets[k].add(p)
    rep: list[int] = []
    seen: set[tuple] = set()
    for i in eval_.nonzero().squeeze(1).tolist():
        if keys[i] not in seen:
            seen.add(keys[i])
            rep.append(i)
    return rep, [true_sets[keys[i]] for i in rep]


def split_rows(n: int, *, eval_frac: float, seed: int) -> tuple[Tensor, Tensor]:
    """Per-row random split. Nearby frames of one span land on both sides, so
    predicate numbers are dominated by the pair→predicate co-occurrence prior
    (the first real run put prior-DistMult at 0.85 p@1). Kept as the leakage
    *reference* condition; the primary split is `split_triples`."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_eval = int(round(eval_frac * n))
    eval_ = torch.zeros(n, dtype=torch.bool)
    eval_[perm[:n_eval]] = True
    return ~eval_, eval_


def split_triples(keys: list[tuple], *, eval_frac: float,
                  seed: int) -> tuple[Tensor, Tensor]:
    """Grouped split: hold out whole triples — every row of an eval triple
    (all its spans, all its frames) is unseen at training. This removes the
    pair→predicate co-occurrence shortcut entirely: at eval, the model has
    never seen THIS pair with THIS predicate. Entity indices remain trainable
    through the entities' other triples. Groups are added to eval until ~
    eval_frac of the rows are covered."""
    g = torch.Generator().manual_seed(seed)
    rows_of: dict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        rows_of[k].append(i)
    groups = sorted(rows_of)
    order = torch.randperm(len(groups), generator=g).tolist()
    eval_ = torch.zeros(len(keys), dtype=torch.bool)
    target = int(round(eval_frac * len(keys)))
    covered = 0
    for gi in order:
        if covered >= target:
            break
        rows = rows_of[groups[gi]]
        eval_[torch.tensor(rows)] = True
        covered += len(rows)
    return ~eval_, eval_


# --- model -----------------------------------------------------------------------

def build_tb(vs: VideoSpace, dim: int, *, evolve: str) -> TB:
    ev = TBEvolve(dim=dim, hidden=64) if evolve == "tb" else IdentityEvolve()
    return TB(dim=dim,
              index_layer=IndexLayer(num_indices=vs.groups.num_indices, dim=dim),
              evolve_module=ev, alpha=learnable_alpha(1.0), beta=1.0)


def sop_positions(masks, fs: Tensor, fo: Tensor, *, order: str = "sop",
                  teacher: tuple[Tensor, Tensor, Tensor] | None = None) -> list[Position]:
    ts, to, tp = teacher if teacher else (None, None, None)
    s_pos = Position(nu=fs, typed_mask=masks["entity"], teacher=ts)
    o_pos = Position(nu=fo, typed_mask=masks["entity"], teacher=to)
    p_pos = Position(nu=None, typed_mask=masks["predicate"], teacher=tp)
    first, second = (s_pos, o_pos) if order == "sop" else (o_pos, s_pos)
    return [first, second, p_pos]


def decode_sop(tb: TB, masks, fs: Tensor, fo: Tensor, *, commit: str,
               order: str = "sop", teacher=None):
    """Returns logits dict keyed subject/object/predicate regardless of order."""
    out = decode_chain(tb, sop_positions(masks, fs, fo, order=order, teacher=teacher),
                       commit=commit, batch_size=fs.shape[0], device=fs.device)
    i_s, i_o = (0, 1) if order == "sop" else (1, 0)
    return {"subject": out.positions[i_s].typed.logits,
            "object": out.positions[i_o].typed.logits,
            "predicate": out.positions[2].typed.logits}


# --- experiment ------------------------------------------------------------------

SLOTS = ("subject", "object", "predicate")


def run(table: PairTable, vs: VideoSpace | None = None, *, eval_frac: float = 0.3,
        seed: int = 0, steps: int = 400, batch: int = 2048, lr: float = 5e-2,
        cs=None, kg_model: str | None = None, kg_epochs: int = 100,
        kg_dim: int = 64, split: str = "row", device: str | None = None,
        log_step=None):
    torch.manual_seed(seed)
    dev = pick_device(device)
    if vs is None:
        cs = cs or build_concept_space()
        vs = build_video_space(videos=sorted(set(table.videos)), cs=cs)
    if kg_model and cs is None:
        cs = build_concept_space()
    masks = {"entity": vs.groups.mask("entity", device=dev),
             "predicate": vs.groups.mask("predicate", device=dev)}
    n_entity = vs.groups.sizes[vs.groups.names.index("entity")]
    ent_slice = vs.groups.slice_of("entity")
    prd_slice = vs.groups.slice_of("predicate")

    subj_g, obj_g, keep = triple_targets(table, vs)
    n_skipped = int((~keep).sum())
    fs, fo = table.feat_s[keep], table.feat_o[keep]
    tgt = {"subject": subj_g[keep], "object": obj_g[keep],
           "predicate": table.pred_g[keep]}
    if split == "row":
        train, eval_ = split_rows(fs.shape[0], eval_frac=eval_frac, seed=seed)
    elif split == "triple":
        vids_all = [v for v, ok in zip(table.videos, keep.tolist()) if ok]
        sub_all = [s for s, ok in zip(table.subj_ids, keep.tolist()) if ok]
        obj_all = [o for o, ok in zip(table.obj_ids, keep.tolist()) if ok]
        tkeys = list(zip(vids_all, sub_all, obj_all, tgt["predicate"].tolist()))
        train, eval_ = split_triples(tkeys, eval_frac=eval_frac, seed=seed)
    else:
        raise ValueError(f"unknown split {split!r}")
    if not train.any() or not eval_.any():
        raise RuntimeError("degenerate train/eval split")
    dim = fs.shape[1]

    models: dict[str, nn.Module] = {
        "TB": build_tb(vs, dim, evolve="tb"),
        "TB-noH": build_tb(vs, dim, evolve="identity"),
        "TB-noCommit": build_tb(vs, dim, evolve="tb"),
        "flat": FlatTriple(dim, n_entity,
                           vs.groups.sizes[vs.groups.names.index("predicate")]),
    }
    for m in models.values():
        m.to(dev)

    fs_tr, fo_tr = fs[train].to(dev), fo[train].to(dev)
    tgt_tr = {s: tgt[s][train].to(dev) for s in SLOTS}
    n_tr = fs_tr.shape[0]

    def train_model(name: str) -> list[float]:
        model = models[name]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 1)
        losses: list[float] = []
        step = 0
        while step < steps:
            for idx in torch.randperm(n_tr, generator=g).split(batch):
                if step >= steps:
                    break
                opt.zero_grad()
                if name == "flat":
                    logits = model(fs_tr[idx], fo_tr[idx])
                    loss = (F.cross_entropy(logits["subject"], tgt_tr["subject"][idx] - ent_slice.start)
                            + F.cross_entropy(logits["object"], tgt_tr["object"][idx] - ent_slice.start)
                            + F.cross_entropy(logits["predicate"], tgt_tr["predicate"][idx] - prd_slice.start))
                else:
                    teacher = (tgt_tr["subject"][idx], tgt_tr["object"][idx],
                               tgt_tr["predicate"][idx])
                    commit = "expectation" if name == "TB-noCommit" else "teacher"
                    if commit == "teacher":
                        logits = decode_sop(model, masks, fs_tr[idx], fo_tr[idx],
                                            commit="teacher", teacher=teacher)
                    else:
                        logits = decode_sop(model, masks, fs_tr[idx], fo_tr[idx],
                                            commit="expectation")
                    loss = sum(F.cross_entropy(logits[s], tgt_tr[s][idx]) for s in SLOTS)
                loss.backward()
                opt.step()
                losses.append(float(loss))
                if log_step is not None:
                    log_step({f"loss/{name}": float(loss)}, step)
                step += 1
        return losses

    losses = {name: train_model(name) for name in models}

    # --- evaluation ---------------------------------------------------------
    # Deduped to one row per (pair, frame) group; the predicate is scored as
    # Hits@k against the group's true set (see module docstring).
    keys = pair_frame_keys(table, keep)
    rep, ev_sets = group_eval(keys, tgt["predicate"], eval_)
    fs_ev, fo_ev = fs[rep].to(dev), fo[rep].to(dev)
    rep_t = torch.tensor(rep)
    tgt_ev = {s: tgt[s][rep_t] for s in ("subject", "object")}
    n_ev = len(rep)
    n_multi = sum(len(s) > 1 for s in ev_sets)
    results: dict[str, dict] = {}

    def batched_preds(name: str, order: str = "sop") -> dict[str, Tensor]:
        """subject/object argmax + predicate top-3 global indices (ranked)."""
        model = models[name]
        preds: dict[str, list] = {"subject": [], "object": [], "pred_top": []}
        with torch.no_grad():
            for s0 in range(0, n_ev, batch):
                fsb, fob = fs_ev[s0:s0 + batch], fo_ev[s0:s0 + batch]
                if name == "flat":
                    lg = model(fsb, fob)
                    preds["subject"].append(lg["subject"].argmax(-1) + ent_slice.start)
                    preds["object"].append(lg["object"].argmax(-1) + ent_slice.start)
                    preds["pred_top"].append(
                        lg["predicate"].topk(3, dim=-1).indices + prd_slice.start)
                else:
                    commit = "expectation" if name == "TB-noCommit" else "argmax"
                    lg = decode_sop(model, masks, fsb, fob, commit=commit, order=order)
                    preds["subject"].append(lg["subject"].argmax(-1))
                    preds["object"].append(lg["object"].argmax(-1))
                    preds["pred_top"].append(lg["predicate"].topk(3, dim=-1).indices)
        return {s: torch.cat(p).cpu() for s, p in preds.items()}

    def pred_metrics(top: Tensor) -> dict[str, float]:
        """hits1: top-1 in the true set (ceiling 1.0 even on multi-label
        frames). hits3: per gold predicate, gold in top-3 — the paper's
        Table-4-style Hits@k, averaged over (group, gold) pairs."""
        rows = top.tolist()
        hits1 = sum(r[0] in s for r, s in zip(rows, ev_sets)) / n_ev
        golds = [(g, r) for r, s in zip(rows, ev_sets) for g in s]
        hits3 = sum(g in r for g, r in golds) / len(golds)
        return {"hits1": hits1, "hits3": hits3}

    preds_by_name = {name: batched_preds(name) for name in models}
    for name, pred in preds_by_name.items():
        pm = pred_metrics(pred["pred_top"])
        results[name] = {"acc": {"subject": float((pred["subject"] == tgt_ev["subject"]).float().mean()),
                                 "object": float((pred["object"] == tgt_ev["object"]).float().mean()),
                                 "predicate": pm["hits1"]},
                         "pred_hits3": pm["hits3"]}

    # annotation-only KGE prior (PyKEEN): trained on the TRAIN rows' triples,
    # scored on the same eval groups — strictly less evidence (no features)
    if kg_model:
        from experiments.pvsg_hierarchy.kg_baseline import (
            predicate_scores,
            train_kge,
        )
        inv_pred = {g: lbl for (lvl, lbl), g in cs.label_to_global.items()
                    if lvl == "predicate"}
        vids_k = [v for v, ok in zip(table.videos, keep.tolist()) if ok]
        sub_k = [s for s, ok in zip(table.subj_ids, keep.tolist()) if ok]
        obj_k = [o for o, ok in zip(table.obj_ids, keep.tolist()) if ok]
        tr_rows = train.nonzero(as_tuple=True)[0].tolist()
        triples = [(f"{vids_k[i]}:{sub_k[i]}", inv_pred[int(tgt['predicate'][i])],
                    f"{vids_k[i]}:{obj_k[i]}") for i in tr_rows]
        kge, tf = train_kge(triples, model=kg_model, dim=kg_dim,
                            epochs=kg_epochs, seed=seed)
        pairs_ev = [(f"{vids_k[i]}:{sub_k[i]}", f"{vids_k[i]}:{obj_k[i]}")
                    for i in rep]
        sc, known = predicate_scores(kge, tf, pairs_ev)
        # re-align PyKEEN's relation columns to global predicate indices
        n_pred = prd_slice.stop - prd_slice.start
        sc_g = torch.full((len(rep), n_pred), -torch.inf)
        for g, lbl in inv_pred.items():
            col = tf.relation_to_id.get(lbl)
            if col is not None:
                sc_g[:, g - prd_slice.start] = sc[:, col]
        top = sc_g.topk(3, dim=-1).indices + prd_slice.start
        krows = known.nonzero(as_tuple=True)[0].tolist()
        hits1 = (sum(top[i, 0].item() in ev_sets[i] for i in krows)
                 / max(len(krows), 1))
        golds = [(g, top[i].tolist()) for i in krows for g in ev_sets[i]]
        hits3 = sum(g in r_ for g, r_ in golds) / max(len(golds), 1)
        results[f"prior-{kg_model}"] = {
            "acc": {"subject": float("nan"), "object": float("nan"),
                    "predicate": hits1},
            "pred_hits3": hits3, "n_unknown_pairs": int((~known).sum()),
        }

    # order-invariance on the full TB
    pred_sop = preds_by_name["TB"]
    pred_osp = batched_preds("TB", order="osp")
    results["TB"]["order"] = {
        "pred_agree": float((pred_sop["pred_top"][:, 0]
                             == pred_osp["pred_top"][:, 0]).float().mean()),
        "pred_acc_osp": pred_metrics(pred_osp["pred_top"])["hits1"],
    }

    return {"models": results, "n_rows": int(keep.sum()), "n_skipped": n_skipped,
            "n_train": int(train.sum()), "n_eval": n_ev,
            "n_eval_rows": int(eval_.sum()), "n_eval_multi": n_multi,
            "n_entity": n_entity, "losses": losses,
            "alphas": {n: float(m.alpha.detach()) for n, m in models.items()
                       if hasattr(m, "alpha")},
            "config": {"split": split, "eval_frac": eval_frac, "seed": seed,
                       "steps": steps, "batch": batch, "lr": lr,
                       "kg_model": kg_model, "kg_epochs": kg_epochs,
                       "kg_dim": kg_dim, "device": str(dev)}}


def print_report(r: dict) -> None:
    print(f"\nV2 — per-moment triple decoding (S → O → P) | "
          f"split: {r['config']['split']}, seed {r['config']['seed']}")
    print(f"  rows {r['n_rows']} (skipped unmapped-participant {r['n_skipped']}) | "
          f"train {r['n_train']} | eval {r['n_eval']} groups "
          f"from {r['n_eval_rows']} rows ({r['n_eval_multi']} multi-label) | "
          f"entities {r['n_entity']}\n")
    print(f"  {'condition':12} | {'subj':>5} {'obj':>5} {'p@1':>5} {'p@3':>5}")
    print("  " + "-" * 42)
    for name in CONDITIONS:
        m = r["models"][name]
        a = m["acc"]
        print(f"  {name:12} | {a['subject']:.2f}  {a['object']:.2f}  "
              f"{a['predicate']:.2f}  {m['pred_hits3']:.2f}")
    for name, m in r["models"].items():
        if name.startswith("prior-"):
            print(f"  {name:12} | {'-':>5} {'-':>5} {m['acc']['predicate']:.2f}  "
                  f"{m['pred_hits3']:.2f}   (triples only, no features; "
                  f"unknown pairs {m['n_unknown_pairs']})")
    o = r["models"]["TB"]["order"]
    print(f"\n  order-invariance (TB): predicate agreement S→O→P vs O→S→P "
          f"{o['pred_agree']:.2f} | O→S→P accuracy {o['pred_acc_osp']:.2f}")


def print_aggregate(runs: list[dict]) -> None:
    from experiments.pvsg_hierarchy.run_v1 import _mean_std
    seeds = [r["config"]["seed"] for r in runs]
    print(f"\n=== V2 aggregate over seeds {seeds} (mean ± std) ===")
    print(f"  {'condition':14} | {'p@1':>11} {'p@3':>11}")
    for name in runs[0]["models"]:
        if name not in runs[-1]["models"]:
            continue
        h1 = _mean_std([r["models"][name]["acc"]["predicate"] for r in runs])
        h3 = _mean_std([r["models"][name]["pred_hits3"] for r in runs])
        print(f"  {name:14} | {h1[0]:.3f}±{h1[1]:.3f} {h3[0]:.3f}±{h3[1]:.3f}")


def main() -> None:
    from experiments.pvsg_hierarchy.results_io import new_run_dir, save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--timelines", required=True)
    ap.add_argument("--split", choices=("row", "triple"), default="triple",
                    help="'triple' (primary; holds out whole triples) or "
                         "'row' (leakage reference)")
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--kg-model", default="DistMult",
                    help="PyKEEN model name for the annotation-only prior "
                         "baseline (e.g. DistMult, RESCAL, ComplEx); "
                         "'none' disables it")
    ap.add_argument("--kg-epochs", type=int, default=100)
    ap.add_argument("--kg-dim", type=int, default=64)
    ap.add_argument("--out", default="results/v2")
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--no-trackio", dest="trackio", action="store_false",
                    help="disable trackio logging (on by default)")
    args = ap.parse_args()

    cs = build_concept_space()
    spans = load_relation_spans()
    table = build_pair_table(spans, cs, args.cache, args.timelines,
                             frame_stride=args.frame_stride)
    print(table.stats.summary())
    print(f"device: {pick_device(args.device)}")
    kg = None if args.kg_model.lower() == "none" else args.kg_model
    out_dir = new_run_dir(args.out, tag=args.split)
    runs = []
    for seed in args.seeds:
        with track("pvsg-v2", f"{out_dir.name}-seed{seed}",
                   {**vars(args), "seed": seed}, enabled=args.trackio) as log:
            r = run(table, cs=cs, split=args.split, eval_frac=args.eval_frac,
                    seed=seed, steps=args.steps, batch=args.batch, kg_model=kg,
                    kg_epochs=args.kg_epochs, kg_dim=args.kg_dim,
                    device=args.device, log_step=log)
            log({**{f"p1/{n}": m["acc"]["predicate"]
                    for n, m in r["models"].items()},
                 **{f"p3/{n}": m["pred_hits3"] for n, m in r["models"].items()}})
        save_json(out_dir / f"seed{seed}.json", r)
        print_report(r)
        runs.append(r)
    if len(runs) > 1:
        print_aggregate(runs)
    print(f"\nsaved {len(runs)} run file(s) to {out_dir}")


if __name__ == "__main__":
    main()
