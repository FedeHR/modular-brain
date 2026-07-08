"""V2.1 — faithful BTN perception on PVSG (see V21_BTN_PERCEPTION.md).

Reimplements the TB paper's perception experiment (Algorithm 1, §6) with every
component chosen deliberately between the paper's *theory* and the official
BTN code (github.com/hangligit/BTN), which diverge at several points. The
choices, each documented in V21_BTN_PERCEPTION.md Part 2.6:

- decode pattern s → unary labels(s) → o → unary labels(o) → p, where unary
  (class-ladder) labels are POST-COMMIT READOUTS of the entity's q — never
  committed back. Algorithm 1 (samples c* without a commit) and the official
  code (group softmax on q_s) agree on this.
- every step has bottom-up input; the predicate step gets ν_p ≈ f(BB_pred).
- commit q ← dropout(q̃) + β·a with learnable scalar β on the embedding —
  the official code's form (`train_scale`); Algorithm 1 has β = 1.
- training chains INTEGRAL (expectation / semantic-attention) commits — the
  official `sampling_training='integral'`, endorsed by §5.3 ("in training …
  we use β = 1"). Teacher forcing is kept as a flag (--train-commit teacher).
- inference: P-SA (integral) and P-Samp (winner-take-all argmax), the paper's
  two modes, from the same checkpoint.
- dynamic context layer: the paper's recurrence (TBEvolve, sigmoids) at
  d_h = 64 ≈ r/6, matching the official width ratio (500/4096); the official
  code's own recurrence uses ReLUs and a pre-activation h-skip — we keep the
  theory form. d_h = r is kept as a width probe (BTN-wideH).
- readout nonlinearity sig(q) (paper §4.4); the official code uses
  LeakyReLU(0.02) — theory form kept.
- dropout 0.5 on the skip branch at commits and on post-commit readouts,
  applied after the readout nonlinearity as in the official
  `A(dropout2(f(q)))` (official `dropout_v`/`dropout2`; the paper never
  mentions dropout).

Conditions: BTN (faithful), BTN-wideH (d_h = r), BTN-noNuP (no predicate
input — V2's central infidelity), BTN-noClasses (no unary-label supervision),
P-Dir (the paper's direct-perception baseline, §6.3), flat (two-brain
concatenation reference), prior-<KGE> (annotation-only PyKEEN prior).

Data, splits (triple primary / row reference) and the multi-label group
evaluation are inherited from run_v2, which stays untouched.

    python -m experiments.pvsg_hierarchy.run_v21 --cache $WORK/cache --timelines $WORK/timelines
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.pvsg_hierarchy.device import pick_device
from experiments.pvsg_hierarchy.pvsg_data import (
    LEVELS,
    VideoSpace,
    build_concept_space,
    build_video_space,
    load_instances,
    load_relation_spans,
)
from experiments.pvsg_hierarchy.pvsg_pairs import PairTable, build_pair_table
from experiments.pvsg_hierarchy.run_v2 import (
    FlatTriple,
    group_eval,
    pair_frame_keys,
    split_rows,
    split_triples,
    triple_targets,
)
from experiments.pvsg_hierarchy.tracking import track
from tb.evolve import TBEvolve
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_beta

BTN_CONDITIONS = ("BTN", "BTN-wideH", "BTN-noNuP", "BTN-noClasses")
CONDITIONS = BTN_CONDITIONS + ("P-Dir", "flat")
INFERENCE_MODES = ("samp", "sa")          # P-Samp (argmax) / P-SA (integral)
COMMIT_OF_MODE = {"samp": "argmax", "sa": "integral"}
H_RATIO = 6                               # d_h = r/6 ≈ official 500/4096

CLASS_SLOTS = tuple(f"{e}_{lvl}" for e in ("s", "o") for lvl in LEVELS)
SLOTS = ("subject", "object", "predicate")


def nu_predicate(fs: Tensor, fo: Tensor) -> Tensor:
    """ν_p ≈ f(BB_pred): superposition of the participants' features, scaled
    to keep the √r feature norm (D2 in V21_BTN_PERCEPTION.md)."""
    return (fs + fo) / math.sqrt(2.0)


# --- class-ladder targets ----------------------------------------------------------

def class_targets(table: PairTable, vs: VideoSpace) -> tuple[Tensor, Tensor]:
    """Global leaf/mid/coarse indices of each row's subject and object,
    shape [M, 3] each. Every entity in `vs.entity_to_global` is taxonomy-mapped
    (same drop rule), so rows kept by `triple_targets` always resolve; unmapped
    participants get -1 here and are dropped by the caller's keep mask."""
    instances, _ = load_instances(cs=vs.cs)
    ladder = {(i.video, i.obj_id): tuple(vs.gindex(lvl, getattr(i, lvl))
                                         for lvl in LEVELS)
              for i in instances}
    missing = (-1,) * len(LEVELS)
    subj = [ladder.get((v, s), missing)
            for v, s in zip(table.videos, table.subj_ids)]
    obj = [ladder.get((v, o), missing)
           for v, o in zip(table.videos, table.obj_ids)]
    return torch.tensor(subj), torch.tensor(obj)


# --- models ------------------------------------------------------------------------

def build_btn(vs: VideoSpace, dim: int, *, hidden: int,
              learn_beta: bool = True) -> TB:
    """The BTN: index layer (tied A/Aᵀ), the paper's g(·) recurrence
    (TBEvolve), fixed skip (α = 1) and embedding scale β on the commit
    `q ← q̃ + β·a`. β learnable is the official code's `train_scale` (the
    default; settles ≈2.8 at lr 5e-3); β = 1 fixed is Algorithm 1's unscaled
    commit, kept as the theory-form probe. (At V2's lr 5e-2 the learnable β
    ran away to ≈5, an early symptom of the broken optimization.)"""
    beta = learnable_beta(1.0) if learn_beta else 1.0
    return TB(dim=dim,
              index_layer=IndexLayer(num_indices=vs.groups.num_indices, dim=dim),
              evolve_module=TBEvolve(dim=dim, hidden=hidden),
              alpha=1.0, beta=beta)


class DirectPerception(nn.Module):
    """P-Direct (§6.3): every label predicted independently from its own
    bounding-box features through the shared index layer — n = aᵀ sig(f(BB)).
    No commits, no dynamic context layer. Readout dropout mirrors the BTN's
    post-commit readout dropout for regularization parity."""

    def __init__(self, num_indices: int, dim: int, *, dropout: float = 0.5) -> None:
        super().__init__()
        self.index_layer = IndexLayer(num_indices=num_indices, dim=dim)
        self.dropout = dropout

    def forward(self, masks: dict[str, Tensor], fs: Tensor, fo: Tensor,
                nu_p: Tensor) -> dict[str, Tensor]:
        def read(nu: Tensor, mask: Tensor) -> Tensor:
            x = F.dropout(torch.sigmoid(nu), self.dropout, self.training)
            return self.index_layer(x).masked_fill(~mask, -torch.inf)

        out = {"subject": read(fs, masks["entity"]),
               "object": read(fo, masks["entity"]),
               "predicate": read(nu_p, masks["predicate"])}
        for lvl in LEVELS:
            out[f"s_{lvl}"] = read(fs, masks[lvl])
            out[f"o_{lvl}"] = read(fo, masks[lvl])
        return out


class FlatDrop(FlatTriple):
    """V2's flat baseline with matching input dropout (regularization parity)."""

    def __init__(self, *a, dropout: float = 0.5, **kw) -> None:
        super().__init__(*a, **kw)
        self.dropout = dropout

    def forward(self, fs: Tensor, fo: Tensor) -> dict[str, Tensor]:
        fs = F.dropout(fs, self.dropout, self.training)
        fo = F.dropout(fo, self.dropout, self.training)
        return super().forward(fs, fo)


def decode_perception(tb: TB, masks: dict[str, Tensor], fs: Tensor, fo: Tensor,
                      nu_p: Tensor | None, *, commit: str,
                      teacher: dict[str, Tensor] | None = None,
                      classes: bool = True, order: str = "sop",
                      dropout: float = 0.5) -> dict[str, Tensor]:
    """One perception episode per Algorithm 1 (lines annotated; episodic head
    omitted — D1). Written with explicit primitive calls so fidelity is
    auditable. Returns logits per slot, keyed by role regardless of order.

    commit ∈ {"integral", "argmax", "teacher"}:
      integral — a ← Σ_k p_k a_k, the official training mode and P-SA (§5.3);
      argmax   — winner-take-all, P-Samp (β → ∞);
      teacher  — a ← a_{ground truth} (requires `teacher`); MLE conditioning.

    Unary (class) labels are post-commit READOUTS of the entity's q — never
    committed (Algorithm 1 lines 21-22 and the official group softmax agree).
    Dropout: skip branch at commits (official dropout_v) and on post-commit
    readouts AFTER the nonlinearity, before the index layer — the official
    placement `A(dropout2(f(q)))`; the pre-commit entity readout is undropped
    (dropout1 = 0). Active only in train mode. (Dropping q *before* the
    sigmoid is not expectation-preserving — dropped units read sig(0) = 0.5
    and kept ones are rescaled inside the saturation — and produced a
    train/eval readout shift that collapsed predicate transfer.)
    """
    if commit == "teacher" and teacher is None:
        raise ValueError("commit='teacher' requires teacher indices")
    B, device = fs.shape[0], fs.device
    q = torch.zeros(B, tb.dim, device=device)
    state = tb.init_state(B, device)                      # line 5: h ← 0
    logits: dict[str, Tensor] = {}

    def read(q: Tensor, mask: Tensor, p: float = 0.0) -> Tensor:
        x = F.dropout(torch.sigmoid(q), p, tb.training)   # n_k = a_kᵀ drop(sig(q))
        return tb.index_layer(x).masked_fill(~mask, -torch.inf)

    def feedback(lg: Tensor, slot: str) -> Tensor:        # the committed a
        if commit == "teacher":
            return tb.index_layer.embed(teacher[slot])
        if commit == "integral":
            return lg.softmax(-1) @ tb.index_layer.weight
        if commit == "argmax":
            return tb.index_layer.embed(lg.argmax(-1))
        raise ValueError(f"unknown commit {commit!r}")

    entity_blocks = [("subject", "s", fs), ("object", "o", fo)]
    if order == "osp":
        entity_blocks.reverse()
    elif order != "sop":
        raise ValueError(f"unknown order {order!r}")

    for slot, prefix, nu in entity_blocks:
        q, state = tb.evolve(q, state)                    # lines 16/23: h-update
        q = tb.attend(q, nu=nu)                           # lines 17/24: + u·f(BB)
        lg = read(q, masks["entity"])                     # lines 18/25: n_S/n_O
        logits[slot] = lg
        a = feedback(lg, slot)                            # lines 19/26: commit
        q = F.dropout(q, dropout, tb.training) + tb.beta * a   # lines 20/27
        if classes:                                       # lines 21-22: unary
            for lvl in LEVELS:                            # labels read q_S/q_O;
                logits[f"{prefix}_{lvl}"] = read(q, masks[lvl], dropout)

    q, state = tb.evolve(q, state)                        # line 28: h-update
    q = tb.attend(q, nu=nu_p)                             # line 29: + u·f(BB_pred)
    logits["predicate"] = read(q, masks["predicate"], dropout)  # lines 30-32:
    return logits                                         # read n_P — no commit


# --- experiment ----------------------------------------------------------------

def run(table: PairTable, vs: VideoSpace | None = None, *, eval_frac: float = 0.3,
        seed: int = 0, steps: int = 400, batch: int = 2048, lr: float = 5e-3,
        cs=None, kg_model: str | None = None, kg_epochs: int = 100,
        kg_dim: int = 64, split: str = "triple", train_commit: str = "integral",
        dropout: float = 0.5, beta: str = "learned", nu: str = "sum",
        device: str | None = None, log_step=None, log_every: int = 25,
        dump_preds: str | None = None):
    torch.manual_seed(seed)
    dev = pick_device(device)
    if vs is None:
        cs = cs or build_concept_space()
        vs = build_video_space(videos=sorted(set(table.videos)), cs=cs)
    if kg_model and cs is None:
        cs = build_concept_space()
    masks = {"entity": vs.groups.mask("entity", device=dev),
             "predicate": vs.groups.mask("predicate", device=dev),
             **{lvl: vs.groups.mask(lvl, device=dev) for lvl in LEVELS}}
    n_entity = vs.groups.sizes[vs.groups.names.index("entity")]
    ent_slice = vs.groups.slice_of("entity")
    prd_slice = vs.groups.slice_of("predicate")

    subj_g, obj_g, keep = triple_targets(table, vs)
    scls, ocls = class_targets(table, vs)
    n_skipped = int((~keep).sum())
    fs, fo = table.feat_s[keep], table.feat_o[keep]
    if nu == "union":
        if table.feat_u is None:
            raise ValueError("nu='union' needs a union cache "
                             "(build_pair_table union_dir=...)")
        nu_p = table.feat_u[keep]                 # f(BB_pred), Algorithm 1 l.29
    elif nu == "sum":
        nu_p = nu_predicate(fs, fo)                # symmetric fallback ν = fs+fo
    else:
        raise ValueError(f"unknown nu mode {nu!r}")
    tgt = {"subject": subj_g[keep], "object": obj_g[keep],
           "predicate": table.pred_g[keep]}
    for i, lvl in enumerate(LEVELS):
        tgt[f"s_{lvl}"] = scls[keep, i]
        tgt[f"o_{lvl}"] = ocls[keep, i]
    assert all((tgt[s] >= 0).all() for s in CLASS_SLOTS)  # keep ⇒ taxonomy-mapped

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

    btn_cfg = {"BTN": dict(hidden=dim // H_RATIO, classes=True, nu_p=True),
               "BTN-wideH": dict(hidden=dim, classes=True, nu_p=True),
               "BTN-noNuP": dict(hidden=dim // H_RATIO, classes=True, nu_p=False),
               "BTN-noClasses": dict(hidden=dim // H_RATIO, classes=False,
                                     nu_p=True)}
    if beta not in ("learned", "fixed"):
        raise ValueError(f"unknown beta mode {beta!r}")
    models: dict[str, nn.Module] = {
        name: build_btn(vs, dim, hidden=c["hidden"],
                        learn_beta=beta == "learned")
        for name, c in btn_cfg.items()
    }
    models["P-Dir"] = DirectPerception(vs.groups.num_indices, dim, dropout=dropout)
    models["flat"] = FlatDrop(dim, n_entity,
                              vs.groups.sizes[vs.groups.names.index("predicate")],
                              dropout=dropout)
    for m in models.values():
        m.to(dev)

    fs_tr, fo_tr, nu_tr = fs[train].to(dev), fo[train].to(dev), nu_p[train].to(dev)
    tgt_tr = {s: tgt[s][train].to(dev) for s in tgt}
    n_tr = fs_tr.shape[0]

    def slots_of(name: str) -> tuple[str, ...]:
        if name in ("flat", "BTN-noClasses"):
            return SLOTS
        return SLOTS + CLASS_SLOTS

    n_models = len(models)
    total_steps = n_models * steps
    t0 = time.perf_counter()
    steps_done = 0                                   # global, across models
    diverged: dict[str, bool] = {}

    def train_model(name: str, mi: int) -> list[float]:
        nonlocal steps_done
        model = models[name]
        model.train()
        cfg = btn_cfg.get(name)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 1)
        losses: list[float] = []
        step = 0
        t_model = time.perf_counter()
        while step < steps:
            for idx in torch.randperm(n_tr, generator=g).split(batch):
                if step >= steps:
                    break
                opt.zero_grad()
                fsb, fob, nub = fs_tr[idx], fo_tr[idx], nu_tr[idx]
                if name == "flat":
                    lg = model(fsb, fob)
                    loss = (F.cross_entropy(lg["subject"], tgt_tr["subject"][idx] - ent_slice.start)
                            + F.cross_entropy(lg["object"], tgt_tr["object"][idx] - ent_slice.start)
                            + F.cross_entropy(lg["predicate"], tgt_tr["predicate"][idx] - prd_slice.start))
                elif name == "P-Dir":
                    lg = model(masks, fsb, fob, nub)
                    loss = sum(F.cross_entropy(lg[s], tgt_tr[s][idx])
                               for s in slots_of(name))
                else:
                    teacher = ({s: tgt_tr[s][idx] for s in SLOTS}
                               if train_commit == "teacher" else None)
                    lg = decode_perception(
                        model, masks, fsb, fob, nub if cfg["nu_p"] else None,
                        commit=train_commit, teacher=teacher,
                        classes=cfg["classes"], dropout=dropout)
                    loss = sum(F.cross_entropy(lg[s], tgt_tr[s][idx])
                               for s in slots_of(name))
                loss.backward()
                # uniform gradient-norm clipping: the dynamic context layer is
                # an RNN and its wide configuration explodes at this lr without
                # it. Applied to every condition identically (no confound). The
                # returned pre-clip norm is our divergence signal.
                gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()
                lval = loss.item()
                losses.append(lval)
                step += 1
                steps_done += 1
                if not math.isfinite(lval) and not diverged.get(name):
                    diverged[name] = True
                    print(f"  !! {name}: non-finite loss at step {step} "
                          f"(gnorm {gnorm:.1f}) — training is diverging",
                          flush=True)
                if step % log_every == 0 or step == steps:
                    recent = losses[-log_every:]
                    sm = sum(recent) / len(recent)
                    elapsed = time.perf_counter() - t0
                    rate = steps_done / elapsed
                    eta = (total_steps - steps_done) / max(rate, 1e-9)
                    print(f"  [{mi}/{n_models}] {name:13} "
                          f"{step:>4}/{steps} | loss {sm:8.3f} | "
                          f"gnorm {gnorm:7.2f} | {rate:5.1f} it/s | "
                          f"ETA {eta / 60:4.1f}m", flush=True)
                    if log_step is not None:
                        log_step({f"loss/{name}": sm, f"gnorm/{name}": gnorm,
                                  "progress/it_per_s": rate,
                                  "progress/eta_min": eta / 60}, steps_done)
        dt = time.perf_counter() - t_model
        print(f"  [{mi}/{n_models}] {name:13} done in {dt / 60:4.1f}m | "
              f"loss {losses[0]:.2f} -> {losses[-1]:.2f}", flush=True)
        return losses

    print(f"training {n_models} models x {steps} steps "
          f"(batch {batch}, {n_tr} train rows) on {dev}", flush=True)
    losses = {name: train_model(name, i + 1)
              for i, name in enumerate(models)}
    print(f"training done in {(time.perf_counter() - t0) / 60:.1f}m; "
          f"evaluating on {int(eval_.sum())} eval rows...", flush=True)
    t_eval = time.perf_counter()

    # --- evaluation (deduped multi-label groups, as V2) ----------------------
    keys = pair_frame_keys(table, keep)
    rep, ev_sets = group_eval(keys, tgt["predicate"], eval_)
    rep_t = torch.tensor(rep)
    fs_ev, fo_ev, nu_ev = (fs[rep_t].to(dev), fo[rep_t].to(dev),
                           nu_p[rep_t].to(dev))
    tgt_ev = {s: tgt[s][rep_t] for s in tgt if s != "predicate"}
    n_ev = len(rep)
    n_multi = sum(len(s) > 1 for s in ev_sets)
    results: dict[str, dict] = {}

    def batched_logits(name: str, mode: str, order: str = "sop"
                       ) -> dict[str, Tensor]:
        model = models[name]
        model.eval()
        cfg = btn_cfg.get(name)
        outs: dict[str, list] = {}
        with torch.no_grad():
            for s0 in range(0, n_ev, batch):
                fsb, fob = fs_ev[s0:s0 + batch], fo_ev[s0:s0 + batch]
                nub = nu_ev[s0:s0 + batch]
                if name == "flat":
                    lg = model(fsb, fob)
                elif name == "P-Dir":
                    lg = model(masks, fsb, fob, nub)
                else:
                    lg = decode_perception(
                        model, masks, fsb, fob, nub if cfg["nu_p"] else None,
                        commit=COMMIT_OF_MODE[mode], classes=cfg["classes"],
                        order=order, dropout=dropout)
                for s, t in lg.items():
                    outs.setdefault(s, []).append(t.cpu())   # free GPU per chunk
        return {s: torch.cat(t) for s, t in outs.items()}

    def offset_of(name: str, slot: str) -> int:
        """flat emits group-local logits; everything else global ones."""
        if name != "flat":
            return 0
        return ent_slice.start if slot in ("subject", "object") else prd_slice.start

    def metrics(name: str, lg: dict[str, Tensor]) -> dict:
        acc = {}
        for slot in ("subject", "object", *(s for s in CLASS_SLOTS if s in lg)):
            pred = lg[slot].argmax(-1) + offset_of(name, slot)
            acc[slot] = float((pred == tgt_ev[slot]).float().mean())
        top = lg["predicate"].topk(3, dim=-1).indices + offset_of(name, "predicate")
        rows = top.tolist()
        hits1 = sum(r[0] in s for r, s in zip(rows, ev_sets)) / n_ev
        golds = [(g, r) for r, s in zip(rows, ev_sets) for g in s]
        hits3 = sum(g in r for g, r in golds) / len(golds)
        out = {"acc": {"subject": acc["subject"], "object": acc["object"],
                       "predicate": hits1},
               "pred_hits3": hits3}
        cls = {s: acc[s] for s in CLASS_SLOTS if s in acc}
        if cls:
            out["class_acc"] = cls
            out["class_acc_mean"] = sum(cls.values()) / len(cls)
        return out

    for name in models:
        if name in btn_cfg:
            for mode in INFERENCE_MODES:
                results[f"{name}/{mode}"] = metrics(name, batched_logits(name, mode))
        else:
            results[name] = metrics(name, batched_logits(name, "samp"))

    # order-invariance of the faithful BTN under P-Samp
    lg_sop = batched_logits("BTN", "samp")
    lg_osp = batched_logits("BTN", "samp", order="osp")
    results["BTN/samp"]["order"] = {
        "pred_agree": float((lg_sop["predicate"].argmax(-1)
                             == lg_osp["predicate"].argmax(-1)).float().mean()),
        "pred_acc_osp": metrics("BTN", lg_osp)["acc"]["predicate"],
    }

    print(f"eval done in {(time.perf_counter() - t_eval) / 60:.1f}m", flush=True)

    # annotation-only KGE prior (identical to run_v2's block)
    if kg_model:
        print(f"training {kg_model} KGE prior...", flush=True)
        t_kge = time.perf_counter()
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
        print(f"KGE prior done in {(time.perf_counter() - t_kge) / 60:.1f}m",
              flush=True)

    if dump_preds:
        _dump_predictions(dump_preds, models, batched_logits, keys, rep,
                          ev_sets, prd_slice, cs, split, seed)

    peak_gb = (torch.cuda.max_memory_allocated(dev) / 1e9
               if dev.type == "cuda" else None)
    if peak_gb is not None:
        print(f"  peak GPU: {peak_gb:.2f} GB", flush=True)

    return {"models": results, "n_rows": int(keep.sum()), "n_skipped": n_skipped,
            "peak_gpu_gb": peak_gb,
            "n_train": int(train.sum()), "n_eval": n_ev,
            "n_eval_rows": int(eval_.sum()), "n_eval_multi": n_multi,
            "n_entity": n_entity, "losses": losses,
            "betas": {n: float(m.beta.detach()) for n, m in models.items()
                      if hasattr(m, "beta")},
            "param_counts": {n: sum(p.numel() for p in m.parameters())
                             for n, m in models.items()},
            "config": {"split": split, "eval_frac": eval_frac, "seed": seed,
                       "steps": steps, "batch": batch, "lr": lr,
                       "train_commit": train_commit, "dropout": dropout,
                       "beta": beta, "nu": nu,
                       "kg_model": kg_model, "kg_epochs": kg_epochs,
                       "kg_dim": kg_dim, "device": str(dev)}}


# --- per-pair prediction dump (for qualitative panels) -------------------------

def _dump_predictions(path, models, batched_logits, keys, rep, ev_sets,
                      prd_slice, cs, split, seed) -> None:
    """Write one JSON record per eval group: its (video, frame, subject,
    object) identity, the true-predicate SET, and BTN vs P-Direct top-3
    predicted predicates. Consumed by qual_v21 to draw grounded panels.

    Correctness anchors: `keys[rep[i]]` is the (video, frame, subj_id, obj_id)
    of eval group i (same indexing group_eval used to build ev_sets). BTN and
    P-Direct read predicates over the FULL index layer (masked to the predicate
    group), so their top-k indices are already global — no slice shift (matches
    metrics()'s offset_of == 0 for every non-flat condition). prd_slice is kept
    only as a sanity bound.
    """
    import json

    cs = cs or build_concept_space()
    inv_pred = {g: lbl for (lvl, lbl), g in cs.label_to_global.items()
                if lvl == "predicate"}
    instances, _ = load_instances(cs=cs)
    leaf_of = {(i.video, i.obj_id): i.leaf for i in instances}

    def top3(name):
        lg = batched_logits(name, "samp")["predicate"]        # [n_ev, n_index]
        idx = lg.topk(3, dim=-1).indices                      # already global
        assert (idx >= prd_slice.start).all() and (idx < prd_slice.stop).all(), \
            "predicate top-k fell outside the predicate group"
        return idx.tolist()
    btn3, pdir3 = top3("BTN"), top3("P-Dir")

    recs = []
    for i, (r, tset) in enumerate(zip(rep, ev_sets)):
        video, frame, sid, oid = keys[r]
        gold = sorted(tset)
        recs.append({
            "video": video, "frame": int(frame),
            "subj_id": int(sid), "obj_id": int(oid),
            "subj_leaf": leaf_of.get((video, sid)),
            "obj_leaf": leaf_of.get((video, oid)),
            "true_preds": [inv_pred[g] for g in gold],
            "btn_top3": [inv_pred[g] for g in btn3[i]],
            "pdir_top3": [inv_pred[g] for g in pdir3[i]],
            "btn_correct": btn3[i][0] in tset,
            "pdir_correct": pdir3[i][0] in tset,
        })
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"split": split, "seed": seed, "n": len(recs), "records": recs}, indent=0))
    n_ok = sum(r["btn_correct"] for r in recs)
    print(f"dumped {len(recs)} eval-pair predictions to {path} "
          f"(BTN top-1 correct {n_ok}/{len(recs)})", flush=True)


# --- reporting -----------------------------------------------------------------

def report_rows(r: dict) -> list[str]:
    """Ordered result keys: BTN conditions × modes, then baselines, prior."""
    rows = [f"{c}/{m}" for c in BTN_CONDITIONS for m in INFERENCE_MODES]
    rows += ["P-Dir", "flat"]
    rows += [k for k in r["models"] if k.startswith("prior-")]
    return rows


def print_report(r: dict) -> None:
    print(f"\nV2.1 — faithful BTN perception | split: {r['config']['split']}, "
          f"seed {r['config']['seed']}, train-commit {r['config']['train_commit']}")
    print(f"  rows {r['n_rows']} (skipped {r['n_skipped']}) | "
          f"train {r['n_train']} | eval {r['n_eval']} groups "
          f"({r['n_eval_multi']} multi-label) | entities {r['n_entity']}\n")
    print(f"  {'condition':18} | {'subj':>5} {'obj':>5} {'class':>5} "
          f"{'p@1':>5} {'p@3':>5}")
    print("  " + "-" * 54)
    for name in report_rows(r):
        m = r["models"][name]
        a = m["acc"]
        cls = m.get("class_acc_mean")
        cls_s = f"{cls:.2f}" if cls is not None else "    -"
        print(f"  {name:18} | {a['subject']:.2f}  {a['object']:.2f}  {cls_s}  "
              f"{a['predicate']:.2f}  {m['pred_hits3']:.2f}")
    o = r["models"]["BTN/samp"]["order"]
    print(f"\n  order-invariance (BTN, P-Samp): agreement S→O→P vs O→S→P "
          f"{o['pred_agree']:.2f} | O→S→P p@1 {o['pred_acc_osp']:.2f}")
    print("  betas: " + ", ".join(f"{n} {v:.2f}" for n, v in r["betas"].items()))


def print_aggregate(runs: list[dict]) -> None:
    from experiments.pvsg_hierarchy.run_v1 import _mean_std
    seeds = [r["config"]["seed"] for r in runs]
    print(f"\n=== V2.1 aggregate over seeds {seeds} (mean ± std) ===")
    print(f"  {'condition':18} | {'p@1':>11} {'p@3':>11}")
    for name in report_rows(runs[0]):
        if any(name not in r["models"] for r in runs):
            continue
        h1 = _mean_std([r["models"][name]["acc"]["predicate"] for r in runs])
        h3 = _mean_std([r["models"][name]["pred_hits3"] for r in runs])
        print(f"  {name:18} | {h1[0]:.3f}±{h1[1]:.3f} {h3[0]:.3f}±{h3[1]:.3f}")


def main() -> None:
    from experiments.pvsg_hierarchy.results_io import new_run_dir, save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--timelines", required=True)
    ap.add_argument("--split", choices=("row", "triple"), default="triple")
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-3,
                    help="Adam learning rate. V2's 5e-2 breaks the nine-head "
                         "sequential decode (entity heads never leave chance); "
                         "5e-3 converges cleanly — see V21_BTN_PERCEPTION.md")
    ap.add_argument("--train-commit", choices=("integral", "teacher"),
                    default="integral",
                    help="training feedback: 'integral' (official BTN, §5.3) "
                         "or 'teacher' (MLE conditioning / exposure-bias probe)")
    ap.add_argument("--dropout", type=float, default=0.5,
                    help="official dropout_v/dropout2 rate (0 disables)")
    ap.add_argument("--beta", choices=("learned", "fixed"), default="learned",
                    help="commit scale: 'learned' (official train_scale) or "
                         "'fixed' β = 1 (Algorithm 1's unscaled commit)")
    ap.add_argument("--nu", choices=("sum", "union"), default="sum",
                    help="predicate bottom-up input ν_p: 'sum' (f_s+f_o) or "
                         "'union' (f(BB_pred) from --union-dir, Algorithm 1 l.29)")
    ap.add_argument("--union-dir", default=None,
                    help="precompute_unions cache dir; required for --nu union")
    ap.add_argument("--kg-model", default="DistMult",
                    help="PyKEEN model for the annotation-only prior; "
                         "'none' disables it")
    ap.add_argument("--kg-epochs", type=int, default=100)
    ap.add_argument("--kg-dim", type=int, default=64)
    ap.add_argument("--out", default="results/v21")
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--log-every", type=int, default=25,
                    help="steps between progress lines (loss/gnorm/it-per-s/ETA)"
                         " to stdout and trackio")
    ap.add_argument("--dump-preds", default=None,
                    help="write per-eval-pair BTN/P-Dir predictions to this JSON "
                         "(one file per run; seed appended) for qual_v21 panels")
    ap.add_argument("--no-trackio", dest="trackio", action="store_false",
                    help="disable trackio logging (on by default)")
    args = ap.parse_args()
    if args.nu == "union" and not args.union_dir:
        ap.error("--nu union requires --union-dir")

    cs = build_concept_space()
    spans = load_relation_spans()
    table = build_pair_table(spans, cs, args.cache, args.timelines,
                             frame_stride=args.frame_stride,
                             union_dir=args.union_dir)
    print(table.stats.summary())
    print(f"device: {pick_device(args.device)}")
    kg = None if args.kg_model.lower() == "none" else args.kg_model
    out_dir = new_run_dir(args.out, tag=f"{args.split}_{args.train_commit}")
    runs = []
    for seed in args.seeds:
        with track("pvsg-v21", f"{out_dir.name}-seed{seed}",
                   {**vars(args), "seed": seed}, enabled=args.trackio) as log:
            r = run(table, cs=cs, split=args.split, eval_frac=args.eval_frac,
                    seed=seed, steps=args.steps, batch=args.batch, lr=args.lr,
                    train_commit=args.train_commit, dropout=args.dropout,
                    beta=args.beta, nu=args.nu, kg_model=kg,
                    kg_epochs=args.kg_epochs,
                    kg_dim=args.kg_dim, device=args.device, log_step=log,
                    log_every=args.log_every,
                    dump_preds=(f"{args.dump_preds}.seed{seed}.json"
                                if args.dump_preds else None))
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
