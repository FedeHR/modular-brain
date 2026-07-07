"""V1 — entity re-identification on real video (Table 5 analog, honest data).

The TB paper's VRD-EX "known entities" were affine warps of the training image,
so entity recognition was circular. Here an entity index is a *tracked instance*
in a PVSG video: the model trains on a subset of that instance's frames and must
recognize it in held-out frames — real viewpoint, lighting, and deformation
change, no synthetic warps.

Task
----
Given one frame's DINO region feature, decode entity → leaf → mid → coarse
through the TB (entity first, then the concept ladder, threading the committed
q). Both index kinds live in one `VideoSpace`, so entity recognition and
classification share the representation layer — the paper's setup.

Two honest deviations from the paper, to be stated in any write-up:
- The paper frames entity recognition as decoding a `(s', sameAs, s*)`
  statement after perception; here the entity index is decoded *directly* as
  the first measurement. Functionally close, but not a literal reproduction of
  the paper's decode pattern.
- The evolve module is never invoked — V1 is a static per-frame task; the
  dynamic context layer is V2's business. The TB therefore uses
  `IdentityEvolve` (zero parameters), and `live_param_counts` reports only
  gradient-receiving parameters (the shared index layer also carries
  predicate/episode rows V1 never trains).

Baselines: `flat` (linear head per group on the same features — does TB
structure add anything over linear readout?) and `ncm` (nearest class mean,
zero parameters — the classical template-matching / few-shot re-ID baseline).
KG-embedding methods don't apply to V1 (no relational structure in the task);
a DistMult-style scorer is the planned extra baseline for V2.

Split: **frames within instance** (unlike H1's split *by* instance — here the
entity index must be trainable, so every kept instance contributes train
frames). Instances with < `min_frames` cached frames are excluded and counted.
Three split modes control how hard re-identification is:

- random  : each frame coin-flipped into train/eval. Eval frames sit 1-2 frames
            from a train frame of the same instance, so this measures near-
            duplicate matching — a sanity ceiling, not re-identification.
- blocked : train on ONE contiguous chunk (`train_frac` of the instance's
            frames, random start), eval on everything outside it. Eval frames
            can be far from all training evidence; the accuracy-vs-distance
            curve is the headline readout.
- fewshot : the extreme of blocked — train on `k` contiguous frames (one brief
            "encounter"), eval on the rest of the instance's timeline.

Decode policies at eval (one trained TB, teacher-forced training):
- P-Direct : every level measured from the same attended q — no feedback.
- P-Samp   : committed winner-take-all, q ← αq + βa_k threaded level to level.
- P-SA     : soft (expectation) readout threaded — QTB's ignorant Y-measurement.

Fairness: the flat baseline gets the identical features and one linear head per
group (entity head included), trained with the same loss/steps/optimizer.

Metrics: per-level accuracy; entity↔leaf agreement (decoded leaf == class of
decoded entity); entity accuracy vs. temporal distance from the nearest train
frame of the same instance (row units, or true annotated frames with
`--timelines`); and two scene-confound diagnostics — `scene_acc` (does the
predicted entity at least belong to the true video?) and `within_video_acc`
(argmax restricted to the true video's entities: pure within-scene
discrimination, immune to scene identification doing the work).

    python -m experiments.pvsg_hierarchy.run_v1 --cache $WORK/cache
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments.pvsg_hierarchy.device import pick_device
from experiments.pvsg_hierarchy.mask_timelines import cache_row_frames
from experiments.pvsg_hierarchy.pvsg_data import VideoSpace, build_video_space
from experiments.pvsg_hierarchy.pvsg_features import LEVELS, FeatureTable, load_feature_table
from experiments.pvsg_hierarchy.results_io import new_run_dir, save_json
from experiments.pvsg_hierarchy.run_h1 import FlatHeads
from experiments.pvsg_hierarchy.tracking import track
from tb.evolve import IdentityEvolve
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha

ORDER = ("entity", *LEVELS)               # decode order: who, then what
SAMEAS_ORDER = (*LEVELS, "entity")        # paper-style: perceive what, then who
DECODE_ORDERS = {"entity-first": ORDER, "sameas": SAMEAS_ORDER}
POLICIES = ("direct", "samp", "sa")
DIST_EDGES = (1, 2, 4, 8, 16, 32, 64, 128)  # temporal-distance buckets


# --- data wiring -----------------------------------------------------------------

def row_positions(instance_id: Tensor) -> Tensor:
    """Temporal position of each row *within its instance* (rows of one instance
    are appended in frame order by `load_feature_table`)."""
    pos = torch.zeros_like(instance_id)
    for i in instance_id.unique():
        rows = (instance_id == i).nonzero(as_tuple=True)[0]
        pos[rows] = torch.arange(len(rows))
    return pos


SPLIT_MODES = ("random", "blocked", "fewshot")


def split_frames(instance_id: Tensor, *, train_frac: float, min_frames: int,
                 seed: int, mode: str = "random",
                 k: int = 3) -> tuple[Tensor, Tensor, int]:
    """Per-instance frame split (see module docstring for the three modes).
    Instances with < min_frames rows are excluded from both sides (returned as
    `n_excluded`). Rows of one instance are in frame order, so a contiguous row
    window is a contiguous stretch of its timeline. Returns boolean row masks
    (train, eval)."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"unknown split mode {mode!r}")
    g = torch.Generator().manual_seed(seed)
    train = torch.zeros(len(instance_id), dtype=torch.bool)
    eval_ = torch.zeros(len(instance_id), dtype=torch.bool)
    n_excluded = 0
    for i in instance_id.unique():
        rows = (instance_id == i).nonzero(as_tuple=True)[0]
        n = len(rows)
        if n < min_frames:
            n_excluded += 1
            continue
        if mode == "random":
            perm = rows[torch.randperm(n, generator=g)]
            n_train = max(1, int(round(train_frac * n)))
            train[perm[:n_train]] = True
            eval_[perm[n_train:]] = True
        else:
            w = max(1, int(round(train_frac * n))) if mode == "blocked" else k
            w = min(w, n - 1)                       # always leave eval rows
            start = int(torch.randint(0, n - w + 1, (1,), generator=g))
            train[rows[start:start + w]] = True
            eval_[rows[:start]] = True
            eval_[rows[start + w:]] = True
    return train, eval_, n_excluded


def true_positions(table: FeatureTable, timelines_dir: str | Path, *,
                   mask_stride: int = 1) -> tuple[Tensor, Tensor]:
    """Temporal position of each row in *annotated-frame* units, recovered from
    the mask timelines (k-th cached row = k-th visible strided frame). Unlike
    `row_positions`, gaps where the object was invisible count at full width.

    Returns (pos, ok). `ok` is False for every row of an instance whose cached
    row count disagrees with the timeline: the extractor drops a visible frame
    when the object shrinks below one patch (`pool_region` → None) or the frame
    JPG is missing, and without per-row frame indices in the cache (planned for
    the next extraction) such instances cannot be aligned — callers exclude and
    count them. If more than half the instances misalign, the cache itself is
    stale or the stride is wrong, and we raise instead."""
    pos = torch.zeros(len(table.instance_id), dtype=torch.long)
    ok = torch.ones(len(table.instance_id), dtype=torch.bool)
    timelines: dict[str, dict] = {}
    n_inst = n_bad = 0
    for i in table.instance_id.unique():
        n_inst += 1
        rows = (table.instance_id == i).nonzero(as_tuple=True)[0]
        video, oid = table.videos[int(rows[0])], table.obj_ids[int(rows[0])]
        tl = timelines.get(video)
        if tl is None:
            tl = torch.load(Path(timelines_dir) / f"{video}.pt", map_location="cpu")
            timelines[video] = tl
        area = tl["areas"].get(oid)
        frames = None if area is None else cache_row_frames(area, mask_stride=mask_stride)
        if frames is None or len(frames) != len(rows):
            ok[rows] = False
            n_bad += 1
            continue
        pos[rows] = frames
    if n_bad > n_inst / 2:
        raise ValueError(
            f"{n_bad}/{n_inst} instances misaligned between cache and timelines "
            f"— wrong mask_stride or stale cache")
    return pos, ok


def eval_train_distances(instance_id: Tensor, pos: Tensor, train: Tensor,
                         eval_: Tensor) -> Tensor:
    """For each eval row, distance (row units) to the nearest train frame of the
    same instance. Returns a tensor aligned with the eval rows' order."""
    train_pos: dict[int, Tensor] = defaultdict(lambda: torch.empty(0, dtype=torch.long))
    for i in instance_id[train].unique():
        train_pos[int(i)] = pos[train & (instance_id == i)]
    out = []
    for i, p in zip(instance_id[eval_].tolist(), pos[eval_].tolist()):
        out.append(int((train_pos[i] - p).abs().min()))
    return torch.tensor(out)


def entity_targets(table: FeatureTable, vs: VideoSpace) -> Tensor:
    """Global entity index per row."""
    return torch.tensor([vs.entity_gindex(v, o)
                         for v, o in zip(table.videos, table.obj_ids)])


# --- model -----------------------------------------------------------------------

def tb_decode_v1(tb: TB, masks: dict[str, Tensor], feat: Tensor, policy: str,
                 teacher: dict[str, Tensor] | None = None,
                 order: tuple[str, ...] = ORDER):
    """Decode the index groups in `order` under one policy. `teacher` supplies
    ground-truth commits for training ('teacher' policy). ORDER is V1's
    entity-first pattern; SAMEAS_ORDER approximates the paper's `(s', sameAs,
    s*)` pattern — perceive the concept ladder first, then measure the entity
    from the concept-committed q (the sameAs statement's object slot)."""
    q = tb.attend(torch.zeros(feat.shape[0], tb.dim, device=feat.device), nu=feat)
    logits: dict[str, Tensor] = {}
    preds: dict[str, Tensor] = {}
    for lvl in order:
        if policy == "direct":
            out = tb.measure(q, mask=masks[lvl], commit="expectation")
            # q is deliberately NOT threaded: every level sees only perception
        elif policy == "samp":
            out = tb.measure(q, mask=masks[lvl], commit="argmax")
            q = out.q
        elif policy == "sa":
            out = tb.measure(q, mask=masks[lvl], commit="expectation")
            q = out.q
        elif policy == "teacher":
            out = tb.measure(q, mask=masks[lvl], commit="teacher", teacher_k=teacher[lvl])
            q = out.q
        else:
            raise ValueError(f"unknown policy {policy!r}")
        logits[lvl] = out.logits
        preds[lvl] = out.logits.argmax(-1)
    return logits, preds


def build_models(vs: VideoSpace, dim: int) -> tuple[TB, FlatHeads]:
    groups = vs.groups
    # IdentityEvolve: V1 never calls evolve (static per-frame task), so the TB
    # carries no dead recurrence parameters.
    tb = TB(dim=dim,
            index_layer=IndexLayer(num_indices=groups.num_indices, dim=dim),
            evolve_module=IdentityEvolve(),
            alpha=learnable_alpha(1.0), beta=1.0)
    sizes = {lvl: groups.sizes[groups.names.index(lvl)] for lvl in ORDER}
    flat = FlatHeads(dim, sizes)
    return tb, flat


def ncm_predictions(feats_tr: Tensor, tgt_tr: dict[str, Tensor], feats_ev: Tensor,
                    *, batch: int) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor]]:
    """Nearest-class-mean template matcher (zero parameters): per level, each
    class is its train-feature mean; predict by dot product (features are
    L2-normalized, so this is cosine). Returns (preds by level, and the entity
    (classes, prototypes) pair for restricted scoring)."""
    preds: dict[str, Tensor] = {}
    ent_protos: tuple[Tensor, Tensor] | None = None
    for lvl in ORDER:
        classes = tgt_tr[lvl].unique()
        remap = torch.searchsorted(classes, tgt_tr[lvl])
        protos = torch.zeros(len(classes), feats_tr.shape[1])
        protos.index_add_(0, remap, feats_tr)
        protos /= torch.bincount(remap, minlength=len(classes)).clamp(min=1)[:, None].float()
        out = [classes[(feats_ev[s:s + batch] @ protos.T).argmax(-1)]
               for s in range(0, feats_ev.shape[0], batch)]
        preds[lvl] = torch.cat(out)
        if lvl == "entity":
            ent_protos = (classes, protos)
    assert ent_protos is not None
    return preds, ent_protos


def within_video_accuracy(ent_scores_fn, feats_ev: Tensor, tgt_ent: Tensor,
                          videos_ev: list[str], vs: VideoSpace, ent_slice,
                          *, batch: int) -> float:
    """Entity accuracy with the argmax restricted to the true video's entities:
    pure within-scene discrimination — scene identification cannot help here."""
    cols_of: dict[str, list[int]] = defaultdict(list)
    for (v, _), g in vs.entity_to_global.items():
        cols_of[v].append(g - ent_slice.start)
    rows_of: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(videos_ev):
        rows_of[v].append(i)
    n_correct = 0
    for v, rows in rows_of.items():
        cols = torch.tensor(sorted(cols_of[v]))
        rows_t = torch.tensor(rows)
        for s in range(0, len(rows_t), batch):
            idx = rows_t[s:s + batch]
            sc = ent_scores_fn(feats_ev[idx]).cpu()[:, cols]
            pred = cols[sc.argmax(-1)] + ent_slice.start
            n_correct += int((pred == tgt_ent[idx]).sum())
    return n_correct / len(videos_ev)


def collect_examples(video: str, videos_ev: list[str], pos_ev: Tensor,
                     obj_ids_ev: list[int], dists: Tensor,
                     model_preds: dict[str, dict[str, Tensor]],
                     tgt_ev: dict[str, Tensor], vs: VideoSpace, cs,
                     topk_by_row: dict[int, dict] | None = None) -> list[dict]:
    """Per-eval-frame prediction record for one video — quantitative thesis
    examples, saved with the run. `topk_by_row` optionally attaches the TB's
    top-k decoded labels + probabilities per level (grounding figures)."""
    inv_leaf = {g: lbl for (lvl, lbl), g in cs.label_to_global.items() if lvl == "leaf"}
    ent_obj = {g: o for (v, o), g in vs.entity_to_global.items()}
    ent_vid = {g: v for (v, o), g in vs.entity_to_global.items()}
    out = []
    for i, v in enumerate(videos_ev):
        if v != video:
            continue
        rec = {"frame": int(pos_ev[i]), "dist": int(dists[i]),
               "true_obj": obj_ids_ev[i],
               "true_entity_g": int(tgt_ev["entity"][i]),
               "true_leaf": inv_leaf[int(tgt_ev["leaf"][i])], "preds": {}}
        if topk_by_row and i in topk_by_row:
            rec["tb_topk"] = topk_by_row[i]
        for name in ("TB-samp", "flat", "ncm"):
            pe = int(model_preds[name]["entity"][i])
            rec["preds"][name] = {
                "obj": ent_obj[pe], "video": ent_vid[pe],
                "leaf": inv_leaf[int(model_preds[name]["leaf"][i])],
                "entity_correct": pe == int(tgt_ev["entity"][i]),
            }
        out.append(rec)
    return sorted(out, key=lambda r: r["frame"])


# --- experiment ------------------------------------------------------------------

def run(table: FeatureTable, *, split: str = "random", fewshot_k: int = 3,
        train_frac: float = 0.5, min_frames: int = 10,
        seed: int = 0, steps: int = 400, batch: int = 4096, lr: float = 5e-2,
        timelines: str | Path | None = None, mask_stride: int = 1,
        log_step=None, examples_video: str | None = None,
        decode_order: str = "entity-first", device: str | None = None):
    torch.manual_seed(seed)
    dev = pick_device(device)
    order = DECODE_ORDERS[decode_order]
    vs = build_video_space(videos=sorted(set(table.videos)), cs=table.cs)
    groups = vs.groups
    masks = {lvl: groups.mask(lvl, device=dev) for lvl in ORDER}
    off = {lvl: groups.offsets[groups.names.index(lvl)] for lvl in ORDER}

    truth = {"entity": entity_targets(table, vs),
             **{lvl: table.target(lvl) for lvl in LEVELS}}
    if timelines is not None:
        pos, aligned = true_positions(table, timelines, mask_stride=mask_stride)
        dist_units = "annotated frames"
    else:
        pos = row_positions(table.instance_id)
        aligned = torch.ones(len(table.instance_id), dtype=torch.bool)
        dist_units = "row units"
    n_inst_misaligned = len(table.instance_id[~aligned].unique())
    train, eval_, n_excluded = split_frames(
        table.instance_id, train_frac=train_frac, min_frames=min_frames,
        seed=seed, mode=split, k=fewshot_k)
    train &= aligned
    eval_ &= aligned
    if not train.any():
        raise RuntimeError(f"no instance has >= {min_frames} frames")
    dists = eval_train_distances(table.instance_id, pos, train, eval_)

    # entity -> its leaf class (for the agreement metric)
    ent_leaf = {int(e): int(l) for e, l in zip(truth["entity"], truth["leaf"])}

    tb, flat = build_models(vs, table.dim)
    tb.to(dev)
    flat.to(dev)
    # CPU masters stay for the ncm baseline; device copies train the models
    feats_tr_cpu = table.feats[train]
    tgt_tr_cpu = {lvl: truth[lvl][train] for lvl in ORDER}
    feats_tr = feats_tr_cpu.to(dev)
    tgt_tr = {lvl: t.to(dev) for lvl, t in tgt_tr_cpu.items()}
    n_tr = feats_tr.shape[0]

    def minibatches(g: torch.Generator):
        while True:
            for idx in torch.randperm(n_tr, generator=g).split(batch):
                yield idx

    def train_model(name, loss_fn, params):
        g = torch.Generator().manual_seed(seed + 1)
        it = minibatches(g)
        opt = torch.optim.Adam(params, lr=lr)
        losses = []
        for step in range(steps):
            idx = next(it)
            opt.zero_grad()
            loss = loss_fn(idx)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if log_step is not None:
                log_step({f"loss/{name}": float(loss)}, step)
        return losses

    def tb_loss(idx):
        logits, _ = tb_decode_v1(tb, masks, feats_tr[idx], "teacher",
                                 teacher={l: tgt_tr[l][idx] for l in ORDER},
                                 order=order)
        return sum(F.cross_entropy(logits[l], tgt_tr[l][idx]) for l in ORDER)

    def flat_loss(idx):
        out = flat(feats_tr[idx])
        return sum(F.cross_entropy(out[l], tgt_tr[l][idx] - off[l]) for l in ORDER)

    losses = {"TB": train_model("TB", tb_loss, tb.parameters()),
              "flat": train_model("flat", flat_loss, flat.parameters())}
    # live = gradient-receiving in THIS task: the shared index layer also holds
    # predicate/episode rows that V1's masked losses never touch
    n_dead_rows = groups.num_indices - sum(int(masks[lvl].sum()) for lvl in ORDER)
    tb_total = sum(p.numel() for p in tb.parameters())
    param_counts = {"TB": tb_total,
                    "flat": sum(p.numel() for p in flat.parameters()), "ncm": 0}
    live_param_counts = {**param_counts,
                         "TB": tb_total - n_dead_rows * (table.dim + 1)}

    # --- evaluation ---------------------------------------------------------
    feats_ev_cpu = table.feats[eval_]
    feats_ev = feats_ev_cpu.to(dev)
    tgt_ev = {lvl: truth[lvl][eval_] for lvl in ORDER}
    n_ev = feats_ev.shape[0]

    def bucket(d: int) -> int:
        for bi, edge in enumerate(DIST_EDGES):
            if d <= edge:
                return bi
        return len(DIST_EDGES)

    buckets = torch.tensor([bucket(int(d)) for d in dists])
    results: dict[str, dict] = {}

    ent_slice = groups.slice_of("entity")
    with torch.no_grad():
        model_preds: dict[str, dict[str, Tensor]] = {}
        for policy in POLICIES:
            chunks = [tb_decode_v1(tb, masks, feats_ev[s:s + batch], policy,
                                   order=order)[1]
                      for s in range(0, n_ev, batch)]
            model_preds[f"TB-{policy}"] = {
                lvl: torch.cat([c[lvl] for c in chunks]).cpu() for lvl in ORDER}
        fout = flat(feats_ev)
        model_preds["flat"] = {lvl: fout[lvl].argmax(-1).cpu() + off[lvl]
                               for lvl in ORDER}
        ncm_preds, (ncm_classes, ncm_protos) = ncm_predictions(
            feats_tr_cpu, tgt_tr_cpu, feats_ev_cpu, batch=batch)
        model_preds["ncm"] = ncm_preds

        # group-local entity scores per model, for within-video restricted decoding
        def tb_ent_scores(f: Tensor) -> Tensor:
            q = tb.attend(torch.zeros(f.shape[0], tb.dim, device=f.device), nu=f)
            out = tb.measure(q, mask=masks["entity"], commit="expectation")
            return out.logits[:, ent_slice]

        def ncm_ent_scores(f: Tensor) -> Tensor:
            s = torch.full((f.shape[0], ent_slice.stop - ent_slice.start),
                           -torch.inf)
            s[:, ncm_classes - ent_slice.start] = f @ ncm_protos.T
            return s

        ent_scores = {"TB": tb_ent_scores,
                      "flat": lambda f: flat(f)["entity"],
                      "ncm": ncm_ent_scores}

        videos_ev = [v for v, keep in zip(table.videos, eval_.tolist()) if keep]
        within = {name: within_video_accuracy(
                      fn, feats_ev_cpu if name == "ncm" else feats_ev,
                      tgt_ev["entity"], videos_ev, vs, ent_slice, batch=batch)
                  for name, fn in ent_scores.items()}

    ent_video = {g: v for (v, _), g in vs.entity_to_global.items()}
    for name, pred in model_preds.items():
        acc = {lvl: float((pred[lvl] == tgt_ev[lvl]).float().mean()) for lvl in ORDER}
        agree = float(torch.tensor(
            [ent_leaf.get(int(pe)) == int(pl)
             for pe, pl in zip(pred["entity"], pred["leaf"])]).float().mean())
        correct = (pred["entity"] == tgt_ev["entity"]).float()
        scene_acc = float(torch.tensor(
            [ent_video[int(p)] == ent_video[int(t)]
             for p, t in zip(pred["entity"], tgt_ev["entity"])]).float().mean())
        by_dist = []
        for bi in range(len(DIST_EDGES) + 1):
            sel = buckets == bi
            by_dist.append((int(sel.sum()),
                            float(correct[sel].mean()) if sel.any() else float("nan")))
        results[name] = {
            "acc": acc, "agree": agree, "by_dist": by_dist,
            "scene_acc": scene_acc,
            "within_video_acc": within["TB" if name.startswith("TB") else name],
        }

    examples = None
    if examples_video is not None:
        # TB top-k labels + probabilities per level for the example rows only
        ex_rows = [i for i, v in enumerate(videos_ev) if v == examples_video]
        inv_lbl = {lvl: {g: lbl for (l, lbl), g in table.cs.label_to_global.items()
                         if l == lvl} for lvl in LEVELS}
        ent_name = {g: f"obj{o}" for (v, o), g in vs.entity_to_global.items()
                    if v == examples_video}
        topk_by_row: dict[int, dict] = {}
        if ex_rows:
            with torch.no_grad():
                lg, _ = tb_decode_v1(tb, masks, feats_ev[torch.tensor(ex_rows)],
                                     "samp", order=order)
            for lvl in ORDER:
                probs, idx = lg[lvl].softmax(-1).topk(3)
                names = inv_lbl.get(lvl) or ent_name
                for j, i in enumerate(ex_rows):
                    topk_by_row.setdefault(i, {})[lvl] = [
                        [names.get(int(g), str(int(g))), round(float(p), 4)]
                        for p, g in zip(probs[j], idx[j])]
        examples = collect_examples(
            examples_video, videos_ev, pos[eval_],
            [o for o, keep in zip(table.obj_ids, eval_.tolist()) if keep],
            dists, model_preds, tgt_ev, vs, table.cs, topk_by_row)

    n_inst_kept = len(table.instance_id[train].unique())
    return {
        "models": results,
        "examples": examples,
        "live_param_counts": live_param_counts,
        "alpha": float(tb.alpha.detach()),
        "n_entities": groups.sizes[groups.names.index("entity")],
        "n_inst_kept": n_inst_kept, "n_inst_excluded": n_excluded,
        "n_inst_misaligned": n_inst_misaligned,
        "n_train_frames": int(train.sum()), "n_eval_frames": n_ev,
        "dist_edges": list(DIST_EDGES), "dist_units": dist_units,
        "split": split if split != "fewshot" else f"fewshot k={fewshot_k}",
        "param_counts": param_counts, "losses": losses,
        "config": {"split": split, "fewshot_k": fewshot_k,
                   "train_frac": train_frac, "min_frames": min_frames,
                   "seed": seed, "steps": steps, "batch": batch, "lr": lr,
                   "timelines": None if timelines is None else str(timelines),
                   "mask_stride": mask_stride, "decode_order": decode_order,
                   "device": str(dev)},
    }


def print_report(r: dict) -> None:
    print(f"\nV1 — entity re-identification "
          f"(frames-within-instance split: {r['split']}, seed {r['config']['seed']})")
    print(f"  entity indices {r['n_entities']} | instances kept {r['n_inst_kept']} "
          f"(excluded <min_frames: {r['n_inst_excluded']}, "
          f"cache/timeline misaligned: {r['n_inst_misaligned']}) | "
          f"train frames {r['n_train_frames']} | eval frames {r['n_eval_frames']}")
    pc, lc = r["param_counts"], r["live_param_counts"]
    print(f"  params (total/live): TB {pc['TB']:,}/{lc['TB']:,} | "
          f"flat {pc['flat']:,}/{lc['flat']:,} | ncm 0\n")
    print(f"  {'model':10} | {'entity':>6} {'leaf':>5} {'mid':>5} {'coarse':>6} | "
          f"{'agree':>5} {'in-vid':>6} {'scene':>5}")
    print("  " + "-" * 62)
    for name, m in r["models"].items():
        a = m["acc"]
        print(f"  {name:10} | {a['entity']:.2f}   {a['leaf']:.2f}  {a['mid']:.2f}  "
              f"{a['coarse']:.2f}  | {m['agree']:.2f}  {m['within_video_acc']:.2f}  "
              f"{m['scene_acc']:.2f}")
    edges = r["dist_edges"]
    labels = [f"<={e}" for e in edges] + [f">{edges[-1]}"]
    print(f"\n  entity accuracy vs. temporal distance to nearest train frame "
          f"({r['dist_units']})")
    print("  " + " ".join(f"{l:>7}" for l in ["model"] + labels))
    for name, m in r["models"].items():
        cells = [f"{acc:.2f}" if n else "-" for n, acc in m["by_dist"]]
        print("  " + " ".join(f"{c:>7}" for c in [name[:7]] + cells))
    ns = [str(n) for n, _ in next(iter(r["models"].values()))["by_dist"]]
    print("  " + " ".join(f"{c:>7}" for c in ["n="] + ns))


def _mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v == v]                    # drop NaN
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
        if len(vals) > 1 else 0.0
    return m, s


def print_aggregate(runs: list[dict]) -> None:
    seeds = [r["config"]["seed"] for r in runs]
    names = list(runs[0]["models"])
    print(f"\n=== V1 aggregate over seeds {seeds} (mean ± std) ===")
    print(f"  {'model':10} | " + " ".join(f"{lvl:>11}" for lvl in ORDER)
          + f" | {'ent-leaf':>11} {'in-vid':>11} {'scene':>11}")
    for name in names:
        cells = []
        for lvl in ORDER:
            m, s = _mean_std([r["models"][name]["acc"][lvl] for r in runs])
            cells.append(f"{m:.3f}±{s:.3f}")
        for key in ("agree", "within_video_acc", "scene_acc"):
            m, s = _mean_std([r["models"][name].get(key, float("nan"))
                              for r in runs])
            cells.append(f"{m:.3f}±{s:.3f}")
        print(f"  {name:10} | " + " ".join(f"{c:>11}" for c in cells[:4])
              + " | " + " ".join(f"{c:>11}" for c in cells[4:]))

    edges = runs[0]["dist_edges"]
    labels = [f"<={e}" for e in edges] + [f">{edges[-1]}"]
    print(f"\n  entity accuracy vs. temporal distance ({runs[0]['dist_units']}), "
          f"mean over seeds")
    print("  " + " ".join(f"{l:>7}" for l in ["model"] + labels))
    for name in names:
        cells = []
        for bi in range(len(edges) + 1):
            m, _ = _mean_std([r["models"][name]["by_dist"][bi][1] for r in runs
                              if r["models"][name]["by_dist"][bi][0] > 0])
            cells.append("-" if m != m else f"{m:.3f}")
        print("  " + " ".join(f"{c:>7}" for c in [name[:7]] + cells))
    mean_n = [int(sum(r["models"][names[0]]["by_dist"][bi][0] for r in runs)
                  / len(runs)) for bi in range(len(edges) + 1)]
    print("  " + " ".join(f"{c:>7}" for c in ["n~="] + [str(n) for n in mean_n]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dir with <video_id>.pt files")
    ap.add_argument("--timelines", default=None,
                    help="timelines dir; distances in true annotated-frame units")
    ap.add_argument("--split", choices=SPLIT_MODES, default="blocked")
    ap.add_argument("--fewshot-k", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="results/v1",
                    help="root for run artifacts (one JSON per seed)")
    ap.add_argument("--examples", default=None, metavar="VIDEO",
                    help="dump per-frame prediction examples for this video")
    ap.add_argument("--decode-order", choices=sorted(DECODE_ORDERS),
                    default="entity-first",
                    help="entity-first (V1 default) or sameas (paper-style: "
                         "concept ladder first, entity measured last)")
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--no-trackio", dest="trackio", action="store_false",
                    help="disable trackio logging (on by default)")
    args = ap.parse_args()

    table = load_feature_table(args.cache)
    print(table.stats.summary())
    print(f"device: {pick_device(args.device)}")
    tag = args.split if args.split != "fewshot" else f"fewshot{args.fewshot_k}"
    if args.decode_order != "entity-first":
        tag += f"_{args.decode_order}"
    out_dir = new_run_dir(args.out, tag=tag)
    runs = []
    for seed in args.seeds:
        with track("pvsg-v1", f"{out_dir.name}-seed{seed}",
                   {**vars(args), "seed": seed}, enabled=args.trackio) as log:
            r = run(table, split=args.split, fewshot_k=args.fewshot_k,
                    train_frac=args.train_frac, min_frames=args.min_frames,
                    seed=seed, steps=args.steps, batch=args.batch,
                    timelines=args.timelines, log_step=log,
                    examples_video=args.examples,
                    decode_order=args.decode_order, device=args.device)
            log({f"acc/{name}/{lvl}": m["acc"][lvl]
                 for name, m in r["models"].items() for lvl in ORDER})
        save_json(out_dir / f"seed{seed}.json", r)
        print_report(r)
        runs.append(r)
    if len(runs) > 1:
        print_aggregate(runs)
    print(f"\nsaved {len(runs)} run file(s) to {out_dir}")


if __name__ == "__main__":
    main()
