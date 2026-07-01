"""Join the precomputed DINO region-feature cache with the concept space.

`precompute_all.py` writes one `<video>.pt` per video =
    {"features": {object_id: tensor[n_frames, D] fp16}, "model": ..., "dim": D}.
`pvsg_data.load_instances()` maps each PVSG object to its (leaf, mid, coarse)
taxonomy labels (dropping unmappable categories). This module is the bridge: it
emits, for every mapped instance, its per-frame DINO features paired with that
instance's three *global* concept indices — the real-perception input H1 needs in
place of `run_h1.fixed_features` (random per-class prototypes + σ·randn noise).

**Granularity is per-frame by default** (one row per (instance, frame)): a frame
carries real occlusion / blur / viewpoint, so it is the honest, harder setting —
no synthetic noise. `FeatureTable.pool_by_instance()` collapses to one mean vector
per instance (the denoised, easier setting) if the per-frame task proves too hard.

Instances whose video isn't cached, or whose object never survived mask-pooling,
are dropped and counted — never fabricated.

    # on the cluster, where the cache lives:
    python -m experiments.pvsg_hierarchy.pvsg_features --cache $WORK/cache
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments.pvsg_hierarchy.pvsg_data import ConceptSpace, build_concept_space, load_instances

LEVELS = ("leaf", "mid", "coarse")


def _norm(x: Tensor) -> Tensor:
    """L2-normalize each row to √D (matches `run_h1.fixed_features`, so TB
    attention scales are consistent whether features are real or stand-in)."""
    return F.normalize(x, dim=-1) * (x.shape[-1] ** 0.5)


@dataclass
class JoinStats:
    """What happened during the join (all counts, nothing guessed)."""

    n_labeled: int = 0                       # mapped instances from load_instances
    dropped_unmapped: int = 0                # categories WordNet couldn't map
    n_instances: int = 0                     # instances with a real feature
    n_frame_features: int = 0                # total (instance, frame) rows
    missing_videos: list[str] = field(default_factory=list)  # no .pt cached
    missing_objects: int = 0                 # object_id absent from its .pt
    dim: int | None = None

    def summary(self) -> str:
        return (f"joined {self.n_instances}/{self.n_labeled} mapped instances "
                f"({self.n_frame_features} frame-features) | dim={self.dim} | "
                f"dropped_unmapped={self.dropped_unmapped} | "
                f"missing_videos={len(self.missing_videos)} | "
                f"missing_objects={self.missing_objects}")


@dataclass
class FeatureTable:
    """Real DINO features + (leaf, mid, coarse) global indices.

    One row per (instance, frame) at `granularity='frame'`; per instance after
    `pool_by_instance()`. `instance_id` is a dense 0..K-1 key grouping the rows of
    one PVSG object, so a train/eval split can be made *by instance* (no frame of
    the same object leaking across the split)."""

    feats: Tensor            # [M, D] float32
    leaf_g: Tensor           # [M] long, global concept index
    mid_g: Tensor            # [M] long
    coarse_g: Tensor         # [M] long
    instance_id: Tensor      # [M] long, dense grouping key
    videos: list[str]        # [M] provenance
    obj_ids: list[int]       # [M] provenance
    cs: ConceptSpace
    stats: JoinStats
    granularity: str = "frame"

    @property
    def dim(self) -> int:
        return self.feats.shape[1]

    @property
    def n_instances(self) -> int:
        return int(self.instance_id.max()) + 1 if len(self.instance_id) else 0

    def target(self, level: str) -> Tensor:
        return {"leaf": self.leaf_g, "mid": self.mid_g, "coarse": self.coarse_g}[level]

    def pool_by_instance(self, normalize: bool = True) -> FeatureTable:
        """Collapse to one mean vector per instance (the denoised H1 fallback)."""
        if self.granularity == "instance":
            return self
        k = self.n_instances
        sums = torch.zeros(k, self.dim).index_add_(0, self.instance_id, self.feats)
        counts = torch.bincount(self.instance_id, minlength=k).clamp(min=1)
        means = sums / counts[:, None]
        if normalize:
            means = _norm(means)
        # labels are constant within an instance -> scatter picks the (only) value
        lab = {lvl: torch.zeros(k, dtype=torch.long) for lvl in LEVELS}
        for lvl in LEVELS:
            lab[lvl][self.instance_id] = self.target(lvl)
        first: dict[int, int] = {}
        for i, iid in enumerate(self.instance_id.tolist()):
            first.setdefault(iid, i)
        order = [first[j] for j in range(k)]
        return FeatureTable(
            feats=means,
            leaf_g=lab["leaf"], mid_g=lab["mid"], coarse_g=lab["coarse"],
            instance_id=torch.arange(k),
            videos=[self.videos[i] for i in order],
            obj_ids=[self.obj_ids[i] for i in order],
            cs=self.cs, stats=self.stats, granularity="instance",
        )


def load_feature_table(
    cache_dir: str | Path,
    cs: ConceptSpace | None = None,
    *,
    normalize: bool = True,
) -> FeatureTable:
    """Per-frame DINO features + (leaf, mid, coarse) global indices per instance."""
    cache_dir = Path(cache_dir)
    cs = cs or build_concept_space()
    instances, dropped = load_instances(cs=cs)
    stats = JoinStats(n_labeled=len(instances), dropped_unmapped=dropped)

    by_video: dict[str, list] = defaultdict(list)
    for inst in instances:
        by_video[inst.video].append(inst)

    feats: list[Tensor] = []
    leaf_g, mid_g, coarse_g, inst_ids, videos, obj_ids = [], [], [], [], [], []
    missing_v: list[str] = []
    k = 0

    for video, insts in by_video.items():
        pt = cache_dir / f"{video}.pt"
        if not pt.exists():
            missing_v.append(video)
            continue
        blob = torch.load(pt, map_location="cpu")
        if stats.dim is None:
            stats.dim = blob.get("dim")
        cache_feats = blob["features"]
        for inst in insts:
            stack = cache_feats.get(inst.obj_id)
            if stack is None:            # object never survived mask-pooling
                stats.missing_objects += 1
                continue
            rows = stack.float()          # [n_frames, D]
            if normalize:
                rows = _norm(rows)
            n = rows.shape[0]
            feats.append(rows)
            leaf_g += [cs.gindex("leaf", inst.leaf)] * n
            mid_g += [cs.gindex("mid", inst.mid)] * n
            coarse_g += [cs.gindex("coarse", inst.coarse)] * n
            inst_ids += [k] * n
            videos += [video] * n
            obj_ids += [inst.obj_id] * n
            k += 1

    stats.missing_videos = sorted(missing_v)
    stats.n_instances = k
    if not feats:
        raise RuntimeError(f"no instances joined against cache at {cache_dir} "
                           f"({len(missing_v)} videos missing) — check --cache path")

    feats_t = torch.cat(feats)
    stats.n_frame_features = feats_t.shape[0]
    return FeatureTable(
        feats=feats_t,
        leaf_g=torch.tensor(leaf_g),
        mid_g=torch.tensor(mid_g),
        coarse_g=torch.tensor(coarse_g),
        instance_id=torch.tensor(inst_ids),
        videos=videos,
        obj_ids=obj_ids,
        cs=cs,
        stats=stats,
    )


def _report(table: FeatureTable) -> None:
    """Per-class counts — the numbers that settle the H1 train/eval split."""
    import statistics

    print(table.stats.summary())
    print(f"feat L2-norm: mean {table.feats.norm(dim=1).mean():.2f} "
          f"[{table.feats.norm(dim=1).min():.2f}, {table.feats.norm(dim=1).max():.2f}]")
    inst = table.pool_by_instance()  # per-instance view for class-balance stats
    # frames per instance
    fpi = torch.bincount(table.instance_id).tolist()
    print(f"frames/instance: min {min(fpi)} median {int(statistics.median(fpi))} max {max(fpi)}")
    for level in LEVELS:
        tgt = inst.target(level).tolist()
        counts = sorted(tgt.count(k) for k in set(tgt))
        singletons = sum(1 for c in counts if c == 1)
        print(f"  {level:>6}: {len(counts):>3} classes present | "
              f"instances/class min {counts[0]} median {int(statistics.median(counts))} "
              f"max {counts[-1]} | singleton classes {singletons}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dir with <video_id>.pt files")
    ap.add_argument("--raw", action="store_true", help="skip L2-normalization")
    args = ap.parse_args()
    table = load_feature_table(args.cache, normalize=not args.raw)
    _report(table)


if __name__ == "__main__":
    main()
