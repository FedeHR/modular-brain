"""Frame-aligned (subject, object) feature pairs — the V2/V7 data join.

The feature cache stores per-object stacks of *visible-frame* features with no
frame indices; `mask_timelines` recovers the alignment (k-th cached row = k-th
visible strided frame). This module joins three artifacts on the frame axis:

    cache      <video>.pt   {"features": {oid: [n_visible, D]}, "dim": ...}
    timelines  <video>.pt   {"areas": {oid: [F]}, "stems": [...], ...}
    spans      pvsg_data.load_relation_spans()  (annotation-frame units)

and emits, per usable (relation, frame), the subject's and object's features at
that same frame — the per-moment relation input V2 decodes and V7 tracks across
span boundaries. Frames where either participant has no cached feature (not
visible, or dropped in mask-pooling) are skipped and counted, never imputed.

Frame units: span frames index the sorted mask-PNG list of the video — the same
axis `video_timeline` measures — so no conversion beyond `mask_stride` is needed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from experiments.pvsg_hierarchy.mask_timelines import cache_row_frames
from experiments.pvsg_hierarchy.pvsg_data import Boundary, ConceptSpace, RelationSpan
from experiments.pvsg_hierarchy.pvsg_features import _norm


class VideoFrames:
    """Frame-indexed feature lookup for one video: `feat(oid, frame)` returns
    the cached DINO feature of object `oid` at mask frame `frame`, or None.

    Objects whose cached row count disagrees with the timeline's visible-frame
    count cannot be aligned (the extractor drops frames where the object is
    smaller than one patch, and the cache stores no per-row frame indices) —
    they are skipped and listed in `misaligned`, never guessed. If more than
    half the objects misalign, the cache itself is stale or the stride is
    wrong, and we raise instead."""

    def __init__(self, cache_blob: dict, timeline: dict, *, mask_stride: int = 1,
                 normalize: bool = True) -> None:
        self._row_of: dict[int, dict[int, int]] = {}
        self._feats: dict[int, Tensor] = {}
        self.misaligned: list[int] = []
        for oid, stack in cache_blob["features"].items():
            area = timeline["areas"].get(oid)
            frames = (None if area is None
                      else cache_row_frames(area, mask_stride=mask_stride).tolist())
            if frames is None or len(frames) != stack.shape[0]:
                self.misaligned.append(oid)
                continue
            rows = stack.float()
            self._feats[oid] = _norm(rows) if normalize else rows
            self._row_of[oid] = {f: k for k, f in enumerate(frames)}
        n = len(cache_blob["features"])
        if n and len(self.misaligned) > n / 2:
            raise ValueError(
                f"{len(self.misaligned)}/{n} objects misaligned between cache and "
                f"timeline — wrong mask_stride or stale cache")

    def feat(self, oid: int, frame: int) -> Tensor | None:
        row = self._row_of.get(oid, {}).get(frame)
        return None if row is None else self._feats[oid][row]

    def frames_of(self, oid: int) -> list[int]:
        return sorted(self._row_of.get(oid, {}))


def load_video_frames(video: str, cache_dir: str | Path, timelines_dir: str | Path,
                      **kw) -> VideoFrames | None:
    """None if either artifact is missing for this video, or if the video is
    wholesale misaligned (majority of objects — e.g. a truncated mp4 decode at
    extraction time shifted every row; 1019_3004044251 is the one such case in
    the 289-video cache). The caller counts either way."""
    cache_pt = Path(cache_dir) / f"{video}.pt"
    tl_pt = Path(timelines_dir) / f"{video}.pt"
    if not (cache_pt.exists() and tl_pt.exists()):
        return None
    try:
        return VideoFrames(torch.load(cache_pt, map_location="cpu"),
                           torch.load(tl_pt, map_location="cpu"), **kw)
    except ValueError:
        return None


@dataclass
class PairStats:
    n_spans: int = 0                 # spans of joinable videos
    n_rows: int = 0                  # emitted (relation, frame) rows
    n_skipped_frames: int = 0        # frame in span, but a participant unfeatured
    n_misaligned_objects: int = 0    # cache rows != timeline visible frames
    n_unknown_pred: int = 0          # span/boundary predicate outside the
                                     # official 57-predicate vocabulary
    missing_videos: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"pairs: {self.n_rows} rows from {self.n_spans} spans | "
                f"skipped frames {self.n_skipped_frames} | "
                f"misaligned objects {self.n_misaligned_objects} | "
                f"unknown-predicate spans {self.n_unknown_pred} | "
                f"missing videos {len(self.missing_videos)}")


@dataclass
class PairTable:
    """One row per usable (relation, frame): the V2 sample."""

    feat_s: Tensor           # [M, D]
    feat_o: Tensor           # [M, D]
    pred_g: Tensor           # [M] global predicate index
    videos: list[str]
    frames: list[int]
    subj_ids: list[int]
    obj_ids: list[int]
    stats: PairStats


def _group_spans(spans: list[RelationSpan]) -> dict[str, list[RelationSpan]]:
    by_video: dict[str, list[RelationSpan]] = defaultdict(list)
    for s in spans:
        by_video[s.video].append(s)
    return by_video


def build_pair_table(spans: list[RelationSpan], cs: ConceptSpace,
                     cache_dir: str | Path, timelines_dir: str | Path, *,
                     videos: list[str] | None = None, mask_stride: int = 1,
                     frame_stride: int = 1, normalize: bool = True) -> PairTable:
    """All (relation, frame) rows inside the given spans, at `frame_stride`
    granularity along each span."""
    by_video = _group_spans(spans)
    wanted = videos if videos is not None else sorted(by_video)
    stats = PairStats()
    fs, fo, pg, vids, frames_, sids, oids = [], [], [], [], [], [], []

    for video in wanted:
        vf = load_video_frames(video, cache_dir, timelines_dir,
                               mask_stride=mask_stride, normalize=normalize)
        if vf is None:
            stats.missing_videos.append(video)
            continue
        stats.n_misaligned_objects += len(vf.misaligned)
        for sp in by_video.get(video, []):
            if ("predicate", sp.predicate) not in cs.label_to_global:
                stats.n_unknown_pred += 1
                continue
            stats.n_spans += 1
            for frame in range(sp.start, sp.end + 1, frame_stride):
                a = vf.feat(sp.subj_id, frame)
                b = vf.feat(sp.obj_id, frame)
                if a is None or b is None:
                    stats.n_skipped_frames += 1
                    continue
                fs.append(a)
                fo.append(b)
                pg.append(cs.gindex("predicate", sp.predicate))
                vids.append(video)
                frames_.append(frame)
                sids.append(sp.subj_id)
                oids.append(sp.obj_id)

    if not fs:
        raise RuntimeError(f"no pairs joined (missing videos: "
                           f"{len(stats.missing_videos)}) — check cache/timeline dirs")
    stats.n_rows = len(fs)
    return PairTable(feat_s=torch.stack(fs), feat_o=torch.stack(fo),
                     pred_g=torch.tensor(pg), videos=vids, frames=frames_,
                     subj_ids=sids, obj_ids=oids, stats=stats)


@dataclass
class BoundaryWindowTable:
    """One row per usable (boundary, frame offset): the V7 sample. `active`
    says whether the triple holds at that frame — the exits-as-negatives rule
    falls out of the annotation itself."""

    feat_s: Tensor           # [M, D]
    feat_o: Tensor           # [M, D]
    pred_g: Tensor           # [M]
    active: Tensor           # [M] bool: triple holds at this frame
    boundary_id: Tensor      # [M] dense id grouping one boundary's window rows
    offset: Tensor           # [M] frame - boundary.frame (negative = before)
    kinds: list[str]         # per-boundary "start"/"end", indexed by boundary_id
    videos: list[str]
    frames: list[int]
    subj_ids: list[int]
    obj_ids: list[int]
    stats: PairStats


def build_boundary_windows(boundaries: list[Boundary], spans: list[RelationSpan],
                           cs: ConceptSpace, cache_dir: str | Path,
                           timelines_dir: str | Path, *, window: int = 5,
                           mask_stride: int = 1, normalize: bool = True,
                           videos: list[str] | None = None) -> BoundaryWindowTable:
    """For each interior boundary, the frames in [frame-window, frame+window]
    where both participants have features, labeled active/inactive per frame."""
    # triple -> its span list, for the per-frame active test
    triple_spans: dict[tuple[str, int, str, int], list[tuple[int, int]]] = defaultdict(list)
    for s in spans:
        triple_spans[(s.video, s.subj_id, s.predicate, s.obj_id)].append((s.start, s.end))

    by_video: dict[str, list[Boundary]] = defaultdict(list)
    for b in boundaries:
        by_video[b.video].append(b)
    wanted = videos if videos is not None else sorted(by_video)

    stats = PairStats()
    fs, fo, pg, act, bid, offs, vids, frames_ = [], [], [], [], [], [], [], []
    sids, oids = [], []
    kinds: list[str] = []

    for video in wanted:
        vf = load_video_frames(video, cache_dir, timelines_dir,
                               mask_stride=mask_stride, normalize=normalize)
        if vf is None:
            stats.missing_videos.append(video)
            continue
        stats.n_misaligned_objects += len(vf.misaligned)
        for b in by_video.get(video, []):
            if ("predicate", b.predicate) not in cs.label_to_global:
                stats.n_unknown_pred += 1
                continue
            key = (b.video, b.subj_id, b.predicate, b.obj_id)
            rows_here = 0
            for frame in range(b.frame - window, b.frame + window + 1):
                if frame < 0:
                    continue
                a = vf.feat(b.subj_id, frame)
                c = vf.feat(b.obj_id, frame)
                if a is None or c is None:
                    stats.n_skipped_frames += 1
                    continue
                fs.append(a)
                fo.append(c)
                pg.append(cs.gindex("predicate", b.predicate))
                act.append(any(s <= frame <= e for s, e in triple_spans[key]))
                bid.append(len(kinds))
                offs.append(frame - b.frame)
                vids.append(video)
                frames_.append(frame)
                sids.append(b.subj_id)
                oids.append(b.obj_id)
                rows_here += 1
            if rows_here:
                kinds.append(b.kind)
                stats.n_spans += 1  # here: boundaries kept

    if not fs:
        raise RuntimeError(f"no boundary windows joined (missing videos: "
                           f"{len(stats.missing_videos)})")
    stats.n_rows = len(fs)
    return BoundaryWindowTable(
        feat_s=torch.stack(fs), feat_o=torch.stack(fo), pred_g=torch.tensor(pg),
        active=torch.tensor(act), boundary_id=torch.tensor(bid),
        offset=torch.tensor(offs), kinds=kinds, videos=vids, frames=frames_,
        subj_ids=sids, obj_ids=oids, stats=stats)
