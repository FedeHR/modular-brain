"""Precompute union-box predicate features for one PVSG video.

The per-object cache (`precompute_features`) pools DINO patches over each
instance's own mask. The predicate step of the Tensor Brain (Algorithm 1 line
29) instead wants f(BB_pred): patches pooled over the UNION BOUNDING BOX of the
subject and object (Sec 4.6), so the region *between* the two participants — the
seat of the relation — is represented. That can only be computed from the dense
patch grid + both masks at extraction time, so it lives in its own cache:

    unions/<video>.pt  {"unions": {(subj_id, obj_id): {"frames": [...],
                                   "feats": [n, D]}}, "model": ..., "dim": D}

`frames` are mask indices, matching `pvsg_pairs.build_pair_table`'s frame keys.
Only (s, o) pairs that appear in a relation span are computed, and only the mask
frames those spans touch are decoded. Batches over a videos/masks tree like
`precompute_all` and is resumable (finished `<id>.pt` are skipped).

    # cluster, all videos (--videos/--masks default to the madeira paths):
    python -m experiments.pvsg_hierarchy.precompute_unions \
        --out $WORK/unions --model dinov2_vitb14 --device cuda
    # local smoke on one video:
    python -m experiments.pvsg_hierarchy.precompute_unions \
        --videos data/videos --masks data/masks --out unions \
        --only 1001_5247398775
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from experiments.pvsg_hierarchy.features import (
    DinoExtractor,
    MockExtractor,
    extract_frame_unions,
)
from experiments.pvsg_hierarchy.precompute_features import load_image
from experiments.pvsg_hierarchy.pvsg_data import load_relation_spans

# Unzipped OpenPVSG VidOR tree on the madeira cluster; the default extraction
# targets. Override --videos/--masks for a local run (e.g. data/videos, data/masks).
CLUSTER_VIDEOS = ("/nfs/data8/harjes/pvsg/VidOR/mnt/lustre/jkyang/CVPR23/"
                  "openpvsg/data/vidor/videos")
CLUSTER_MASKS = ("/nfs/data8/harjes/pvsg/VidOR/mnt/lustre/jkyang/CVPR23/"
                 "openpvsg/data/vidor/masks")


def active_pairs_by_frame(spans, subvideo: str) -> dict[int, set[tuple[int, int]]]:
    """mask_index -> {(subj_id, obj_id)} active there, from this video's spans."""
    out: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for sp in spans:
        if sp.video != subvideo:
            continue
        for f in range(sp.start, sp.end + 1):
            out[f].add((sp.subj_id, sp.obj_id))
    return out


def _mp4_frames(mp4: Path, n_masks: int, want: set[int]) -> dict[int, torch.Tensor]:
    """Decode the wanted mask indices from an mp4, mirroring the annotation
    stride (frame 0 -> mask 0, every round(fps/5)th)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(mp4), "ffmpeg")
    fps = reader.get_meta_data().get("fps") or 30
    step = max(1, round(fps / 5.0))
    out, mi = {}, 0
    for i, frame in enumerate(reader):
        if i % step == 0:
            if mi in want:
                arr = np.asarray(frame, dtype=np.float32) / 255.0
                out[mi] = torch.from_numpy(arr).permute(2, 0, 1)
            mi += 1
            if mi >= n_masks:
                break
    reader.close()
    return out


def precompute_unions(masks_dir, extractor, *, frames_dir=None, mp4=None,
                      spans=None, video_id=None) -> dict:
    spans = spans if spans is not None else load_relation_spans()
    masks = sorted(Path(masks_dir).glob("*.png"))
    by_frame = active_pairs_by_frame(spans, video_id)
    want = {k for k in range(len(masks)) if by_frame.get(k)}

    imgs = (_mp4_frames(Path(mp4), len(masks), want) if mp4 else None)
    acc: dict[tuple[int, int], dict[int, torch.Tensor]] = defaultdict(dict)
    for k in sorted(want):
        if imgs is not None:
            img = imgs.get(k)
        else:
            fp = next(Path(frames_dir).glob(masks[k].stem + ".*"), None)
            img = load_image(fp) if fp else None
        if img is None:
            continue
        pan = np.array(Image.open(masks[k]))
        for (s, o), f in extract_frame_unions(extractor, img, pan,
                                              by_frame[k]).items():
            acc[(s, o)][k] = f

    unions = {so: {"frames": sorted(d), "feats": torch.stack([d[f] for f in sorted(d)])}
              for so, d in acc.items() if d}
    return {"unions": unions, "model": getattr(extractor, "name", "mock"),
            "dim": extractor.dim}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=CLUSTER_VIDEOS,
                    help="dir of <id>.mp4 (default: madeira cluster path)")
    ap.add_argument("--masks", default=CLUSTER_MASKS,
                    help="dir of <id>/ mask folders (default: madeira cluster path)")
    ap.add_argument("--out", required=True, help="output dir for <id>.pt union caches")
    ap.add_argument("--only", default=None, help="restrict to one video id (local test)")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="no model — smoke-test IO")
    args = ap.parse_args()

    extractor = (MockExtractor() if args.mock
                 else DinoExtractor(args.model, device=args.device))
    if not args.mock:
        extractor.name = args.model
    spans = load_relation_spans()
    videos_with_spans = {s.video for s in spans}

    vids = sorted(p.name for p in Path(args.masks).iterdir() if p.is_dir())
    if args.only:
        vids = [v for v in vids if v == args.only]
    vids = [v for v in vids if v in videos_with_spans][:args.limit]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for n, vid in enumerate(vids, 1):
        out_pt = Path(args.out) / f"{vid}.pt"
        if out_pt.exists():                                   # resumable
            print(f"[{n}/{len(vids)}] {vid}: exists, skip", flush=True)
            continue
        mp4 = next(Path(args.videos).glob(f"{vid}.*"), None)
        if mp4 is None:
            print(f"[{n}/{len(vids)}] {vid}: NO VIDEO FILE, skip", flush=True)
            continue
        blob = precompute_unions(Path(args.masks) / vid, extractor, mp4=mp4,
                                 spans=spans, video_id=vid)
        torch.save(blob, out_pt)
        nfeat = sum(v["feats"].shape[0] for v in blob["unions"].values())
        print(f"[{n}/{len(vids)}] {vid}: {len(blob['unions'])} pairs / "
              f"{nfeat} union features -> {out_pt}", flush=True)


if __name__ == "__main__":
    main()
