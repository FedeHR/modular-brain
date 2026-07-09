"""Precompute whole-frame ("scene") features for PVSG videos on the cluster.

The Tensor Brain analyses **four** bounding boxes per sample (TB §6.3): the
complete **scene**, the **subject** box, the **object** box, and the
**predicate** box. `precompute_all`/`precompute_features` cache the per-object
(subject/object) features; `precompute_unions` caches the predicate union box.
This script caches the missing first one: f(scene) — Algorithm 1 line 6's
bottom-up input `q̃_T ← u·f(scene)`, the seat of scene-level context the BTN
loses without it (V21_BTN_PERCEPTION.md, deferral D1).

A scene feature is the DINO patch grid mean-pooled over the *whole* frame — same
pooling as the per-object and union features (with a uniform mask), so all four
f(·) share one feature statistic and feed the same representation-layer
projection. There is one scene vector per frame, not per instance:

    scenes/<video>.pt  {"scenes": tensor[n_frames, D] fp16,
                        "frames": [mask indices, aligned 1:1 with `scenes`],
                        "model": ..., "dim": D}

`frames` are mask indices (frame 0 → mask 0, every round(fps/5)th), matching the
frame keys of `precompute_all` and `pvsg_pairs.build_pair_table`. Mirrors
`precompute_all`: decodes each mp4 on the fly (no extracted-frame tree),
resumable (finished `<id>.pt` are skipped), one cache file per video.

    # cluster, all videos (defaults target the madeira VidOR tree):
    python -m experiments.pvsg_hierarchy.precompute_scenes \
        --out $WORK/scenes --model dinov2_vitb14 --device cuda
    # only frames that a relation span touches (cheap, aligns with the union cache):
    python -m experiments.pvsg_hierarchy.precompute_scenes \
        --out $WORK/scenes --device cuda --spans-only
    # local smoke on one video, no model:
    python -m experiments.pvsg_hierarchy.precompute_scenes \
        --videos data/videos --masks data/masks --out scenes \
        --only 1001_5247398775 --mock
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.pvsg_hierarchy.features import (
    DinoExtractor,
    MockExtractor,
    extract_frame_scene,
)
from experiments.pvsg_hierarchy.precompute_all import find_video, frames_for_masks
from experiments.pvsg_hierarchy.precompute_unions import CLUSTER_MASKS, CLUSTER_VIDEOS
from experiments.pvsg_hierarchy.pvsg_data import load_relation_spans


def span_frames_by_video(spans) -> dict[str, set[int]]:
    """video_id -> {mask indices touched by any relation span}."""
    out: dict[str, set[int]] = defaultdict(set)
    for sp in spans:
        out[sp.video].update(range(sp.start, sp.end + 1))
    return out


def precompute_one(mp4: Path, masks_dir: Path, extractor, *, mask_stride: int = 1,
                   want: set[int] | None = None, fp16: bool = True):
    """Return (frames, scenes) — parallel lists/tensor of mask indices and their
    whole-frame features. `want` restricts to those mask indices (after stride)."""
    masks = sorted(Path(masks_dir).glob("*.png"))[::mask_stride]
    frames: list[int] = []
    feats: list[torch.Tensor] = []
    for mi, frame in frames_for_masks(mp4, len(masks)):
        if want is not None and mi not in want:
            continue
        img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        f = extract_frame_scene(extractor, img)
        frames.append(mi)
        feats.append(f.half() if fp16 else f)
    scenes = torch.stack(feats) if feats else torch.empty(0, extractor.dim)
    return frames, scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=CLUSTER_VIDEOS,
                    help="dir of <id>.mp4 (default: madeira cluster path)")
    ap.add_argument("--masks", default=CLUSTER_MASKS,
                    help="dir of <id>/ mask folders (default: madeira cluster path)")
    ap.add_argument("--out", required=True, help="output dir for <id>.pt scene caches")
    ap.add_argument("--only", default=None, help="restrict to one video id (local test)")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="process first N videos (0 = all)")
    ap.add_argument("--mask-stride", type=int, default=1, help="subsample masks (2 ≈ 2.5 FPS)")
    ap.add_argument("--spans-only", action="store_true",
                    help="extract only frames touched by a relation span")
    ap.add_argument("--mock", action="store_true", help="no model — smoke-test the loop")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extractor = MockExtractor() if args.mock else DinoExtractor(args.model, device=args.device)

    spans = load_relation_spans()
    by_video = span_frames_by_video(spans)
    vids = sorted(p.name for p in Path(args.masks).iterdir() if p.is_dir())
    if args.spans_only:
        vids = [v for v in vids if by_video.get(v)]  # skip videos with no spans
    if args.only:
        vids = [v for v in vids if v == args.only]
    if args.limit:
        vids = vids[: args.limit]
    print(f"{len(vids)} videos | model={args.model} device={args.device} "
          f"dim={extractor.dim} spans_only={args.spans_only}", flush=True)

    done = 0
    for n, vid in enumerate(vids, 1):
        dst = out / f"{vid}.pt"
        if dst.exists():                                   # resumable
            print(f"[{n}/{len(vids)}] {vid}: cached, skip", flush=True)
            done += 1
            continue
        mp4 = find_video(Path(args.videos), vid)
        if mp4 is None:
            print(f"[{n}/{len(vids)}] {vid}: NO VIDEO FILE, skip", flush=True)
            continue
        want = by_video.get(vid, set()) if args.spans_only else None
        t0 = time.time()
        try:
            frames, scenes = precompute_one(mp4, Path(args.masks) / vid, extractor,
                                            mask_stride=args.mask_stride, want=want)
        except Exception as e:  # keep going; a resubmit retries this video
            print(f"[{n}/{len(vids)}] {vid}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        torch.save({"scenes": scenes, "frames": frames,
                    "model": args.model, "dim": extractor.dim}, dst)
        done += 1
        print(f"[{n}/{len(vids)}] {vid}: {len(frames)} scene feats "
              f"in {time.time() - t0:.1f}s → {dst.name}", flush=True)
    print(f"done: {done}/{len(vids)} videos cached in {args.out}", flush=True)


if __name__ == "__main__":
    main()
