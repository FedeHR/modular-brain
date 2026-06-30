"""Resumable driver: precompute DINO region features over many PVSG videos.

Decodes each mp4 on the fly (no 22 GB of extracted frames), pairs each 5-FPS
frame with its panoptic mask PNG, mask-pools DINO features per instance, and
writes one cache file per video. Re-running **skips finished videos**, so a
dropped SLURM job just resumes — point it at the same `--out` and resubmit.

Expected layout (after unzipping the PVSG zips):
    <videos>/<video_id>.mp4
    <masks>/<video_id>/*.png          (pixel value = object_id; one PNG / 5-FPS frame)

Output: `<out>/<video_id>.pt` = {"features": {object_id: tensor[n_frames, D] fp16},
"model": ..., "dim": ...}. Pool an instance's frames for H1
(`features.aggregate_instance`); keep the stack for H2.

    python -m experiments.pvsg_hierarchy.precompute_all \
        --videos $WORK/pvsg/VidOR/videos --masks $WORK/pvsg/VidOR/masks \
        --out $WORK/cache --model dinov2_vitb14 --device cuda --limit 25
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.pvsg_hierarchy.features import DinoExtractor, MockExtractor, extract_frame


def find_video(videos_dir: Path, vid: str) -> Path | None:
    hits = sorted(Path(videos_dir).glob(f"{vid}.*"))
    return hits[0] if hits else None


def frames_for_masks(mp4: Path, n_masks: int):
    """Yield (mask_index, frame_uint8_HWC): a 5-FPS subsample paired 1:1 with the
    sorted mask list (frame 0 → mask 0, frame `round(fps/5)` → mask 1, …)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(mp4), "ffmpeg")
    fps = reader.get_meta_data().get("fps") or 30
    step = max(1, round(fps / 5.0))
    mi = 0
    for i, frame in enumerate(reader):
        if i % step == 0:
            yield mi, frame
            mi += 1
            if mi >= n_masks:
                break
    reader.close()


def precompute_one(mp4: Path, masks_dir: Path, extractor, *, mask_stride: int = 1, fp16: bool = True):
    masks = sorted(Path(masks_dir).glob("*.png"))[::mask_stride]
    per_obj: dict[int, list[torch.Tensor]] = defaultdict(list)
    for mi, frame in frames_for_masks(mp4, len(masks)):
        img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        pan = np.array(Image.open(masks[mi]))
        for oid, f in extract_frame(extractor, img, pan).items():
            per_obj[oid].append(f.half() if fp16 else f)
    return {oid: torch.stack(v) for oid, v in per_obj.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="process first N videos (0 = all)")
    ap.add_argument("--mask-stride", type=int, default=1, help="subsample masks (2 ≈ 2.5 FPS)")
    ap.add_argument("--mock", action="store_true", help="no model — smoke-test the loop")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extractor = MockExtractor() if args.mock else DinoExtractor(args.model, device=args.device)

    vids = sorted(p.name for p in Path(args.masks).iterdir() if p.is_dir())
    if args.limit:
        vids = vids[: args.limit]
    print(f"{len(vids)} videos | model={args.model} device={args.device} dim={extractor.dim}", flush=True)

    done = 0
    for n, vid in enumerate(vids, 1):
        dst = out / f"{vid}.pt"
        if dst.exists():
            print(f"[{n}/{len(vids)}] {vid}: cached, skip", flush=True)
            done += 1
            continue
        mp4 = find_video(Path(args.videos), vid)
        if mp4 is None:
            print(f"[{n}/{len(vids)}] {vid}: NO VIDEO FILE, skip", flush=True)
            continue
        t0 = time.time()
        try:
            cache = precompute_one(mp4, Path(args.masks) / vid, extractor, mask_stride=args.mask_stride)
        except Exception as e:  # keep going; the next resubmit can retry this video
            print(f"[{n}/{len(vids)}] {vid}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        torch.save({"features": cache, "model": args.model, "dim": extractor.dim}, dst)
        nf = sum(v.shape[0] for v in cache.values())
        done += 1
        print(f"[{n}/{len(vids)}] {vid}: {len(cache)} instances / {nf} feats "
              f"in {time.time() - t0:.1f}s → {dst.name}", flush=True)
    print(f"done: {done}/{len(vids)} videos cached in {args.out}", flush=True)


if __name__ == "__main__":
    main()
