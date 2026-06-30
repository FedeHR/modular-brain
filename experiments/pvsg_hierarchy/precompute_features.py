"""Precompute region features for one PVSG video and cache them.

PVSG layout after frame extraction (OpenPVSG's prep step turns each mp4 into
`frames/<video_id>/*.jpg` at the annotated FPS, alongside `masks/<video_id>/*.png`):

    frames/<video_id>/000000.jpg   masks/<video_id>/000000.png   (pixel = object_id)

This runs the foundation model once per frame, mask-pools per instance, and saves
`{object_id: features[n_frames, D]}` — the per-frame stack (keep it for H2; pool
with `instance_features` for H1).

Designed to run on a GPU box over the whole subset. Locally, use `--mock` to
smoke-test the IO + loop without a model; on the GPU box drop `--mock` and pick a
model (`--model dinov2_vitb14`, or DINOv3 once its license is accepted).

    uv run python -m experiments.pvsg_hierarchy.precompute_features \
        --frames data/frames/0001_xxx --masks data/masks/0001_xxx --out cache/0001.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from experiments.pvsg_hierarchy.features import DinoExtractor, MockExtractor, extract_frame


def load_image(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def precompute_video(frames_dir, masks_dir, extractor, *, stride: int = 1) -> dict[int, Tensor]:
    """{object_id: features (n_frames_present, D)} for one video."""
    masks = sorted(Path(masks_dir).glob("*.png"))[::stride]
    per_obj: dict[int, list[Tensor]] = defaultdict(list)
    for mpath in masks:
        frame = next(Path(frames_dir).glob(mpath.stem + ".*"), None)
        if frame is None:
            continue
        img = load_image(frame)
        pan = np.array(Image.open(mpath))
        for oid, f in extract_frame(extractor, img, pan).items():
            per_obj[oid].append(f)
    return {oid: torch.stack(v) for oid, v in per_obj.items()}


def instance_features(cache: dict[int, Tensor]) -> dict[int, Tensor]:
    """Pool each instance's frames into one vector (H1 input)."""
    return {oid: f.mean(0) for oid, f in cache.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--stride", type=int, default=1, help="process every k-th frame")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mock", action="store_true", help="no model — smoke-test IO only")
    args = ap.parse_args()

    extractor = MockExtractor() if args.mock else DinoExtractor(args.model, device=args.device)
    cache = precompute_video(args.frames, args.masks, extractor, stride=args.stride)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": cache, "model": args.model, "dim": extractor.dim}, args.out)
    n_feats = sum(v.shape[0] for v in cache.values())
    print(f"saved {len(cache)} instances / {n_feats} frame-features (D={extractor.dim}) → {args.out}")


if __name__ == "__main__":
    main()
