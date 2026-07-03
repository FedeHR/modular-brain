"""Per-instance mask timelines from PVSG's panoptic mask PNGs.

For each video, scan every mask PNG (pixel value = object_id, matching
`precompute_all.py`) and record each object's mask *area per frame*. This one
cheap CPU pass yields everything the video experiments need from the masks:

- **row→frame alignment for the feature cache**: `precompute_all` stacked
  features only for frames where an object is present, without storing frame
  indices. The k-th cached row of object `o` is the k-th frame where
  `areas[o] > 0` (after applying the same `--mask-stride`), so the timeline
  restores the lost alignment.
- **V5 degradation bins**: relative mask area (small = far / partial).
- **V6a events**: visibility gaps = occlusion candidates; visible-then-gone
  -until-the-end = exit candidates.

Output: one `<video_id>.pt` per video =
    {"stems":  [mask filename stems, sorted],           # length F
     "areas":  {object_id: LongTensor[F]},              # pixels per frame
     "height": int, "width": int}

Run (CPU, no GPU needed; resumable — skips finished videos):
    python -m experiments.pvsg_hierarchy.mask_timelines \
        --masks $WORK/pvsg/VidOR/masks --out $WORK/timelines
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

BACKGROUND = 0  # pixel value used for "no object" in PVSG panoptic masks


def video_timeline(masks_dir: Path) -> dict:
    """Scan one video's mask PNGs -> {"stems", "areas", "height", "width"}."""
    paths = sorted(Path(masks_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no mask PNGs under {masks_dir}")
    per_obj: dict[int, dict[int, int]] = defaultdict(dict)  # oid -> {frame: area}
    h = w = None
    for fi, p in enumerate(paths):
        pan = np.array(Image.open(p))
        if h is None:
            h, w = pan.shape[:2]
        oids, counts = np.unique(pan, return_counts=True)
        for oid, c in zip(oids.tolist(), counts.tolist()):
            if oid != BACKGROUND:
                per_obj[oid][fi] = c
    n = len(paths)
    areas: dict[int, Tensor] = {}
    for oid, frame_area in per_obj.items():
        t = torch.zeros(n, dtype=torch.long)
        t[list(frame_area)] = torch.tensor(list(frame_area.values()), dtype=torch.long)
        areas[oid] = t
    return {"stems": [p.stem for p in paths], "areas": areas, "height": h, "width": w}


# --- analysis helpers (run locally, downstream of the cluster pass) -----------

def visibility(area: Tensor) -> Tensor:
    """Boolean per-frame visibility from an area timeline."""
    return area > 0


def intervals(vis: Tensor) -> list[tuple[int, int]]:
    """Maximal [start, end] (inclusive) runs of True."""
    idx = vis.nonzero(as_tuple=True)[0].tolist()
    if not idx:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def gaps(vis: Tensor) -> dict[str, list[tuple[int, int]]]:
    """Classify the invisible stretches of one instance.

    - "occlusion": a gap *between* two visible runs (the entity returns —
      identity is confirmed by the dataset's tracked id).
    - "exit": visible, then gone through the video's end (never returns).
    Leading invisibility (before first appearance) is ignored — nothing to
    remember yet.
    """
    runs = intervals(vis)
    out: dict[str, list[tuple[int, int]]] = {"occlusion": [], "exit": []}
    for (_, e1), (s2, _) in zip(runs, runs[1:]):
        out["occlusion"].append((e1 + 1, s2 - 1))
    if runs and runs[-1][1] < len(vis) - 1:
        out["exit"].append((runs[-1][1] + 1, len(vis) - 1))
    return out


def cache_row_frames(area: Tensor, mask_stride: int = 1) -> Tensor:
    """Frame index of each cached feature row for this object.

    `precompute_all` iterated masks `sorted(...)[::mask_stride]` and appended a
    feature row whenever the object was present. So row k corresponds to the
    k-th visible frame of the *strided* timeline; returned indices are in the
    unstrided frame numbering.
    """
    strided_frames = torch.arange(len(area))[::mask_stride]
    vis = area[::mask_stride] > 0
    return strided_frames[vis]


# --- resumable driver ----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="root dir: <masks>/<video_id>/*.png")
    ap.add_argument("--out", required=True, help="output dir for <video_id>.pt")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (0 = all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(p.name for p in Path(args.masks).iterdir() if p.is_dir())
    if args.limit:
        vids = vids[: args.limit]

    done = skipped = failed = 0
    t0 = time.time()
    for i, vid in enumerate(vids):
        dst = out / f"{vid}.pt"
        if dst.exists():
            skipped += 1
            continue
        try:
            tl = video_timeline(Path(args.masks) / vid)
        except Exception as e:  # keep the sweep alive; report at the end
            print(f"[{i + 1}/{len(vids)}] {vid}  FAILED: {e}", flush=True)
            failed += 1
            continue
        tmp = dst.with_suffix(".tmp")
        torch.save(tl, tmp)
        tmp.rename(dst)  # atomic: a crash never leaves a truncated .pt
        done += 1
        print(f"[{i + 1}/{len(vids)}] {vid}  frames={len(tl['stems'])} "
              f"objects={len(tl['areas'])}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done={done} skipped={skipped} failed={failed} total={len(vids)}")


if __name__ == "__main__":
    main()
