"""Sanity-check the precomputed DINO region-feature cache on the cluster.

Run on the box where the .pt files live (no GPU needed):

    cd $WORK/modular-brain
    source .venv/bin/activate
    python -m experiments.pvsg_hierarchy.cluster.check_cache \
        --cache $WORK/cache --masks $WORK/pvsg/VidOR/masks

Four checks:
  1. COVERAGE   every mask dir got a .pt; nothing missing/extra; expected count.
  2. MANIFEST   #instances per .pt vs #objects PVSG annotates for that video.
  3. INTEGRITY  each tensor is [n_frames, 768] fp16, finite, non-zero norm.
  4. SEMANTICS  spot-check that distinct instances have distinct pooled vectors.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
EXPECTED_DIM = 768  # dinov2_vitb14


def load_manifest_counts() -> dict[str, int]:
    raw = json.load(open(HERE / "pvsg_instances.json"))
    return {v["video_id"]: len(v["objects"]) for v in raw["videos"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dir with <video_id>.pt files")
    ap.add_argument("--masks", required=True, help="dir with <video_id>/ mask subdirs")
    ap.add_argument("--full", action="store_true", help="load every .pt (slower, thorough)")
    args = ap.parse_args()

    cache = Path(args.cache)
    masks = Path(args.masks)

    pt_ids = {p.stem for p in cache.glob("*.pt")}
    mask_ids = {p.name for p in masks.iterdir() if p.is_dir()}
    manifest = load_manifest_counts()

    print("=" * 70)
    print("1. COVERAGE")
    print(f"   .pt files:          {len(pt_ids)}")
    print(f"   mask subdirs:       {len(mask_ids)}")
    print(f"   VidOR in manifest:  {sum(1 for k in manifest if k in mask_ids)}")
    missing = sorted(mask_ids - pt_ids)
    extra = sorted(pt_ids - mask_ids)
    print(f"   missing (mask but no .pt): {len(missing)}  {missing[:8]}")
    print(f"   extra   (.pt but no mask): {len(extra)}  {extra[:8]}")
    print(f"   verdict: {'OK — every mask dir cached' if not missing else 'INCOMPLETE'}")

    print("=" * 70)
    print("2. MANIFEST CROSS-CHECK  (#instances in .pt  vs  #objects annotated)")
    print("   note: .pt count <= annotated is normal (unmapped/never-visible objs)")
    to_load = sorted(pt_ids) if args.full else sorted(pt_ids)[:20]
    bad_dim, nonfinite, zero_norm = [], [], []
    n_frames_hist, inst_over = Counter(), []
    total_inst = total_feat = 0
    norm_min = norm_max = None

    for vid in to_load:
        blob = torch.load(cache / f"{vid}.pt", map_location="cpu")
        if blob.get("dim") != EXPECTED_DIM:
            bad_dim.append((vid, blob.get("dim")))
        feats = blob["features"]
        total_inst += len(feats)
        ann = manifest.get(vid)
        if ann is not None and len(feats) > ann:
            inst_over.append((vid, len(feats), ann))
        for oid, t in feats.items():
            total_feat += t.shape[0]
            n_frames_hist[t.shape[0]] += 1
            if t.ndim != 2 or t.shape[1] != EXPECTED_DIM:
                bad_dim.append((vid, oid, tuple(t.shape)))
            tf = t.float()
            if not torch.isfinite(tf).all():
                nonfinite.append((vid, oid))
            norms = tf.norm(dim=1)
            if (norms == 0).any():
                zero_norm.append((vid, oid))
            lo, hi = float(norms.min()), float(norms.max())
            norm_min = lo if norm_min is None else min(norm_min, lo)
            norm_max = hi if norm_max is None else max(norm_max, hi)

    print(f"   checked {len(to_load)} videos  ({'ALL' if args.full else 'first 20 — pass --full for all'})")
    print(f"   total instances: {total_inst}   total frame-features: {total_feat}")
    print(f"   .pt has MORE instances than annotated (should be empty): {inst_over[:8]}")

    print("=" * 70)
    print("3. INTEGRITY")
    print(f"   wrong dim/shape:  {len(bad_dim)}  {bad_dim[:5]}")
    print(f"   non-finite feats: {len(nonfinite)}  {nonfinite[:5]}")
    print(f"   zero-norm feats:  {len(zero_norm)}  {zero_norm[:5]}")
    print(f"   feat L2-norm range: [{norm_min:.2f}, {norm_max:.2f}]  (expect ~O(10-100), not 0)")
    common = n_frames_hist.most_common(5)
    print(f"   n_frames per instance (top): {common}")
    print(f"   single-frame instances: {n_frames_hist[1]}")

    print("=" * 70)
    print("4. SEMANTICS  (one video: are distinct instances distinguishable?)")
    vid = to_load[0]
    feats = torch.load(cache / f"{vid}.pt", map_location="cpu")["features"]
    pooled = {oid: t.float().mean(0) for oid, t in feats.items()}
    oids = list(pooled)[:6]
    print(f"   {vid}: {len(feats)} instances, showing pairwise cosine of pooled vectors")
    for i in range(len(oids)):
        row = []
        for j in range(len(oids)):
            a, b = pooled[oids[i]], pooled[oids[j]]
            cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)
            row.append(f"{cos:+.2f}")
        print(f"     obj {oids[i]:>3}: " + " ".join(row))
    print("   healthy: diagonal +1.00, off-diagonal clearly < 1 (features aren't collapsed)")
    print("=" * 70)


if __name__ == "__main__":
    main()
