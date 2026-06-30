"""Mechanics of the region-feature pipeline (mask decode → pool → per-instance),
tested with the deterministic MockExtractor so no model download is needed.

Run: uv run pytest experiments/pvsg_hierarchy/test_features.py
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.pvsg_hierarchy.features import (
    MockExtractor,
    aggregate_instance,
    extract_frame,
    pool_region,
    prep_image,
)


def _synthetic_frame(h=140, w=210):
    """An image split into 3 solid-colour vertical bands + a panoptic mask whose
    pixel values are object_ids 1/2/3 (0 = background border)."""
    img = torch.zeros(3, h, w)
    mask = np.zeros((h, w), dtype=np.int32)
    colours = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0)}
    for k, (r, g, b) in colours.items():
        x0 = (k - 1) * (w // 3)
        x1 = k * (w // 3)
        img[0, 10:-10, x0:x1] = r
        img[1, 10:-10, x0:x1] = g
        img[2, 10:-10, x0:x1] = b
        mask[10:-10, x0:x1] = k
    return img, mask


def test_extract_frame_returns_one_feature_per_instance():
    img, mask = _synthetic_frame()
    ex = MockExtractor(dim=32)
    feats = extract_frame(ex, img, mask)
    assert set(feats.keys()) == {1, 2, 3}
    assert all(f.shape == (32,) for f in feats.values())
    # distinct regions (distinct colours) → distinct pooled features
    assert not torch.allclose(feats[1], feats[2])
    assert not torch.allclose(feats[2], feats[3])


def test_background_is_skipped():
    img, mask = _synthetic_frame()
    feats = extract_frame(MockExtractor(), img, mask)
    assert 0 not in feats


def test_pool_region_guards_empty_and_handles_tiny():
    dense = torch.randn(10, 15, 32)  # (gh, gw, D)
    # an empty mask pools to nothing → None (the only drop path)
    assert pool_region(dense, torch.zeros(140, 210, dtype=torch.bool), 14) is None
    # a single-pixel instance still yields its containing patch's feature, no crash
    tiny = torch.zeros(140, 210, dtype=torch.bool)
    tiny[0, 0] = True
    f = pool_region(dense, tiny, 14)
    assert f is not None and f.shape == (32,)


def test_prep_image_dims_are_patch_multiples():
    img = torch.rand(3, 137, 201)
    out = prep_image(img, patch=14)
    assert out.shape[1] % 14 == 0 and out.shape[2] % 14 == 0


def test_aggregate_instance_pools_frames():
    feats = [torch.ones(8) * i for i in range(4)]
    assert torch.allclose(aggregate_instance(feats), torch.ones(8) * 1.5)


def test_precompute_video_io_loop(tmp_path):
    """Write synthetic frame/mask PNGs and run the full precompute loop (mock)."""
    from PIL import Image

    from experiments.pvsg_hierarchy.precompute_features import (
        instance_features,
        precompute_video,
    )

    frames, masks = tmp_path / "frames", tmp_path / "masks"
    frames.mkdir()
    masks.mkdir()
    n_frames = 3
    for i in range(n_frames):
        img, mask = _synthetic_frame()
        rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb).save(frames / f"{i:06d}.jpg")
        Image.fromarray(mask.astype(np.uint8), mode="L").save(masks / f"{i:06d}.png")

    cache = precompute_video(frames, masks, MockExtractor(dim=32))
    assert set(cache) == {1, 2, 3}
    assert all(v.shape == (n_frames, 32) for v in cache.values())  # per-frame stack
    pooled = instance_features(cache)
    assert pooled[1].shape == (32,)


def test_precompute_all_decodes_mp4(tmp_path):
    """Driver path: synthetic mp4 + mask PNGs → on-the-fly decode → per-instance."""
    import imageio.v2 as imageio

    from experiments.pvsg_hierarchy.precompute_all import precompute_one

    masks_dir = tmp_path / "masks" / "vid0"
    masks_dir.mkdir(parents=True)
    frames = []
    n = 6
    for i in range(n):
        img, mask = _synthetic_frame()
        frames.append((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        from PIL import Image

        Image.fromarray(mask.astype(np.uint8), mode="L").save(masks_dir / f"{i:04d}.png")
    mp4 = tmp_path / "vid0.mp4"
    imageio.mimwrite(mp4, frames, fps=5)

    cache = precompute_one(mp4, masks_dir, MockExtractor(dim=32), fp16=True)
    assert set(cache) == {1, 2, 3}
    assert all(v.dtype == torch.float16 and v.shape[1] == 32 for v in cache.values())
