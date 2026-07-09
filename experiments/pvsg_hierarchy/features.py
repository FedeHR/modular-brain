"""Region-feature extraction over PVSG's ground-truth tracked masks.

PVSG ships pixel-accurate panoptic masks: one PNG per frame whose pixel values
are `object_id`s (0 = background), stable across frames within a video. So we do
**not** run SAM — masks and tracking are given. We only extract one feature
vector per (frame, instance) by pooling a foundation-model's dense patch features
over each object's mask.

Pipeline per frame:
    image ─DINO─▶ dense patch grid [gh, gw, D]
    mask  ──────▶ per-instance boolean masks (pan_mask == object_id)
    pool  ──────▶ {object_id: feature[D]}   (mask-weighted mean over patches)

Granularity: features are per-frame-per-instance; identity is the per-video
`object_id`. `aggregate_instance` pools an instance's frames → one vector (H1);
keep the per-frame stack for H2.

The extractor is pluggable: `DinoExtractor` (real, DINOv2 default / DINOv3 swap)
for the GPU run, `MockExtractor` (deterministic, no downloads) for testing the
PVSG-specific plumbing.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RegionFeatureExtractor(Protocol):
    patch: int
    dim: int

    def dense(self, image: Tensor) -> Tensor:
        """(3, H, W) normalized image -> (H//patch, W//patch, dim) patch grid."""
        ...


def _round_to(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def prep_image(img: Tensor, patch: int, target: int = 518) -> Tensor:
    """Resize a (3, H, W) float[0,1] image so each side is a multiple of `patch`
    (longest side ~ target), then ImageNet-normalize."""
    _, h, w = img.shape
    scale = target / max(h, w)
    nh, nw = _round_to(round(h * scale), patch), _round_to(round(w * scale), patch)
    img = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def pool_region(dense: Tensor, mask: Tensor, patch: int) -> Tensor | None:
    """Mask-weighted mean of patch features. `dense` is (gh, gw, D); `mask` is a
    boolean (H, W) at the *image* resolution fed to the extractor. Returns (D,) or
    None if the instance covers no patch (smaller than one patch)."""
    gh, gw, d = dense.shape
    m = mask.float()[None, None]  # (1,1,H,W)
    # average mask coverage within each patch -> (gh, gw)
    cov = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
    total = cov.sum()
    if total <= 0:
        return None
    return (dense * cov[..., None]).sum(dim=(0, 1)) / total


def _bbox(mask: Tensor) -> tuple[int, int, int, int] | None:
    """(y0, x0, y1, x1) inclusive bounding box of a boolean mask, or None."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def pool_union_box(dense: Tensor, mask_a: Tensor, mask_b: Tensor,
                   patch: int) -> Tensor | None:
    """Pool patch features over the UNION BOUNDING BOX of two masks — the
    paper's f(BB_pred) (Sec 4.6): the box enclosing both participants, so the
    region *between* them (where the relation lives) is included, unlike a plain
    mask union. Returns (D,) or None if either mask is empty."""
    ba, bb = _bbox(mask_a), _bbox(mask_b)
    if ba is None or bb is None:
        return None
    y0, x0 = min(ba[0], bb[0]), min(ba[1], bb[1])
    y1, x1 = max(ba[2], bb[2]), max(ba[3], bb[3])
    box = torch.zeros_like(mask_a, dtype=torch.bool)
    box[y0:y1 + 1, x0:x1 + 1] = True
    return pool_region(dense, box, patch)


def extract_frame(
    extractor: RegionFeatureExtractor, image01: Tensor, pan_mask: np.ndarray
) -> dict[int, Tensor]:
    """image01: (3,H,W) float in [0,1]; pan_mask: (H,W) int array of object_ids.
    Returns {object_id: feature[D]} for every non-background instance present."""
    img = prep_image(image01, extractor.patch)
    dense = extractor.dense(img)  # (gh, gw, D)
    _, ih, iw = img.shape
    mask_t = torch.from_numpy(np.ascontiguousarray(pan_mask))[None, None].float()
    mask_rs = F.interpolate(mask_t, size=(ih, iw), mode="nearest")[0, 0]  # align to image
    feats: dict[int, Tensor] = {}
    for oid in np.unique(pan_mask):
        if oid == 0:  # background
            continue
        f = pool_region(dense, mask_rs == oid, extractor.patch)
        if f is not None:
            feats[int(oid)] = f
    return feats


def extract_frame_unions(
    extractor: RegionFeatureExtractor, image01: Tensor, pan_mask: np.ndarray,
    pairs,
) -> dict[tuple[int, int], Tensor]:
    """Union-box feature per (subject, object) pair present in this frame. Shares
    one dense-grid pass across all requested pairs. `pairs` is an iterable of
    (subj_id, obj_id); a pair is emitted only if both masks cover ≥1 patch."""
    img = prep_image(image01, extractor.patch)
    dense = extractor.dense(img)
    _, ih, iw = img.shape
    mask_t = torch.from_numpy(np.ascontiguousarray(pan_mask))[None, None].float()
    mask_rs = F.interpolate(mask_t, size=(ih, iw), mode="nearest")[0, 0]
    out: dict[tuple[int, int], Tensor] = {}
    for s, o in pairs:
        f = pool_union_box(dense, mask_rs == s, mask_rs == o, extractor.patch)
        if f is not None:
            out[(int(s), int(o))] = f
    return out


def extract_frame_scene(
    extractor: RegionFeatureExtractor, image01: Tensor
) -> Tensor:
    """Whole-frame ("scene") feature — the paper's f(scene) / f(BB) over the
    complete-scene bounding box (TB §6.3, Alg. 1 line 6, `q̃_T ← u·f(scene)`).

    Pooled the same way as the per-object and union-box features (mask-weighted
    mean of DINO patches, here over a uniform all-ones mask), so f(scene),
    f(BB_sub), f(BB_obj) and f(BB_pred) all share one feature statistic and can
    feed the same representation-layer projection `u·f(·)`. Returns (D,)."""
    img = prep_image(image01, extractor.patch)
    dense = extractor.dense(img)  # (gh, gw, D)
    return dense.reshape(-1, dense.shape[-1]).mean(dim=0)


def aggregate_instance(per_frame: list[Tensor]) -> Tensor:
    """Pool an instance's per-frame features into one vector (H1). Stack instead
    for H2."""
    return torch.stack(per_frame).mean(0)


# --- extractors ---------------------------------------------------------------


class MockExtractor:
    """Deterministic stand-in (no downloads): each patch feature is a fixed linear
    map of the patch's mean colour, so pooling is exercised end-to-end."""

    def __init__(self, patch: int = 14, dim: int = 32) -> None:
        self.patch = patch
        self.dim = dim
        g = torch.Generator().manual_seed(0)
        self.proj = torch.randn(3, dim, generator=g)

    def dense(self, image: Tensor) -> Tensor:
        _, h, w = image.shape
        gh, gw = h // self.patch, w // self.patch
        patches = image.reshape(3, gh, self.patch, gw, self.patch).mean(dim=(2, 4))  # (3,gh,gw)
        return torch.einsum("chw,cd->hwd", patches, self.proj)


class DinoExtractor:
    """Real extractor. Default DINOv2 (ungated, torch.hub). For the GPU run, swap
    to DINOv3 (`source`/`name` below) after accepting its license.

        DinoExtractor("dinov2_vitb14")            # default, 768-d
        DinoExtractor("dinov3_vitl16", source=…)  # DINOv3 on the GPU box
    """

    def __init__(self, name: str = "dinov2_vitb14", repo: str = "facebookresearch/dinov2",
                 device: str = "cpu") -> None:
        # trust_repo=True avoids an interactive prompt on offline compute nodes;
        # set TORCH_HOME to a shared dir and pre-fetch weights on the login node.
        self.model = torch.hub.load(repo, name, trust_repo=True).to(device).eval()
        self.device = device
        self.patch = 14 if "v2" in name else 16
        self.dim = self.model.embed_dim

    @torch.no_grad()
    def dense(self, image: Tensor) -> Tensor:
        _, h, w = image.shape
        gh, gw = h // self.patch, w // self.patch
        out = self.model.forward_features(image[None].to(self.device))
        tokens = out["x_norm_patchtokens"][0]  # (gh*gw, D)
        return tokens.reshape(gh, gw, -1).cpu()
