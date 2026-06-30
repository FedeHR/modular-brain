# Video-data branch — progress

Branch: `stage-b-pvsg-wordnet`. This is "Stage B" of the TB modernization:
move the Tensor Brain off VRD-E's static images + affine "simulated change" onto
**PVSG** (real video panoptic scene graphs), enrich its flat object labels with a
hierarchy, and get a first hypothesis (H1) running on real data.

Detailed docs: project `README.md` ("Dataset enrichment") for the design;
`experiments/pvsg_hierarchy/DESIGN.md` for the taxonomy specifics + the H1 result.
This file is the integrative status + roadmap.

> Sibling branch `worktree-sanity-check-vrd` (kept on disk, not merged) separately
> verified that our modular TB reimplementation reproduces the official BTN
> behaviour on real VRD-E features. That de-risked the *model*; this branch is the
> *data*.

---

## 1. What we implemented

All under `experiments/pvsg_hierarchy/`.

| File | Purpose |
|------|---------|
| `pvsg_categories.json` | PVSG vocabulary (115 thing + 11 stuff = 126 object classes, 57 relations), extracted from the dataset's `pvsg.json`. |
| `pvsg_instances.json` | Compact per-video slice of `pvsg.json`: objects (id, category) + relations (subj, obj, predicate). 400 videos, 7596 objects, 4587 triples (326 KB). |
| `taxonomy.py` | **Layer A** — automatic WordNet object taxonomy: `class → first physical-entity sense → 3 levels (leaf / mid / coarse)`. Source-agnostic `Taxonomy` contract (levels + `parent_map` + `index_groups`). |
| `derive.py` | Report: coverage, the coarse grouping, how `mid` refines the big bucket, unmapped classes. `--vrd` reproduces it on VRD-100 for comparison. |
| `pvsg_data.py` | Adapter: PVSG instances → one concept `IndexGroups` (leaf 126 / mid 62 / coarse 11 / predicate 57) + per-instance label triples + relation triples. |
| `run_h1.py` | **H1 experiment** — hierarchical grounding under perceptual noise: TB (shared index + skip) vs flat independent heads. |
| `features.py` | **Perception front-end** — region features over PVSG's GT masks: DINO dense patches → mask-pooled per-instance vector. Pluggable extractor (`DinoExtractor` real / `MockExtractor` for tests). |
| `precompute_features.py` | Runs the front-end over one video's frames+masks → cached `{object_id: features[n_frames, D]}`. `--mock` smoke-tests IO locally. |
| `precompute_all.py` | **Resumable driver** over many videos: on-the-fly mp4 decode (no stored frames), per-video checkpoint (skip-if-done), fp16, `--device cuda`. For the cluster. |
| `cluster/` | SLURM walkthrough (`CLUSTER.md`) + `env_setup.sh` (login-node uv setup + asset prefetch) + `job.sh` (SBATCH script). |
| `test_taxonomy.py`, `test_pvsg_data.py`, `test_features.py` | 18 tests locking the taxonomy, adapter, and feature-pipeline (incl. mp4 decode) invariants. |
| `vrd_objects.json` | VRD-100 class list (from BTN) for the VRD-E comparison. |

Results in numbers:
- Taxonomy over PVSG 126: **97% auto-mapped**, levels **leaf 126 / mid 62 / coarse 11**; 4 unmapped surfaced (`ballon`, `vaccum`, `scissor`, `others`).
- Adapter: **7358 instances mapped**, 238 dropped (unmapped categories).
- H1 (clean-feature stand-in, DINO deferred): TB stays hierarchically consistent under noise (**0.98 vs flat 0.47** at σ=1); at σ=2 TB **coarse acc 0.74 > leaf acc 0.64** — coarse survives when the fine class is lost; flat collapses to **0.15** consistency.

---

## 2. Why — design choices & goals

**Goal (Tresp's priority).** Not SOTA — show that *useful concepts arise from
cognitive motivations*. H1 is the semantic story: index-grounded hierarchical
decoding is consistent / robust where flat decoders aren't ("semantic memory as a
prior"). PVSG gives real video + real relations + tracked instances, so the
temporal/episodic story (H2) is reachable later from the same dataset.

**Three label layers, kept separate** (so complexity doesn't compound):
- **A — object taxonomy** (`dog → animal`): *not* in PVSG; we add it from WordNet.
- **B — observed relations** (PVSG's 57 predicates): already real and rich; we keep
  them, never synthesize.
- **C — background knowledge** (`dangerous`, `pet`): optional, trained only via the
  semantic-memory pathway; the modern replacement for VRD-E's hand-injected hidden
  labels. Deferred until a hypothesis needs it.

**Why WordNet (not LLM / Wikidata) for layer A, now.** It is the recognized
standard (VRD-E, ImageNet, Visual Genome use it), fully *automatic* (zero
per-class hardcoding), offline, and reproducible. An LLM-generated *ground-truth*
taxonomy would hurt credibility (model drift, circularity) — kept as an appendix
robustness check only. Wikidata (subclass + typed relations, would also give layer
C) is the recommended *future* upgrade. The `Taxonomy` contract is source-agnostic,
so swapping the producer touches no model code.

**Why three automatic levels, no hand-curation:**
- `leaf` = the PVSG class.
- `coarse` = WordNet supersense (`noun.animal → animal`): total, comparable, no
  "Other" bucket needed.
- `mid` = ancestor at a fixed **depth-from-root** (6): a same-depth cut is
  comparable across classes and nests under the supersense, splitting the broad
  `artifact` bucket (63% of PVSG) into meaningful groups (container, furnishing,
  implement, conveyance, ware, device …). Shallow branches (animal/plant/food)
  collapse `mid → coarse`, i.e. no spurious split.
- Disambiguation is automatic (first *physical-entity* sense → `table` = furniture,
  not the data table). Unmappable names are **surfaced, not guessed**.

This deliberately does **not** copy VRD-E's fixed B/P/G scheme: its derivation
code was never released, and a fixed hypernym-step count lands classes at
incomparable depths (measured: 59 distinct "grandparents" for 100 VRD classes).

**Perception front-end — no SAM, GT masks + DINO.** PVSG ships pixel-accurate
*tracked* panoptic masks (per-frame PNGs whose pixel value is `object_id`, stable
across frames), so the boxes→masks modernization is provided by the dataset — we
do **not** run SAM. SAM 3 (open-vocab concept segmentation) is a later, optional
"predicted-mask realism" axis only. The feature extractor is **DINOv3** for the
final run (best 2026 dense features; license-gated) with **DINOv2** as the
ungated default and a one-line swap. Granularity: features are **per-frame,
per-instance** (mask-pooled DINO patches); identity is the per-video `object_id`;
H1 pools an instance's frames into one vector, H2 keeps the per-frame sequence.

**Why a clean-feature stand-in for the *first* H1 run.** The H1 claim is
*architectural* (shared index + skip ⇒ consistency), not perceptual, so the first
run uses one deterministic feature per class to isolate the mechanism. The real
DINO features (above) replace it once precomputed — both the mock mechanics and
the real DINOv2 path are verified end-to-end.

**Working principles (from direct feedback):** keep the data enrichment *minimal,
standard, automatic, inspectable*; prefer one deterministic step over a multi-stage
pipeline; don't hand-patch individual labels.

---

## 3. How to proceed

In rough priority order:

1. **Perception front-end — pipeline + resumable driver + cluster scripts built and
   tested (mock + real DINOv2). Remaining = run it on the cluster.** Follow
   `cluster/CLUSTER.md` on the LMU SLURM cluster (uv venv on `/nfs/data8`, prefetch
   DINO+data on the login node, `srun` to debug, `sbatch -p major job.sh`) → per-video
   feature cache. Then wire the cache into `run_h1.py` (replace `fixed_features`
   with `instance_features` keyed by `(video, object_id)`; set TB `dim` to the
   feature dim). This turns the H1 mechanism demo into a perception-grounded result
   and is the only compute-heavy step. (Locally feasible too: M3/MPS ≈ 15 fps for
   ViT-S; a 25-video subset is minutes.)
2. **Use layer B (relations).** `pvsg_data.load_triples` already exposes the S–P–O
   triples; decode the full scene-graph triple (subject + concepts, predicate,
   object + concepts) instead of isolated objects, and aggregate observed relations
   into semantic memory.
3. **Order-invariance variant of H1.** The other half of the QTB
   consistency story (decode order doesn't change the answer); architectural, no new
   data needed.
4. **H2 (temporal / episodic).** Use PVSG's tracked instance ids across frames as
   episodic-instance indices; test object permanence / occlusion and remote recall.
5. **Optional upgrades.** Layer C (background knowledge via Wikidata/ConceptNet);
   taxonomy source Options 2/3 (Wikidata / LLM); a curated mid-level if the depth-cut
   proves too coarse for a given experiment.

**Known limitations (documented, not hidden):**
- Coarse level imbalanced (63% `artifact`) — addressed by the `mid` level.
- ~5 polysemy residuals the physical filter can't resolve (`washer→person`,
  `iron→substance`, `microwave→phenomenon`, `egg→animal`) — left visible; exactly
  what a context-aware source (Wikidata/LLM) would fix.
- 4 unmapped classes (dataset typos + the `others` catch-all) — dropped, reported.

---

## Reproduce

```sh
uv run python -m experiments.pvsg_hierarchy.derive          # taxonomy report (PVSG)
uv run python -m experiments.pvsg_hierarchy.derive --vrd    # same, on VRD-100
uv run python -m experiments.pvsg_hierarchy.run_h1          # the H1 experiment
uv run pytest experiments/pvsg_hierarchy/ -q                # 17 experiment tests
uv run pytest -q                                            # 57 core tests

# perception front-end (GPU box; --mock smoke-tests IO locally without a model)
uv run python -m experiments.pvsg_hierarchy.precompute_features \
    --frames <frames_dir> --masks <masks_dir> --out cache/<vid>.pt --model dinov2_vitb14
```
