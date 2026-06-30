# PVSG object taxonomy (layer A) — design notes

The layer-A/B/C framing and the source alternatives (WordNet / Wikidata / LLM)
live in the top-level `README.md` ("Dataset enrichment"). This note records the
concrete WordNet derivation and its measured behaviour on the real PVSG classes.

## Method (one deterministic step, no hand-curation)

```
class name → first physical-entity noun sense → 3 levels (leaf, mid, coarse)
"dog"      → dog.n.01                         → dog,  animal,      animal
"car"      → car.n.01                         → car,  conveyance,  artifact
"table"    → table.n.02                       → table, furnishing, artifact
```

- **Disambiguation** is automatic: among `wn.synsets(name, NOUN)` take the first
  whose hypernyms include `physical_entity`. This rejects the abstract senses that
  misfire for a vision vocabulary (`table.n.01` = a *data* table) with no override
  list — `table → table.n.02 (furniture)`, automatically.
- **Levels** (fine → coarse):
  - `leaf` — the PVSG class.
  - `coarse` — WordNet supersense / lexname (`noun.animal` → `animal`). Total and
    comparable; no "Other" bucket. The 25 "unique beginners" carry the placeholder
    lexname `noun.Tops`; we fall back to the lemma (`person → person`).
  - `mid` — ancestor at a fixed **depth-from-root** (default 6). A same-depth cut
    is comparable across classes and nests under the supersense. depth 6 splits the
    broad `artifact` supersense into meaningful groups (container, furnishing,
    implement, conveyance, ware, device …); shallow branches (animal/plant/food)
    collapse `mid → coarse`, i.e. no spurious split. `mid_depth` is the one tunable
    knob.
- Output is the source-agnostic `Taxonomy` (levels + `parent_map(child, parent)` +
  `index_groups()`), so a Wikidata/LLM producer can replace this one later with no
  change to TB code.

## Measured on the real 126 PVSG classes (`derive.py`)

- **97% auto-mapped**; levels **leaf 126 / mid 62 / coarse 11**.
- **4 unmapped, surfaced not guessed:** `ballon`, `vaccum` (dataset typos),
  `scissor` (should be plural), `others` (catch-all). We do not silently invent a
  label for these.
- **Coarse imbalance handled by the mid level.** 63% of classes share the
  `artifact` supersense (PVSG is kitchen/indoor-heavy), so the coarse level alone
  is shallow. The depth-6 `mid` splits `artifact` (79) into meaningful groups —
  device (11), container (9), implement (8), furnishing (6), equipment (5),
  ware (4), consumer_goods (4), conveyance (3) … + a tail of ~19 singletons.
- **Honest limitations (visible in the printed table):**
  - *A handful of polysemy errors the physical-filter can't fix* because both
    senses are physical or the intended sense isn't first: `washer → person`
    (the agent, not the machine), `iron → substance` (the metal, not the
    appliance), `microwave → phenomenon` (the radiation, not the oven), `egg →
    animal` (biology, not food). ~5/126. These are exactly the cases a
    context-aware source (Wikidata typing / LLM) would resolve; we leave them
    visible rather than hand-patch.

## Comparison to VRD-E's fixed B/P/G (why we did not copy it)

VRD-E assigned each class a fixed Basic/Parent/Grandparent by walking a fixed
number of hypernyms up the path. Because WordNet chains have variable depth, that
lands classes at incomparable granularities (measured earlier on VRD-100: 59
distinct "grandparents" for 100 classes — `dog→carnivore` vs `car→
self-propelled_vehicle`). Our supersense level is comparable by construction (one
shared lexicographer partition) and fully automatic. Run `derive.py --vrd` to
reproduce the taxonomy on the same 100 VRD classes.

## First H1 run (`pvsg_data.py` + `run_h1.py`)

`pvsg_data.py` maps PVSG's per-object instances (`pvsg_instances.json`, the compact
objects+relations slice of `pvsg.json`) through the taxonomy into one concept
`IndexGroups` — leaf 126 / mid 62 / coarse 11 / predicate 57 — dropping the 238
instances whose category can't map (typos / `others`), never guessing.

`run_h1.py` tests the H1 claim: the TB decodes leaf→mid→coarse through a *shared*
index with the skip update carrying each commit, so its decode stays
hierarchically consistent under perceptual noise, where independent flat heads do
not. Trained on clean per-class prototypes (SAM2/DINO deferred — the question is
architectural), evaluated on the real 7358 instances with growing feature noise:

| σ   | model | leaf | mid | coarse | consistent |
|-----|-------|------|-----|--------|------------|
| 1.0 | TB    | 1.00 | 0.99| **0.98** | **0.98**  |
| 1.0 | flat  | 1.00 | 0.80| 0.58   | 0.47       |
| 2.0 | TB    | 0.64 | 0.55| **0.74** | **0.68**  |
| 2.0 | flat  | 0.68 | 0.38| 0.38   | 0.15       |

Two takeaways: (1) the TB stays hierarchically *consistent* under noise (0.98 vs
0.47 at σ=1); (2) at σ=2 its coarse accuracy (0.74) **exceeds** its leaf accuracy
(0.64) — when the fine class is lost, the coarse class survives via the shared
semantic backbone (the "occluded object still decodes as *animal*" effect). Flat
heads collapse into contradiction (consistency 0.15). This is a clean-feature
mechanism demonstration, not a benchmark.

## Status / next

- **Done:** automatic 3-level taxonomy over the real PVSG 126; instance→concept
  `IndexGroups` adapter; first H1 (hierarchical-consistency-under-noise) run.
- **Next:** swap stand-in features for SAM2/DINO (Stage B feature cache); use the
  relation triples (layer B) for the full scene-graph decode; order-invariance
  variant; Options 2/3 (Wikidata/LLM) and layer C as listed in the README.
