# tb

A minimal implementation of the Tensor Brain (TB) primitives, factored
following the Quantum Tensor Brain (QTB) paper into three operations:

- `evolve(q, state) -> (q, state)` — apply the recurrent dynamics on q (and any per-episode state)
- `attend(q, ν, μ) -> q` — sensory injection: `q ← q + μ·g(ν)`. Pure; no index readout
- `measure(q, mask, α, β, commit, ...) -> (k, q, logits)` — committed measurement: `q ← α·q + β·a_k` with `commit ∈ {sample, argmax, expectation, gumbel, teacher}`

The QTB §7.4 "ignorant Y-measurement" (attention readout) is recovered
as `measure(commit="expectation")`. Modes (perception / episodic recall /
semantic recall) are different *input patterns* to the same loop — they
are not separate code paths.

## Install

```sh
uv venv
uv pip install -e ".[dev]"
```

## Modes (three runnable demos)

All three share the same tiny KG (`tb/examples/_toy_data.py`): 10 instances,
8 entities, 3 concept classes, 4 predicates, 10 triples.

| Mode              | Seed q with               | Sensory ν? | Mask pattern                | Composer script                | Explicit script                          |
| ----------------- | ------------------------- | ---------- | --------------------------- | ------------------------------ | ---------------------------------------- |
| Episodic recall   | `a_t` (instance embedding) | no         | S/O/P                       | `tb.examples.toy_kg`           | `tb.examples.toy_kg_explicit`            |
| Semantic recall   | learnable `ā`             | no         | S (teacher-forced) / O / P  | `tb.examples.semantic_recall`  | `tb.examples.semantic_recall_explicit`   |
| Perception        | zeros                     | yes        | T / S (+concept) / O (+concept) / P | `tb.examples.perception` | `tb.examples.perception_explicit` |

Each "composer" script uses `decode_chain`. Each "_explicit" script does
the same thing with direct `tb.evolve / tb.attend / tb.measure` calls.
Both versions of each mode produce **bit-identical** results — diff them
to see exactly what the composer is doing.

Run any of them with `uv run python -m tb.examples.<name>`.

The perception example also exercises the **n_C concept measurement**
(the pink boxes in Figure 2 of Tensor_Brain.pdf): at each entity position
the model predicts both the entity index and its class (Person/Animal/Place).

## Tests

```sh
uv run pytest
```

## Layout

```
src/tb/
├── primitives.py    # TB module: evolve / attend / measure
├── evolve.py        # EvolveModule protocol + QTBEvolve, TBEvolve
├── indices.py       # IndexLayer (weight-tied), IndexGroups, masking
├── composers.py     # Position, decode_chain, decode_triple
└── examples/
    ├── _toy_data.py                  # Shared toy KG used by all modes
    ├── toy_kg.py                     # Episodic recall (composer)
    ├── toy_kg_explicit.py            # Episodic recall (direct primitives)
    ├── semantic_recall.py            # Semantic recall (composer)
    ├── semantic_recall_explicit.py   # Semantic recall (direct primitives)
    ├── perception.py                 # Perception + concept measurement (composer)
    └── perception_explicit.py        # Perception + concept measurement (direct primitives)
```

## A note on α

The three modes settle to different learned values of α:

- Perception (informative ν): α ≈ 0.3-0.6 (sensory drives most of q; prior shrinks)
- Semantic recall (no ν, ambiguous targets): α ≈ 1.0
- Episodic recall (no ν, specific targets): α ≈ 4.0 (prior carries the signal across positions)

This is consistent with the QTB §7.3 framing of α as a "prior strength" knob.


## Dataset enrichment (PVSG)

We use **PVSG** (panoptic video scene graphs: real video, tracked instance masks,
126 object classes = 115 thing + 11 stuff, 57 relations). PVSG's labels are flat,
so we add a hierarchy. We keep the enrichment to *one deterministic, inspectable
step* and separate three label layers so complexity does not compound:

- **A — Object taxonomy (we add this).** Three automatic levels per class from
  **WordNet**, no hand-curation: `leaf` (the PVSG class) → `mid` (a fixed
  depth-from-root cut: `car → conveyance`, `table → furnishing`, `bottle →
  container`) → `coarse` (WordNet supersense: `→ artifact`). Disambiguation is
  automatic (first physical-entity sense, so `table` → furniture not the data
  table). The output is a 126-row table you can read and check by eye; unmappable
  names (typos, the catch-all `others`) are surfaced, not guessed. WordNet is the
  recognized standard (VRD-E, ImageNet, Visual Genome use it), so this stays
  reproducible and comparable to prior work.
- **B — Observed relations (already in PVSG).** The 57 real predicates annotated
  on video. Rich and realistic by construction; we keep them as-is (they also
  cluster natively into spatial / contact / interaction — a free predicate
  grouping). We do **not** synthesize these.
- **C — Background knowledge (optional, later).** Non-observed facts
  (`dog is-a pet`, `dangerous`) — the modern replacement for VRD-E's hand-injected
  `Dangerous/Harmless`. Trained only through the semantic-memory pathway, not
  perception. Added only when a hypothesis needs the non-visual-prior effect.

**Taxonomy source is swappable** behind one `Taxonomy` contract (levels +
parent-maps + `IndexGroups`); switching sources touches no model code:

1. **WordNet** — *current*. Offline, automatic, standard, zero hand-curation.
2. **Wikidata (P279/P31)** — *TODO*. Live KG: subclass hierarchy **and** typed
   relations (would supply A + C together). Heavier (entity-linking + API).
3. **LLM-assisted tree** — *TODO, appendix only*. Cheapest to produce, but using
   an LLM for the *ground-truth* taxonomy hurts reproducibility/credibility
   (model drift, potential circularity), so keep it as a robustness check, not the
   primary labels.

Derivation code + a measured comparison to VRD-E's fixed B/P/G scheme live in
`experiments/pvsg_hierarchy/`.

## TODO / things to test: 

1. Modernize building blocks
  1. SAM instead of bounding boxes
  2. xLSTM / Mamba (state space models in general) instead of the RNN dynamic context layer
  3. Deep stacking between the representation and index layers

2. Creating and testing new datasets
  1. Ergo3D: millions of first-person videos, could test semantic and episodic memory, entitiy consistency and many other properties
  2. Knowledge graph datasets (ICEWS and similar, PyKEEN) as a simple benchmark for small experiments
  3. Action Genome

3. Implementing components which were only mentioned in the TB / QTB papers
  1. Grounding: explicit top-down inference

4. Testing general DL hypotheses from the QTB papers / related ideas
  1. Role of unistochastic matrices: convergence towards them, see Sinkformer and recent DeepSeek mHC paper
  2. Skip connections as Bayesian priors: how far can we take this hypothesis? How can we make it more precise?
  3. Testing the practical effect of the one-brain hypothesis