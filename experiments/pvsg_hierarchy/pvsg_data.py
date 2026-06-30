"""Map PVSG per-object instances through the WordNet taxonomy into the TB's
concept index space.

`pvsg_instances.json` is the compact slice of the dataset's `pvsg.json` we need:
per video, `objects` (id, category) and `relations` ([subject_id, object_id,
predicate]). Each object's `category` is looked up in the `Taxonomy` to attach
its three concept levels (leaf / mid / coarse). Categories WordNet can't map
(typos, the catch-all `others`) are dropped and counted — never guessed.

Output for the TB:
- `ConceptSpace`: one `IndexGroups` with three contiguous concept groups
  (leaf, mid, coarse) plus a predicate group, and the maps to turn a label into a
  *global* index and back. This is the n_C concept space the TB decodes.
- `instances`: per-object (leaf, mid, coarse) global-index triples.
- `triples`:   per-relation (subject-instance, predicate, object-instance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.pvsg_hierarchy.taxonomy import Taxonomy, build_taxonomy

HERE = Path(__file__).parent
LEVELS = ("leaf", "mid", "coarse")


@dataclass(frozen=True)
class ObjectInstance:
    video: str
    obj_id: int
    leaf: str
    mid: str
    coarse: str


@dataclass(frozen=True)
class Triple:
    video: str
    subj_id: int
    predicate: str
    obj_id: int


@dataclass
class ConceptSpace:
    """The TB concept index space derived from the taxonomy + PVSG predicates."""

    groups: object                 # tb.indices.IndexGroups (leaf, mid, coarse, predicate)
    label_to_global: dict[tuple[str, str], int]  # (level, label) -> global index
    tax: Taxonomy

    def gindex(self, level: str, label: str) -> int:
        return self.label_to_global[(level, label)]


def _load_raw() -> dict:
    return json.load(open(HERE / "pvsg_instances.json"))


def build_concept_space(raw: dict | None = None) -> ConceptSpace:
    from tb.indices import IndexGroups

    raw = raw or _load_raw()
    tax = build_taxonomy(raw["objects"]["thing"] + raw["objects"]["stuff"])
    predicates = sorted(raw["relations"])

    sizes = [(lvl, len(tax.vocab(lvl))) for lvl in LEVELS]
    sizes.append(("predicate", len(predicates)))
    groups = IndexGroups.from_sizes(sizes)

    label_to_global: dict[tuple[str, str], int] = {}
    for lvl in LEVELS:
        off = groups.offsets[groups.names.index(lvl)]
        for i, label in enumerate(tax.vocab(lvl)):
            label_to_global[(lvl, label)] = off + i
    poff = groups.offsets[groups.names.index("predicate")]
    for i, p in enumerate(predicates):
        label_to_global[("predicate", p)] = poff + i

    return ConceptSpace(groups=groups, label_to_global=label_to_global, tax=tax)


def load_instances(raw: dict | None = None, cs: ConceptSpace | None = None):
    """Return (instances, dropped_count). Each instance carries leaf/mid/coarse
    labels for objects whose category maps cleanly through the taxonomy."""
    raw = raw or _load_raw()
    cs = cs or build_concept_space(raw)
    entry = {e.name: e for e in cs.tax.entries}
    instances: list[ObjectInstance] = []
    dropped = 0
    for v in raw["videos"]:
        for o in v["objects"]:
            e = entry.get(o["cat"])
            if e is None or e.coarse is None:  # unmapped category -> skip
                dropped += 1
                continue
            instances.append(ObjectInstance(v["video_id"], o["id"], e.name, e.mid, e.coarse))
    return instances, dropped


def load_triples(raw: dict | None = None):
    """Per-relation (subject_id, predicate, object_id) within each video."""
    raw = raw or _load_raw()
    triples: list[Triple] = []
    for v in raw["videos"]:
        for s, o, p in v["relations"]:
            triples.append(Triple(v["video_id"], s, p, o))
    return triples
