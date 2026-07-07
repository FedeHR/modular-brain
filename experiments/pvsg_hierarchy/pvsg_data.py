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
        for rel in v["relations"]:
            s, o, p = rel[:3]  # rel[3] (spans, if present) is used by V7 tooling
            triples.append(Triple(v["video_id"], s, p, o))
    return triples


# --- temporal relation table (V2 labels, V7 boundaries) -------------------------

@dataclass(frozen=True)
class RelationSpan:
    """One contiguous stretch where a triple holds: frames [start, end] incl."""

    video: str
    subj_id: int
    predicate: str
    obj_id: int
    start: int
    end: int


@dataclass(frozen=True)
class Boundary:
    """An *interior* relation-state change: the triple starts or stops inside
    the video (not at frame 0 / the last frame). The unit of analysis for V7."""

    video: str
    frame: int          # first frame of the new state
    subj_id: int
    predicate: str
    obj_id: int
    kind: str           # "start" (triple begins) | "end" (triple stops)


def load_relation_spans(raw: dict | None = None) -> list[RelationSpan]:
    """Flatten every relation's span list. Requires the span-preserving
    `pvsg_instances.json` (regenerate via `make_instances.py` if this raises)."""
    raw = raw or _load_raw()
    spans: list[RelationSpan] = []
    for v in raw["videos"]:
        for rel in v["relations"]:
            if len(rel) < 4:
                raise ValueError(
                    f"relation {rel!r} in {v['video_id']} has no spans — "
                    "regenerate pvsg_instances.json with make_instances.py"
                )
            s, o, p, sp = rel
            for a, b in sp:
                spans.append(RelationSpan(v["video_id"], s, p, o, a, b))
    return spans


def relations_at(spans: list[RelationSpan], video: str, frame: int) -> set[tuple[int, str, int]]:
    """The set of (subj_id, predicate, obj_id) true at `frame` — the per-frame
    relation target for V2."""
    return {(r.subj_id, r.predicate, r.obj_id)
            for r in spans
            if r.video == video and r.start <= frame <= r.end}


def load_boundaries(raw: dict | None = None) -> list[Boundary]:
    """Interior state changes, matching `make_instances.py`'s census logic:
    a span start > 0, or a span end before the video's last frame."""
    raw = raw or _load_raw()
    out: list[Boundary] = []
    for v in raw["videos"]:
        spans = [rel[3] for rel in v["relations"]]
        horizon = v.get("num_frames") or (
            max((e for sp in spans for _, e in sp), default=0) + 1)
        for rel in v["relations"]:
            s, o, p, sp = rel
            for a, b in sp:
                if a > 0:
                    out.append(Boundary(v["video_id"], a, s, p, o, "start"))
                if b < horizon - 1:
                    out.append(Boundary(v["video_id"], b + 1, s, p, o, "end"))
    return out


# --- video index space: concepts + entities + episodes (V1/V3 machinery) --------

@dataclass
class VideoSpace:
    """`ConceptSpace` extended with the paper's remaining index kinds: one
    *entity* index per tracked instance and one *episode* index per video.

    The concept groups come first and keep exactly the `ConceptSpace` offsets,
    so every mask/gindex computed against `cs` stays valid against `groups`.
    """

    groups: object                                # IndexGroups (+entity, +episode)
    cs: ConceptSpace
    entity_to_global: dict[tuple[str, int], int]  # (video, obj_id) -> global index
    episode_to_global: dict[str, int]             # video -> global index

    def gindex(self, level: str, label: str) -> int:
        return self.cs.gindex(level, label)

    def entity_gindex(self, video: str, obj_id: int) -> int:
        return self.entity_to_global[(video, obj_id)]

    def episode_gindex(self, video: str) -> int:
        return self.episode_to_global[video]


def build_video_space(videos: list[str] | None = None, raw: dict | None = None,
                      cs: ConceptSpace | None = None) -> VideoSpace:
    """Index space over a chosen video subset (default: all videos in the slice).

    Entities are the taxonomy-mapped instances of those videos (same drop rule
    as `load_instances` — unmapped categories are excluded, never guessed);
    episodes are whole videos, one index each (finer windows are a later knob).
    """
    from tb.indices import IndexGroups

    raw = raw or _load_raw()
    cs = cs or build_concept_space(raw)
    wanted = set(videos) if videos is not None else {v["video_id"] for v in raw["videos"]}

    instances, _ = load_instances(raw, cs)
    entities = [(i.video, i.obj_id) for i in instances if i.video in wanted]
    episodes = sorted(wanted)

    named = list(zip(cs.groups.names, cs.groups.sizes))
    named += [("entity", len(entities)), ("episode", len(episodes))]
    groups = IndexGroups.from_sizes(named)

    eoff = groups.offsets[groups.names.index("entity")]
    entity_to_global = {key: eoff + i for i, key in enumerate(entities)}
    poff = groups.offsets[groups.names.index("episode")]
    episode_to_global = {vid: poff + i for i, vid in enumerate(episodes)}

    return VideoSpace(groups=groups, cs=cs,
                      entity_to_global=entity_to_global,
                      episode_to_global=episode_to_global)
