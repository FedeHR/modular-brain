"""Object-class taxonomy (layer A), derived automatically from WordNet.

One deterministic step per class, no hand-curation:

    name  ->  first physical-entity noun sense  ->  WordNet supersense
    "dog" ->  dog.n.01                          ->  noun.animal -> "animal"

giving two comparable levels: `leaf` (the PVSG class) and `coarse` (the WordNet
supersense). Disambiguation is automatic — we take the first sense whose
hypernyms include `physical_entity`, which rejects abstract senses that misfire
for a vision vocabulary (e.g. `table` → the *data* table). No override lists, no
anchor lists, no LLM. Unmappable names (dataset typos, the catch-all `others`)
are reported, never silently guessed.

Source-swappable: everything downstream (the TB's concept `IndexGroups`, the
hierarchical-consistency metric) talks only to the `Taxonomy` object below, so a
Wikidata or LLM producer can fill the same object later without touching model
code (see README, "Dataset enrichment").
"""

from __future__ import annotations

from dataclasses import dataclass

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

_PHYSICAL = wn.synset("physical_entity.n.01")


def physical_synset(name: str) -> Synset | None:
    """First physical-entity noun sense of `name` (automatic disambiguation)."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    cands = wn.synsets(key, pos=wn.NOUN)
    if not cands and "_" in key:  # compound -> back off to the head noun
        cands = wn.synsets(key.split("_")[-1], pos=wn.NOUN)
    for ss in cands:
        if any(_PHYSICAL in path for path in ss.hypernym_paths()):
            return ss
    return cands[0] if cands else None


def supersense_of(ss: Synset) -> str:
    """WordNet lexicographer category: 'noun.animal' -> 'animal'. The 25 unique
    beginners carry 'noun.Tops'; for those use the lemma (person -> 'person')."""
    lex = ss.lexname().split(".", 1)[1]
    return ss.name().split(".", 1)[0] if lex == "Tops" else lex


def ancestor_at_depth(ss: Synset, depth: int) -> str:
    """Label of the ancestor `depth` steps from the root (entity = depth 0).

    A same-depth cut is comparable across classes by construction and nests under
    the supersense (which sits near the top), giving a middle granularity that
    splits broad supersenses — `car → vehicle`, `cup → vessel`, `table →
    furniture`. Shallow classes clamp to their own leaf. This is the standard
    automatic way to read a fixed level out of WordNet; no hand-curation.
    """
    path = max(ss.hypernym_paths(), key=len)  # root -> ... -> leaf
    return path[min(depth, len(path) - 1)].name().split(".", 1)[0]


@dataclass(frozen=True)
class ClassEntry:
    name: str
    synset: str | None            # e.g. "dog.n.01"; None if unmappable
    coarse: str | None            # supersense; None if unmappable
    mid: str | None = None         # fixed depth-from-root cut; None if unmappable
    chain: tuple[str, ...] = ()    # leaf->root lemmas, for eyeball inspection


@dataclass
class Taxonomy:
    """Source-agnostic taxonomy contract the TB consumes.

    `levels` runs fine -> coarse: ('leaf', 'mid', 'coarse').
    """

    entries: list[ClassEntry]
    levels: tuple[str, ...] = ("leaf", "mid", "coarse")

    _ATTR = {"leaf": "name", "mid": "mid", "coarse": "coarse"}

    def vocab(self, level: str) -> list[str]:
        attr = self._ATTR[level]
        return sorted({getattr(e, attr) for e in self.entries if getattr(e, attr)})

    def parent_map(self, child: str = "leaf", parent: str = "coarse") -> dict[str, str]:
        """{child-label -> parent-label} for the hierarchical-consistency metric.
        Defaults to leaf->coarse; pass ('leaf','mid') or ('mid','coarse') for the
        adjacent steps."""
        ca, pa = self._ATTR[child], self._ATTR[parent]
        return {
            getattr(e, ca): getattr(e, pa)
            for e in self.entries
            if getattr(e, ca) and getattr(e, pa)
        }

    def index_groups(self):
        """TB `IndexGroups`: one contiguous concept group per level."""
        from tb.indices import IndexGroups

        return IndexGroups.from_sizes([(lvl, len(self.vocab(lvl))) for lvl in self.levels])

    @property
    def unmapped(self) -> list[str]:
        return [e.name for e in self.entries if e.coarse is None]

    def coverage(self) -> dict[str, float]:
        n = len(self.entries)
        mapped = sum(e.coarse is not None for e in self.entries)
        out: dict[str, float] = {"classes": n, "mapped": mapped / n}
        for lvl in self.levels:
            out[f"n_{lvl}"] = len(self.vocab(lvl))
        return out


def build_taxonomy(class_names: list[str], *, mid_depth: int = 6) -> Taxonomy:
    """Leaf + `mid` (depth-`mid_depth` cut) + `coarse` (supersense), automatic.

    depth 6 splits the broad `artifact` supersense into meaningful groups
    (container, furnishing, implement, conveyance, ware, device …) while shallow
    branches (animal/plant/food) collapse mid→coarse, i.e. no spurious split.
    """
    entries: list[ClassEntry] = []
    for name in class_names:
        ss = physical_synset(name)
        if ss is None:
            entries.append(ClassEntry(name, None, None))
            continue
        chain = tuple(
            n.name().split(".", 1)[0] for n in reversed(max(ss.hypernym_paths(), key=len))
        )
        entries.append(
            ClassEntry(
                name=name,
                synset=ss.name(),
                coarse=supersense_of(ss),
                mid=ancestor_at_depth(ss, mid_depth),
                chain=chain,
            )
        )
    return Taxonomy(entries=entries)
