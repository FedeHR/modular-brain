"""Shared toy KG used by all mode examples.

Four index groups, one tiny knowledge graph:
    - 10 episodic instances (one per triple),    [0, 10)
    - 8  entities (Alice, Bob, Cat, ...),         [10, 18)
    - 3  concepts/classes (Person, Animal, Place), [18, 21)
    - 4  predicates (owns, knows, lives_in, eats), [21, 25)

The concept group is the n_C ("unary label") group from Figure 2 of
Tensor_Brain.pdf: at each S/O position the BTN measures both the entity
*and* its class. Each entity has a single ground-truth class.

The same 10 triples are reused across the episodic-recall, semantic-recall,
and perception examples so that mode behavior can be compared on identical
data.
"""

from __future__ import annotations

import torch

from tb.indices import IndexGroups

ENTITIES = ["Alice", "Bob", "Cat", "Dog", "Park", "House", "Fish", "Bird"]
CONCEPTS = ["Person", "Animal", "Place"]
PREDICATES = ["owns", "knows", "lives_in", "eats"]

# Class membership for each entity (local concept indices).
ENTITY_CLASS_LOCAL: list[int] = [
    0,  # Alice  → Person
    0,  # Bob    → Person
    1,  # Cat    → Animal
    1,  # Dog    → Animal
    2,  # Park   → Place
    2,  # House  → Place
    1,  # Fish   → Animal
    1,  # Bird   → Animal
]

# (subject_local, predicate_local, object_local).
TRIPLES_LOCAL: list[tuple[int, int, int]] = [
    (0, 0, 3),  # Alice owns Dog
    (1, 0, 2),  # Bob   owns Cat
    (0, 1, 1),  # Alice knows Bob
    (0, 2, 5),  # Alice lives_in House
    (1, 2, 5),  # Bob   lives_in House
    (3, 2, 4),  # Dog   lives_in Park
    (2, 3, 6),  # Cat   eats Fish
    (7, 3, 6),  # Bird  eats Fish
    (1, 1, 0),  # Bob   knows Alice
    (3, 1, 2),  # Dog   knows Cat
]


def make_toy_kg() -> tuple[IndexGroups, torch.Tensor]:
    """Return (groups, triples). triples is shape (T, 4) — columns (t, s, p, o)
    in global indices ready for use with masked measurements.
    """
    groups = IndexGroups.from_sizes(
        [
            ("instance", 10),
            ("entity", len(ENTITIES)),
            ("concept", len(CONCEPTS)),
            ("predicate", len(PREDICATES)),
        ]
    )
    ent_off = groups.offsets[groups.names.index("entity")]
    pred_off = groups.offsets[groups.names.index("predicate")]
    triples = torch.tensor(
        [(t, s + ent_off, p + pred_off, o + ent_off) for t, (s, p, o) in enumerate(TRIPLES_LOCAL)],
        dtype=torch.long,
    )
    return groups, triples


def entity_to_class_global(entity_global: torch.Tensor, groups: IndexGroups) -> torch.Tensor:
    """Map entity global indices → concept (class) global indices."""
    ent_off = groups.offsets[groups.names.index("entity")]
    concept_off = groups.offsets[groups.names.index("concept")]
    classes_local = torch.tensor(ENTITY_CLASS_LOCAL, dtype=torch.long)
    local = entity_global - ent_off
    return classes_local[local] + concept_off


def describe(triple: tuple[int, int, int, int], groups: IndexGroups) -> str:
    """Human-readable rendering of a (t, s, p, o) tuple of global indices."""
    t, s, p, o = triple
    ent_off = groups.offsets[groups.names.index("entity")]
    pred_off = groups.offsets[groups.names.index("predicate")]
    return f"t={t}: {ENTITIES[s - ent_off]} {PREDICATES[p - pred_off]} {ENTITIES[o - ent_off]}"
