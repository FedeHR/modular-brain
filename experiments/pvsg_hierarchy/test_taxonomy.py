"""Invariants for the automatic WordNet taxonomy (layer A).

Run: uv run pytest experiments/pvsg_hierarchy/test_taxonomy.py
"""

from __future__ import annotations

from experiments.pvsg_hierarchy.taxonomy import build_taxonomy


def _by_name(tax):
    return {e.name: e for e in tax.entries}


def test_physical_filter_beats_frequency_sense():
    # 'table'/'board' first WordNet sense is abstract; the physical filter must
    # pick the object sense automatically (no override list).
    tax = build_taxonomy(["table"])
    assert _by_name(tax)["table"].synset == "table.n.02"  # furniture, not data table
    assert _by_name(tax)["table"].coarse == "artifact"


def test_tops_fallback():
    # person.n.01 has lexname 'noun.Tops'; must not leak through.
    tax = build_taxonomy(["person", "adult"])
    assert _by_name(tax)["person"].coarse == "person"


def test_known_supersenses():
    tax = build_taxonomy(["dog", "car", "tree", "bread"])
    e = _by_name(tax)
    assert e["dog"].coarse == "animal"
    assert e["car"].coarse == "artifact"
    assert e["tree"].coarse == "plant"
    assert e["bread"].coarse == "food"


def test_unmapped_is_surfaced_not_guessed():
    tax = build_taxonomy(["ballon", "others", "dog"])
    assert set(tax.unmapped) == {"ballon", "others"}
    assert _by_name(tax)["dog"].coarse == "animal"


def test_mid_level_splits_broad_supersense_and_nests():
    # depth-6 mid refines the broad artifact bucket into meaningful groups,
    # and stays nested under the coarse supersense.
    tax = build_taxonomy(["car", "bottle", "table", "dog"])
    e = _by_name(tax)
    assert e["car"].mid == "conveyance"
    assert e["bottle"].mid == "container"
    assert e["table"].mid == "furnishing"
    # shallow branch: mid collapses onto coarse rather than inventing a split.
    assert e["dog"].mid == e["dog"].coarse == "animal"
    # nesting: every leaf's mid label is on its hypernym chain above the leaf.
    for x in tax.entries:
        if x.mid and x.coarse and x.mid != x.coarse:
            assert x.chain.index(x.mid) < x.chain.index(x.coarse)


def test_taxonomy_bridge_to_tb():
    tax = build_taxonomy(["dog", "car", "tree", "person"])
    groups = tax.index_groups()
    assert groups.names == ("leaf", "mid", "coarse")
    assert groups.num_indices == sum(len(tax.vocab(l)) for l in ("leaf", "mid", "coarse"))
    pm = tax.parent_map()  # leaf -> coarse by default
    assert pm["dog"] == "animal" and pm["car"] == "artifact"
    pm_mid = tax.parent_map("leaf", "mid")
    assert pm_mid["car"] == "conveyance"


def test_chain_is_inspectable():
    tax = build_taxonomy(["dog"])
    chain = _by_name(tax)["dog"].chain
    assert chain[0] == "dog" and chain[-1] == "entity"
