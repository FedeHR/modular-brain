"""Invariants for the PVSG instance → concept-space adapter.

Run: uv run pytest experiments/pvsg_hierarchy/test_pvsg_data.py
"""

from __future__ import annotations

from experiments.pvsg_hierarchy.pvsg_data import (
    build_concept_space,
    load_instances,
    load_triples,
)


def test_concept_space_layout():
    cs = build_concept_space()
    g = cs.groups
    assert g.names == ("leaf", "mid", "coarse", "predicate")
    sizes = dict(zip(g.names, g.sizes))
    assert sizes["coarse"] == len(cs.tax.vocab("coarse"))
    assert sizes["predicate"] == 57  # PVSG relation classes


def test_gindex_lands_in_its_group():
    cs = build_concept_space()
    gi = cs.gindex("coarse", "animal")
    s = cs.groups.slice_of("coarse")
    assert s.start <= gi < s.stop


def test_instances_drop_unmapped_and_stay_on_taxonomy_path():
    cs = build_concept_space()
    inst, dropped = load_instances(cs=cs)
    assert dropped > 0  # 'others' + typos can't map → dropped, not guessed
    assert inst and all(i.coarse for i in inst)
    by = {e.name: e for e in cs.tax.entries}
    for i in inst[:200]:
        assert by[i.leaf].mid == i.mid and by[i.leaf].coarse == i.coarse


def test_triples_reference_object_ids():
    trips = load_triples()
    assert trips and isinstance(trips[0].predicate, str)
    assert isinstance(trips[0].subj_id, int) and isinstance(trips[0].obj_id, int)
