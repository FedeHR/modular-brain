"""Perception mode — explicit version.

Identical to `tb.examples.perception`, but written with direct calls to
the three primitives. Two things are now clearly visible that the
composer hides:

  - The double `tb.measure` per S/O position: first the entity (gray box
    in Figure 2 of Tensor_Brain.pdf), then immediately the concept/class
    (pink box). Critically, there is no `evolve` between the two — the
    second measurement adds to the q that already has the entity's
    embedding mixed in.
  - The `attend` calls inject sensory ν *between* the evolve and the
    measurement: this is what TB Algorithm 1 (lines 17, 24, 29) does
    as `˜q ← μ · f(BB) + g(h)`.

Run:
    uv run python -m tb.examples.perception_explicit
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tb.evolve import QTBEvolve
from tb.examples._toy_data import CONCEPTS, describe, entity_to_class_global, make_toy_kg
from tb.indices import IndexLayer
from tb.primitives import TB, Commit, MeasureOutput, learnable_alpha


def make_fixed_features(num_indices: int, dim: int, seed: int = 42) -> torch.Tensor:
    """One pseudo-random unit-ish vector per index — the 'sensory signature'."""
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(num_indices, dim, generator=gen)
    return F.normalize(features, dim=-1) * (dim**0.5)


def decode_perception(
    tb: TB,
    nu_scene: torch.Tensor,
    nu_sub: torch.Tensor,
    nu_obj: torch.Tensor,
    nu_pred: torch.Tensor,
    instance_mask: torch.Tensor,
    entity_mask: torch.Tensor,
    concept_mask: torch.Tensor,
    predicate_mask: torch.Tensor,
    *,
    commit: Commit,
) -> tuple[
    MeasureOutput,  # T (instance)
    MeasureOutput,  # S entity
    MeasureOutput,  # S concept (class)
    MeasureOutput,  # O entity
    MeasureOutput,  # O concept (class)
    MeasureOutput,  # P (predicate)
]:
    """Full perception cycle: T, S+nC, O+nC, P. Sensory input at every position."""
    n_batch = nu_scene.shape[0]
    q = torch.zeros(n_batch, tb.dim, device=nu_scene.device)
    state = tb.init_state(n_batch, q.device)

    # --- Position T: scene → instance ---
    q, state = tb.evolve(q, state)
    q = tb.attend(q, nu=nu_scene, mu=1.0)
    t_out = tb.measure(q, mask=instance_mask, commit=commit)
    q = t_out.q

    # --- Position S: BB-sub → entity, then concept ---
    q, state = tb.evolve(q, state)
    q = tb.attend(q, nu=nu_sub, mu=1.0)
    s_entity = tb.measure(q, mask=entity_mask, commit=commit)
    q = s_entity.q
    # n_C concept measurement: NO evolve or attend in between.
    s_concept = tb.measure(q, mask=concept_mask, commit=commit)
    q = s_concept.q

    # --- Position O: BB-obj → entity, then concept ---
    q, state = tb.evolve(q, state)
    q = tb.attend(q, nu=nu_obj, mu=1.0)
    o_entity = tb.measure(q, mask=entity_mask, commit=commit)
    q = o_entity.q
    o_concept = tb.measure(q, mask=concept_mask, commit=commit)
    q = o_concept.q

    # --- Position P: BB-pred → predicate ---
    q, state = tb.evolve(q, state)
    q = tb.attend(q, nu=nu_pred, mu=1.0)
    p_out = tb.measure(q, mask=predicate_mask, commit=commit)
    q = p_out.q

    return t_out, s_entity, s_concept, o_entity, o_concept, p_out


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    groups, triples = make_toy_kg()
    dim = 64

    tb = TB(
        dim=dim,
        index_layer=IndexLayer(num_indices=groups.num_indices, dim=dim),
        evolve_module=QTBEvolve(dim=dim, hidden=32, skip=True),
        alpha=learnable_alpha(1.0),
        beta=1.0,
    ).to(device)

    features = make_fixed_features(groups.num_indices, dim).to(device)

    instance_mask = groups.mask("instance", device=device)
    entity_mask = groups.mask("entity", device=device)
    concept_mask = groups.mask("concept", device=device)
    predicate_mask = groups.mask("predicate", device=device)

    instances = triples[:, 0].to(device)
    subjects = triples[:, 1].to(device)
    predicates = triples[:, 2].to(device)
    objects = triples[:, 3].to(device)
    subj_class = entity_to_class_global(subjects, groups).to(device)
    obj_class = entity_to_class_global(objects, groups).to(device)

    opt = torch.optim.Adam(tb.parameters(), lr=5e-2)

    for step in range(500):
        opt.zero_grad()
        t_out, s_ent, s_cls, o_ent, o_cls, p_out = decode_perception(
            tb,
            features[instances],
            features[subjects],
            features[objects],
            features[predicates],
            instance_mask,
            entity_mask,
            concept_mask,
            predicate_mask,
            commit="expectation",
        )
        loss = (
            F.cross_entropy(t_out.logits, instances)
            + F.cross_entropy(s_ent.logits, subjects)
            + F.cross_entropy(s_cls.logits, subj_class)
            + F.cross_entropy(o_ent.logits, objects)
            + F.cross_entropy(o_cls.logits, obj_class)
            + F.cross_entropy(p_out.logits, predicates)
        )
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                t_acc = (t_out.logits.argmax(-1) == instances).float().mean().item()
                s_acc = (s_ent.logits.argmax(-1) == subjects).float().mean().item()
                sc_acc = (s_cls.logits.argmax(-1) == subj_class).float().mean().item()
                o_acc = (o_ent.logits.argmax(-1) == objects).float().mean().item()
                oc_acc = (o_cls.logits.argmax(-1) == obj_class).float().mean().item()
                p_acc = (p_out.logits.argmax(-1) == predicates).float().mean().item()
            print(
                f"step {step:4d} | loss {loss.item():.3f} | "
                f"acc t/s+c/o+c/p {t_acc:.2f}/{s_acc:.2f}+{sc_acc:.2f}/"
                f"{o_acc:.2f}+{oc_acc:.2f}/{p_acc:.2f} | α {tb.alpha.item():+.3f}"
            )

    print("\nFinal perception (argmax) with concept (class) labels:")
    with torch.no_grad():
        t_out, s_ent, s_cls, o_ent, o_cls, p_out = decode_perception(
            tb,
            features[instances],
            features[subjects],
            features[objects],
            features[predicates],
            instance_mask,
            entity_mask,
            concept_mask,
            predicate_mask,
            commit="argmax",
        )
        concept_off = groups.offsets[groups.names.index("concept")]
        for i in range(triples.shape[0]):
            truth = (
                int(instances[i]),
                int(subjects[i]),
                int(predicates[i]),
                int(objects[i]),
            )
            pred = (
                int(t_out.k[i]),
                int(s_ent.k[i]),
                int(p_out.k[i]),
                int(o_ent.k[i]),
            )
            sc_name = CONCEPTS[int(s_cls.k[i]) - concept_off]
            oc_name = CONCEPTS[int(o_cls.k[i]) - concept_off]
            ok = "✓" if pred == truth else "✗"
            print(
                f"  {ok}  truth = {describe(truth, groups)}   [classes: S={sc_name}, O={oc_name}]"
            )


if __name__ == "__main__":
    main()
