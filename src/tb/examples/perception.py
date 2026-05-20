"""Perception mode (with concept measurement).

Each index has a deterministic "sensory signature" — a fixed pseudo-random
feature vector standing in for what a DCNN would extract from an image.
At each position the TB receives this signature as ν and must decode the
correct index. At the S and O positions we *also* perform the n_C concept
measurement (Tensor_Brain.pdf Figure 2, pink boxes): given the subject's
post-measurement q, decode its class (Person / Animal / Place).

Setup (perception branch of Algorithm 1):
    Position T : ν = scene-features ;  decode instance.
    Position S : ν = BB-subject     ;  decode entity ; then decode concept (class).
    Position O : ν = BB-object      ;  decode entity ; then decode concept (class).
    Position P : ν = BB-predicate   ;  decode predicate.

q is seeded to zero (no prior episodic context at the start of perception).
Each position's correct answer is recoverable only by attending to ν.

Run:
    uv run python -m tb.examples.perception
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tb.composers import Position, decode_chain
from tb.evolve import QTBEvolve
from tb.examples._toy_data import describe, entity_to_class_global, make_toy_kg
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha


def make_fixed_features(num_indices: int, dim: int, seed: int = 42) -> torch.Tensor:
    """One pseudo-random unit-ish vector per index — the 'sensory signature'.

    These are the fixed, ground-truth perceptual signatures. The TB has
    to learn embeddings A such that the right index gets high softmax
    probability when its signature appears as ν.
    """
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(num_indices, dim, generator=gen)
    return F.normalize(features, dim=-1) * (dim**0.5)


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
    # Ground-truth classes for subject and object.
    subj_class = entity_to_class_global(subjects, groups).to(device)
    obj_class = entity_to_class_global(objects, groups).to(device)
    n_train = triples.shape[0]

    opt = torch.optim.Adam(tb.parameters(), lr=5e-2)

    for step in range(500):
        opt.zero_grad()
        chain = decode_chain(
            tb,
            positions=[
                # T: scene features → instance
                Position(nu=features[instances], typed_mask=instance_mask),
                # S: BB-sub features → entity, then concept (class)
                Position(
                    nu=features[subjects],
                    typed_mask=entity_mask,
                    concept_mask=concept_mask,
                ),
                # O: BB-obj features → entity, then concept (class)
                Position(
                    nu=features[objects],
                    typed_mask=entity_mask,
                    concept_mask=concept_mask,
                ),
                # P: BB-pred features → predicate
                Position(nu=features[predicates], typed_mask=predicate_mask),
            ],
            commit="expectation",
            initial_q=torch.zeros(n_train, dim, device=device),
        )
        t_logits = chain.positions[0].typed.logits
        s_logits = chain.positions[1].typed.logits
        s_cls_logits = chain.positions[1].concept.logits
        o_logits = chain.positions[2].typed.logits
        o_cls_logits = chain.positions[2].concept.logits
        p_logits = chain.positions[3].typed.logits

        loss = (
            F.cross_entropy(t_logits, instances)
            + F.cross_entropy(s_logits, subjects)
            + F.cross_entropy(s_cls_logits, subj_class)
            + F.cross_entropy(o_logits, objects)
            + F.cross_entropy(o_cls_logits, obj_class)
            + F.cross_entropy(p_logits, predicates)
        )
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                t_acc = (t_logits.argmax(-1) == instances).float().mean().item()
                s_acc = (s_logits.argmax(-1) == subjects).float().mean().item()
                sc_acc = (s_cls_logits.argmax(-1) == subj_class).float().mean().item()
                o_acc = (o_logits.argmax(-1) == objects).float().mean().item()
                oc_acc = (o_cls_logits.argmax(-1) == obj_class).float().mean().item()
                p_acc = (p_logits.argmax(-1) == predicates).float().mean().item()
            print(
                f"step {step:4d} | loss {loss.item():.3f} | "
                f"acc t/s+c/o+c/p {t_acc:.2f}/{s_acc:.2f}+{sc_acc:.2f}/"
                f"{o_acc:.2f}+{oc_acc:.2f}/{p_acc:.2f} | α {tb.alpha.item():+.3f}"
            )

    # Inference: argmax decoding.
    print("\nFinal perception (argmax) with concept (class) labels:")
    with torch.no_grad():
        chain = decode_chain(
            tb,
            positions=[
                Position(nu=features[instances], typed_mask=instance_mask),
                Position(
                    nu=features[subjects],
                    typed_mask=entity_mask,
                    concept_mask=concept_mask,
                ),
                Position(
                    nu=features[objects],
                    typed_mask=entity_mask,
                    concept_mask=concept_mask,
                ),
                Position(nu=features[predicates], typed_mask=predicate_mask),
            ],
            commit="argmax",
            initial_q=torch.zeros(n_train, dim, device=device),
        )
        t_pred = chain.positions[0].typed.k
        s_pred = chain.positions[1].typed.k
        sc_pred = chain.positions[1].concept.k
        o_pred = chain.positions[2].typed.k
        oc_pred = chain.positions[2].concept.k
        p_pred = chain.positions[3].typed.k
        from tb.examples._toy_data import CONCEPTS

        concept_off = groups.offsets[groups.names.index("concept")]
        for i in range(n_train):
            truth = (int(instances[i]), int(subjects[i]), int(predicates[i]), int(objects[i]))
            pred = (int(t_pred[i]), int(s_pred[i]), int(p_pred[i]), int(o_pred[i]))
            sc_name = CONCEPTS[int(sc_pred[i]) - concept_off]
            oc_name = CONCEPTS[int(oc_pred[i]) - concept_off]
            ok = "✓" if pred == truth else "✗"
            print(
                f"  {ok}  truth = {describe(truth, groups)}   [classes: S={sc_name}, O={oc_name}]"
            )


if __name__ == "__main__":
    main()
