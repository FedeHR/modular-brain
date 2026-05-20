"""Episodic recall (TKG mode).

The model is given an episodic instance index `t` and must reproduce the
triple stored at that instance. This is the BTN's `forward_tkg` mode
(Tensor_Brain.pdf §4.4):

    Seed q with the episodic embedding a_t.
    Decode (S, O, P) by composing primitives — no sensory input (μ = 0).

This is the simplest of the three operational modes and exercises the
primitive loop end-to-end. Convergence to 100% on the training triples
in a few hundred steps is the smoke test.

Run:
    uv run python -m tb.examples.toy_kg
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tb.composers import Position, decode_chain
from tb.evolve import QTBEvolve
from tb.examples._toy_data import describe, make_toy_kg
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha


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

    entity_mask = groups.mask("entity", device=device)
    predicate_mask = groups.mask("predicate", device=device)

    instances = triples[:, 0].to(device)
    subjects = triples[:, 1].to(device)
    predicates = triples[:, 2].to(device)
    objects = triples[:, 3].to(device)

    opt = torch.optim.Adam(tb.parameters(), lr=5e-2)

    for step in range(400):
        opt.zero_grad()
        # Seed q with the episodic embedding for each requested instance.
        q0 = tb.index_layer.embed(instances)
        chain = decode_chain(
            tb,
            positions=[
                Position(typed_mask=entity_mask),  # subject
                Position(typed_mask=entity_mask),  # object
                Position(typed_mask=predicate_mask),  # predicate
            ],
            commit="expectation",
            initial_q=q0,
        )
        s_logits = chain.positions[0].typed.logits
        o_logits = chain.positions[1].typed.logits
        p_logits = chain.positions[2].typed.logits
        loss = (
            F.cross_entropy(s_logits, subjects)
            + F.cross_entropy(o_logits, objects)
            + F.cross_entropy(p_logits, predicates)
        )
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                s_acc = (s_logits.argmax(-1) == subjects).float().mean().item()
                o_acc = (o_logits.argmax(-1) == objects).float().mean().item()
                p_acc = (p_logits.argmax(-1) == predicates).float().mean().item()
            print(
                f"step {step:4d} | loss {loss.item():.3f} | "
                f"acc s/o/p {s_acc:.2f}/{o_acc:.2f}/{p_acc:.2f} | α {float(tb.alpha):+.3f}"
            )

    # Inference with hard argmax.
    print("\nFinal recall (argmax):")
    with torch.no_grad():
        q0 = tb.index_layer.embed(instances)
        chain = decode_chain(
            tb,
            positions=[
                Position(typed_mask=entity_mask),
                Position(typed_mask=entity_mask),
                Position(typed_mask=predicate_mask),
            ],
            commit="argmax",
            initial_q=q0,
        )
        s_pred = chain.positions[0].typed.k
        o_pred = chain.positions[1].typed.k
        p_pred = chain.positions[2].typed.k
        for i in range(triples.shape[0]):
            truth = (int(instances[i]), int(subjects[i]), int(predicates[i]), int(objects[i]))
            pred = (int(instances[i]), int(s_pred[i]), int(p_pred[i]), int(o_pred[i]))
            ok = "✓" if pred == truth else "✗"
            print(f"  {ok}  truth = {describe(truth, groups)}")


if __name__ == "__main__":
    main()
