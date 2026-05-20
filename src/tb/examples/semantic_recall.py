"""Semantic recall (SKG mode).

The model is given a subject entity and must produce its most-likely
(object, predicate) associations from semantic memory — i.e., it
marginalizes over instances. This is the BTN's `forward_skg` mode
(Tensor_Brain.pdf §4.2): episodic memory is averaged into a single
"semantic memory embedding" ā, which seeds q before subject-conditional
decoding.

Setup:
    Seed q with a learnable ā (one global parameter).
    Position S : teacher-force to the queried subject (commit="teacher").
    Position O : sample / predict.
    Position P : sample / predict.

Loss is computed only on O and P (S is given). Because some subjects
appear in multiple triples, the "correct" object/predicate for a given
subject is not unique — semantic memory learns a *distribution*. We
report top-1 accuracy and also a top-1-amongst-valid-completions metric.

Run:
    uv run python -m tb.examples.semantic_recall
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tb.composers import Position, decode_chain
from tb.evolve import QTBEvolve
from tb.examples._toy_data import ENTITIES, PREDICATES, make_toy_kg
from tb.indices import IndexLayer
from tb.primitives import TB, learnable_alpha


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    groups, triples = make_toy_kg()
    dim = 64
    ent_off = groups.offsets[groups.names.index("entity")]
    pred_off = groups.offsets[groups.names.index("predicate")]

    tb = TB(
        dim=dim,
        index_layer=IndexLayer(num_indices=groups.num_indices, dim=dim),
        evolve_module=QTBEvolve(dim=dim, hidden=32, skip=True),
        alpha=learnable_alpha(1.0),
        beta=1.0,
    ).to(device)

    # Semantic memory embedding ā: one learnable vector that seeds every recall.
    a_bar = nn.Parameter(torch.zeros(dim, device=device))
    nn.init.normal_(a_bar, std=0.1)

    entity_mask = groups.mask("entity", device=device)
    predicate_mask = groups.mask("predicate", device=device)

    subjects = triples[:, 1].to(device)
    predicates = triples[:, 2].to(device)
    objects = triples[:, 3].to(device)
    n_train = triples.shape[0]

    opt = torch.optim.Adam([*tb.parameters(), a_bar], lr=5e-2)

    for step in range(400):
        opt.zero_grad()
        q0 = a_bar.unsqueeze(0).expand(n_train, -1)
        chain = decode_chain(
            tb,
            positions=[
                Position(typed_mask=entity_mask, teacher=subjects),  # S: forced
                Position(typed_mask=entity_mask),  # O
                Position(typed_mask=predicate_mask),  # P
            ],
            commit="expectation",
            initial_q=q0,
        )
        o_logits = chain.positions[1].typed.logits
        p_logits = chain.positions[2].typed.logits
        # No loss on S — it's teacher-forced.
        loss = F.cross_entropy(o_logits, objects) + F.cross_entropy(p_logits, predicates)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                o_acc = (o_logits.argmax(-1) == objects).float().mean().item()
                p_acc = (p_logits.argmax(-1) == predicates).float().mean().item()
            print(
                f"step {step:4d} | loss {loss.item():.3f} | "
                f"acc o/p {o_acc:.2f}/{p_acc:.2f} | α {tb.alpha.item():+.3f}"
            )

    # Per-subject report: show the model's distribution over completions.
    # Semantic memory with multiple valid completions per subject (e.g. Alice
    # appears in 3 triples) should put mass on each valid (p, o), not collapse
    # onto one. We verify this by inspecting top-k predictions.
    print("\nPer-subject distribution (top-3 predictions vs. ground-truth set):")
    unique_subjects = sorted(set(subjects.tolist()))
    with torch.no_grad():
        for s_global in unique_subjects:
            s_local = s_global - ent_off
            q0 = a_bar.unsqueeze(0)
            chain = decode_chain(
                tb,
                positions=[
                    Position(
                        typed_mask=entity_mask,
                        teacher=torch.tensor([s_global], device=device),
                    ),
                    Position(typed_mask=entity_mask),
                    Position(typed_mask=predicate_mask),
                ],
                commit="expectation",  # we want logit distributions, not commits
                initial_q=q0,
            )
            p_logits = chain.positions[2].typed.logits[0]
            o_logits = chain.positions[1].typed.logits[0]
            # Top-3 predicates and objects (ignoring -inf masked positions).
            top_p = torch.topk(p_logits, k=min(3, (p_logits > -1e9).sum().item()))
            top_o = torch.topk(o_logits, k=min(3, (o_logits > -1e9).sum().item()))

            true_triples = [
                (int(p) - pred_off, int(o) - ent_off)
                for s, p, o in zip(subjects, predicates, objects, strict=False)
                if int(s) == s_global
            ]
            truth_str = ", ".join(f"({PREDICATES[p]}, {ENTITIES[o]})" for p, o in true_triples)
            top_p_str = ", ".join(PREDICATES[int(i) - pred_off] for i in top_p.indices)
            top_o_str = ", ".join(ENTITIES[int(i) - ent_off] for i in top_o.indices)
            print(
                f"  {ENTITIES[s_local]:8s} | truth = {truth_str}\n"
                f"           | top-3 P: {top_p_str}\n"
                f"           | top-3 O: {top_o_str}"
            )


if __name__ == "__main__":
    main()
