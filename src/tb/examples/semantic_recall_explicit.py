"""Semantic recall (SKG mode) — explicit version.

Identical to `tb.examples.semantic_recall`, but written with direct calls
to the three primitives instead of `decode_chain`. The teacher-forcing
of the subject position (the SKG-defining trait — we *give* the model
the subject and ask for the marginal P/O distribution) is visible as a
direct `commit="teacher"` call.

Run:
    uv run python -m tb.examples.semantic_recall_explicit
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tb.evolve import QTBEvolve
from tb.examples._toy_data import ENTITIES, PREDICATES, make_toy_kg
from tb.indices import IndexLayer
from tb.primitives import TB, Commit, MeasureOutput, learnable_alpha


def decode_semantic(
    tb: TB,
    a_bar: torch.Tensor,
    subjects: torch.Tensor,
    entity_mask: torch.Tensor,
    predicate_mask: torch.Tensor,
    *,
    commit: Commit,
) -> tuple[MeasureOutput, MeasureOutput, MeasureOutput]:
    """Decode (S | O | P) conditioned on the given subject and semantic ā.

    The subject is teacher-forced: we *give* the model which subject the
    query is about. The model then predicts a marginal distribution over
    (object, predicate). This is SKG mode.
    """
    # Seed q with the semantic memory embedding ā, broadcast over batch.
    n_batch = subjects.shape[0]
    q = a_bar.unsqueeze(0).expand(n_batch, -1)
    state = tb.init_state(n_batch, q.device)

    # --- Position S: subject (teacher-forced) ---
    q, state = tb.evolve(q, state)
    subject = tb.measure(q, mask=entity_mask, commit="teacher", teacher_k=subjects)
    q = subject.q

    # --- Position O: object ---
    q, state = tb.evolve(q, state)
    obj = tb.measure(q, mask=entity_mask, commit=commit)
    q = obj.q

    # --- Position P: predicate ---
    q, state = tb.evolve(q, state)
    predicate = tb.measure(q, mask=predicate_mask, commit=commit)
    q = predicate.q

    return subject, obj, predicate


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

    opt = torch.optim.Adam([*tb.parameters(), a_bar], lr=5e-2)

    for step in range(400):
        opt.zero_grad()
        _, o_out, p_out = decode_semantic(
            tb, a_bar, subjects, entity_mask, predicate_mask, commit="expectation"
        )
        # No loss on subject (teacher-forced).
        loss = F.cross_entropy(o_out.logits, objects) + F.cross_entropy(p_out.logits, predicates)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                o_acc = (o_out.logits.argmax(-1) == objects).float().mean().item()
                p_acc = (p_out.logits.argmax(-1) == predicates).float().mean().item()
            print(
                f"step {step:4d} | loss {loss.item():.3f} | "
                f"acc o/p {o_acc:.2f}/{p_acc:.2f} | α {tb.alpha.item():+.3f}"
            )

    print("\nPer-subject distribution (top-3 predictions vs. ground-truth set):")
    unique_subjects = sorted(set(subjects.tolist()))
    with torch.no_grad():
        for s_global in unique_subjects:
            s_local = s_global - ent_off
            s_tensor = torch.tensor([s_global], device=device)
            _, o_out, p_out = decode_semantic(
                tb, a_bar, s_tensor, entity_mask, predicate_mask, commit="expectation"
            )
            p_logits = p_out.logits[0]
            o_logits = o_out.logits[0]
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
