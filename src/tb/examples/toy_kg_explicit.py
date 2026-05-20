"""Episodic recall (TKG mode) — explicit version.

Identical to `tb.examples.toy_kg`, but written with direct calls to the
three primitives (evolve / attend / measure) instead of using the
`decode_chain` composer. Every operation the model performs is visible
in the training loop below.

Compare side by side with `toy_kg.py` to see exactly what `decode_chain`
does — the explicit version is ~15 lines longer but has no indirection.

Run:
    uv run python -m tb.examples.toy_kg_explicit
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tb.evolve import QTBEvolve
from tb.examples._toy_data import describe, make_toy_kg
from tb.indices import IndexLayer
from tb.primitives import TB, Commit, MeasureOutput, learnable_alpha


def decode_episodic(
    tb: TB,
    instances: torch.Tensor,
    entity_mask: torch.Tensor,
    predicate_mask: torch.Tensor,
    *,
    commit: Commit,
) -> tuple[MeasureOutput, MeasureOutput, MeasureOutput]:
    """Decode three positions (S, O, P) starting from the episodic embedding a_t.

    No sensory input. No concept measurements. Pure memory recall.
    """
    # Seed q with the episodic embedding for each requested instance.
    q = tb.index_layer.embed(instances)
    state = tb.init_state(q.shape[0], q.device)

    # --- Position S: subject ---
    q, state = tb.evolve(q, state)
    # No sensory input → attend is a no-op; omit it entirely.
    subject = tb.measure(q, mask=entity_mask, commit=commit)
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
        s_out, o_out, p_out = decode_episodic(
            tb, instances, entity_mask, predicate_mask, commit="expectation"
        )
        loss = (
            F.cross_entropy(s_out.logits, subjects)
            + F.cross_entropy(o_out.logits, objects)
            + F.cross_entropy(p_out.logits, predicates)
        )
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                s_acc = (s_out.logits.argmax(-1) == subjects).float().mean().item()
                o_acc = (o_out.logits.argmax(-1) == objects).float().mean().item()
                p_acc = (p_out.logits.argmax(-1) == predicates).float().mean().item()
            print(
                f"step {step:4d} | loss {loss.item():.3f} | "
                f"acc s/o/p {s_acc:.2f}/{o_acc:.2f}/{p_acc:.2f} | "
                f"α {tb.alpha.item():+.3f}"
            )

    print("\nFinal recall (argmax):")
    with torch.no_grad():
        s_out, o_out, p_out = decode_episodic(
            tb, instances, entity_mask, predicate_mask, commit="argmax"
        )
        for i in range(triples.shape[0]):
            truth = (
                int(instances[i]),
                int(subjects[i]),
                int(predicates[i]),
                int(objects[i]),
            )
            pred = (int(instances[i]), int(s_out.k[i]), int(p_out.k[i]), int(o_out.k[i]))
            ok = "✓" if pred == truth else "✗"
            print(f"  {ok}  truth = {describe(truth, groups)}")


if __name__ == "__main__":
    main()
