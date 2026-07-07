"""Annotation-only KG-embedding prior for V2, on PyKEEN built-ins.

A KGE model consumes only the training *triples* — no visual features — so it
is V2's mandated **prior baseline** (fairness §2 of the plan): it measures how
much per-moment predicate prediction is solvable from relational co-occurrence
statistics alone. It sees strictly less evidence than the TB/flat conditions
(no perception), so it is reported as a prior, never as a like-for-like
condition. Using PyKEEN built-ins keeps it reproducible and the code minimal:
any `pykeen.models` name that takes `embedding_dim` works — DistMult is the
default; RESCAL and ComplEx are natural alternates.

Entities are video-scoped ("<video>:<obj_id>"), matching the V2 entity index
space. Training triples keep their per-frame multiplicity (no dedup): SLCWA
sampling then weights each relation by how long it holds — a frequency prior,
which is exactly what a prior baseline should encode.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def train_kge(triples: list[tuple[str, str, str]], *, model: str = "DistMult",
              dim: int = 64, epochs: int = 100, seed: int = 0,
              batch: int = 4096):
    """Train a PyKEEN model on labeled (head, relation, tail) triples.
    Returns (model, triples_factory)."""
    from pykeen.models import model_resolver
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
    kge = model_resolver.make(model, triples_factory=tf, embedding_dim=dim,
                              random_seed=seed)
    loop = SLCWATrainingLoop(model=kge, triples_factory=tf)
    loop.train(triples_factory=tf, num_epochs=epochs, batch_size=batch,
               use_tqdm=False)
    kge.eval()
    return kge, tf


@torch.no_grad()
def predicate_scores(kge, tf, pairs: list[tuple[str, str]],
                     *, batch: int = 4096) -> tuple[Tensor, Tensor]:
    """Relation scores for (head, tail) pairs: (scores [n, num_relations]
    aligned to tf.relation_to_id, known mask). Pairs with an entity unseen in
    training are known=False with -inf scores — skipped and counted by the
    caller, never guessed."""
    e2i = tf.entity_to_id
    known = torch.tensor([h in e2i and t in e2i for h, t in pairs])
    scores = torch.full((len(pairs), tf.num_relations), -torch.inf)
    idx = known.nonzero(as_tuple=True)[0]
    if len(idx):
        ht = torch.tensor([[e2i[pairs[i][0]], e2i[pairs[i][1]]]
                           for i in idx.tolist()])
        out = [kge.score_r(ht[s:s + batch]).float()
               for s in range(0, len(ht), batch)]
        scores[idx] = torch.cat(out)
    return scores, known
