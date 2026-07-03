"""Regenerate `pvsg_instances.json` from the dataset's `pvsg.json` — WITH relation spans.

The previously committed slice dropped each relation's temporal spans, reducing
relations to (subject_id, object_id, predicate). This script re-extracts the same
compact slice but keeps the spans, and prints the relation-*boundary* census that
decides whether PVSG supports the V7 relation-dynamics experiment
(see VIDEO_EXPERIMENTS.md, Action 0).

Output format (superset of the old one; `pvsg_data.py` reads rel[:3]):
    {
      "objects":   [...],                      # unchanged: category vocab
      "relations": [...],                      # unchanged: predicate vocab
      "videos": [
        {"video_id": ..., "split": ...,
         "num_frames": int | null,
         "objects":   [{"id": object_id, "category": category}, ...],
         "relations": [[s_id, o_id, predicate, [[start, end], ...]], ...]},
        ...
      ]
    }

Run on the machine that has the dataset:
    python -m experiments.pvsg_hierarchy.make_instances \
        --pvsg /path/to/pvsg.json --out experiments/pvsg_hierarchy/pvsg_instances.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _rel_fields(rel) -> tuple[int, int, str, list[list[int]]]:
    """Extract (s_id, o_id, predicate, spans) from one relation entry.

    Handles the two encodings seen in OpenPVSG releases:
    list-form  [s_id, o_id, predicate, spans]  and dict-form
    {"s_id":…, "o_id":…, "relation":…, <span key>: …}. Fails loudly otherwise.
    """
    if isinstance(rel, (list, tuple)):
        if len(rel) < 4:
            raise ValueError(
                f"relation entry has no span field: {rel!r} — the source file "
                "itself lacks spans; check you are reading the full pvsg.json"
            )
        s, o, p, spans = rel[0], rel[1], rel[2], rel[3]
    elif isinstance(rel, dict):
        s = rel.get("s_id", rel.get("subject_id"))
        o = rel.get("o_id", rel.get("object_id"))
        p = rel.get("relation", rel.get("predicate"))
        spans = next(
            (rel[k] for k in ("spans", "duration", "durations", "frames", "time") if k in rel),
            None,
        )
        if s is None or o is None or p is None or spans is None:
            raise ValueError(f"unrecognized relation dict keys: {sorted(rel)}")
    else:
        raise ValueError(f"unrecognized relation entry type: {type(rel)}: {rel!r}")

    # normalize spans to [[start, end], ...]
    if spans and isinstance(spans[0], (int, float)):
        spans = [list(spans)]
    return int(s), int(o), str(p), [[int(a), int(b)] for a, b in spans]


def _video_entries(raw: dict) -> list[dict]:
    for key in ("data", "videos"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    raise ValueError(f"can't find video list in pvsg.json; top-level keys: {sorted(raw)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvsg", required=True, help="path to the dataset's pvsg.json")
    ap.add_argument("--out", required=True, help="output pvsg_instances.json")
    args = ap.parse_args()

    raw = json.load(open(args.pvsg))
    videos_raw = _video_entries(raw)

    videos, n_rel, n_spans, boundaries = [], 0, 0, 0
    boundary_by_pred: Counter[str] = Counter()
    boundary_by_video: Counter[str] = Counter()

    for v in videos_raw:
        vid = v.get("video_id", v.get("vid"))
        meta = v.get("meta", {})
        num_frames = v.get("num_frames", meta.get("num_frames"))
        objects = [
            {"id": int(o.get("object_id", o.get("id")) if isinstance(o, dict) else o[0]),
             "category": str(o["category"] if isinstance(o, dict) else o[1])}
            for o in v.get("objects", [])
        ]
        relations = [list(_rel_fields(rel)) for rel in v.get("relations", [])]
        n_rel += len(relations)
        # interior boundaries: a span start/end that is not the video's own
        # start/end — i.e., the relation's state visibly changes inside the
        # video. Without num_frames, the max span end in the video is the horizon.
        all_spans = [sp for _, _, _, spans in relations for sp in spans]
        n_spans += len(all_spans)
        horizon = int(num_frames) if num_frames else (
            max((e for _, e in all_spans), default=0) + 1)
        for s, o, p, spans in relations:
            for a, b in spans:
                for is_boundary in (a > 0, b < horizon - 1):
                    if is_boundary:
                        boundaries += 1
                        boundary_by_pred[p] += 1
                        boundary_by_video[vid] += 1
        videos.append({
            "video_id": vid,
            "split": v.get("split", meta.get("split")),
            "num_frames": num_frames,
            "objects": objects,
            "relations": relations,
        })

    out = {
        "objects": raw.get("objects", raw.get("classes")),
        "relations": raw.get("relations", raw.get("predicates")),
        "videos": videos,
    }
    Path(args.out).write_text(json.dumps(out))

    n_videos_with_b = sum(1 for c in boundary_by_video.values() if c > 0)
    print(f"videos: {len(videos)}   relations: {n_rel}   spans: {n_spans}")
    print(f"interior boundaries (state changes): {boundaries}")
    print(f"videos with >=1 boundary: {n_videos_with_b}")
    print("top predicates by boundary count:")
    for p, c in boundary_by_pred.most_common(15):
        print(f"  {c:6d}  {p}")
    print("\nV7 decision rule (VIDEO_EXPERIMENTS.md): >=500 boundaries in the")
    print("cached-video subset -> PVSG suffices. Cross-reference with the cache")
    print("manifest to get the per-subset count.")


if __name__ == "__main__":
    main()
