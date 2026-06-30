"""Derive and inspect the PVSG object taxonomy (layer A) from WordNet.

Defaults to the real 126 PVSG classes (115 thing + 11 stuff) extracted from the
dataset's `pvsg.json` into `pvsg_categories.json`. Prints a coverage summary, the
coarse (supersense) grouping, the classes WordNet can't map (dataset typos /
catch-alls — surfaced, not guessed), and the full leaf→coarse table for eyeball
review.

Run:
    uv run python -m experiments.pvsg_hierarchy.derive            # PVSG 126
    uv run python -m experiments.pvsg_hierarchy.derive --vrd      # VRD 100 (comparison)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.pvsg_hierarchy.taxonomy import build_taxonomy

HERE = Path(__file__).parent


def load_classes(vrd: bool) -> list[str]:
    if vrd:
        return json.load(open(HERE / "vrd_objects.json"))
    cats = json.load(open(HERE / "pvsg_categories.json"))
    return cats["thing"] + cats["stuff"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrd", action="store_true", help="use VRD-100 instead of PVSG-126")
    args = ap.parse_args()

    names = load_classes(args.vrd)
    tax = build_taxonomy(names)
    cov = tax.coverage()

    tag = "VRD-100" if args.vrd else "PVSG-126"
    print(f"=== {tag} object taxonomy (WordNet, automatic) ===")
    print(f"classes: {cov['classes']} | mapped: {cov['mapped']:.0%} | "
          f"levels  leaf:{cov['n_leaf']}  mid:{cov['n_mid']}  coarse:{cov['n_coarse']}")
    if tax.unmapped:
        print(f"\nUNMAPPED ({len(tax.unmapped)}) — dataset typos / catch-alls, surfaced not guessed:")
        print(f"  {tax.unmapped}")

    print("\n--- coarse grouping (supersense) ---")
    by_coarse: dict[str, list[str]] = defaultdict(list)
    for e in tax.entries:
        if e.coarse:
            by_coarse[e.coarse].append(e.name)
    for coarse in sorted(by_coarse, key=lambda c: -len(by_coarse[c])):
        print(f"  {coarse:12s} ({len(by_coarse[coarse]):2d})")

    # show how the mid level refines the largest coarse bucket
    biggest = max(by_coarse, key=lambda c: len(by_coarse[c]))
    print(f"\n--- mid level refines the largest coarse bucket ('{biggest}', "
          f"{len(by_coarse[biggest])}) ---")
    by_mid: dict[str, list[str]] = defaultdict(list)
    for e in tax.entries:
        if e.coarse == biggest and e.mid:
            by_mid[e.mid].append(e.name)
    for mid in sorted(by_mid, key=lambda m: -len(by_mid[m])):
        if len(by_mid[mid]) > 1:  # show the groups, skip singletons for brevity
            print(f"  {mid:18s} ({len(by_mid[mid]):2d}): {', '.join(by_mid[mid])}")
    singles = [m for m in by_mid if len(by_mid[m]) == 1]
    print(f"  + {len(singles)} singleton mid-groups")


if __name__ == "__main__":
    main()
