"""Plain-JSON result storage shared by all experiment harnesses.

Layout: one folder per invocation under `results/<experiment>/`, one JSON per
seed inside it:

    results/v1/20260705-101500_blocked/seed0.json
                                       seed1.json ...

Each JSON is self-contained — config, parameter counts, loss curves, every
metric with its counts — so tables and plots can be regenerated or restyled
from disk without rerunning anything (see e.g. `report_v1`). `results/` is
git-ignored; the JSONs are the plotting source of truth, the scripts that
render them live in the repo.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def new_run_dir(root: str | Path, *, tag: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = Path(root) / (f"{stamp}_{tag}" if tag else stamp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True))


def load_runs(run_dir: str | Path) -> list[dict]:
    """All seed*.json of one invocation, sorted by filename."""
    return [json.loads(p.read_text())
            for p in sorted(Path(run_dir).glob("seed*.json"))]
