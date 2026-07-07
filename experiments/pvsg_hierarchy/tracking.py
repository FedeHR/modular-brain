"""Optional trackio logging shared by the runners.

Results JSONs (`results_io`) remain the source of truth; trackio is the
live/comparison view (`trackio show`, or `TRACKIO_DIR=... trackio show` for
DBs fetched from the cluster). Tracking lives in the runners' `main()` only —
`run()` takes a plain `log_step` callback, so tests never touch trackio.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def track(project: str, name: str, config: dict, *, enabled: bool = True):
    """Yields `log(metrics: dict, step: int | None = None)`; a no-op when
    disabled, so callers need no conditionals."""
    if not enabled:
        yield lambda metrics, step=None: None
        return
    import trackio

    trackio.init(project=project, name=name, config=config)
    try:
        yield lambda metrics, step=None: trackio.log(metrics, step=step)
    finally:
        trackio.finish()
