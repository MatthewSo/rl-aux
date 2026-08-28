"""Run directory layout and the epoch-metrics writer.

Deliberately not `utils/path_name.create_path_name`: that takes 16 required
positional parameters with no defaults (46 existing call sites already crash on
it) and encodes vision/RL fields that mean nothing here. One directory per cell,
greppable, so aggregation can walk the tree instead of parsing filenames.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DONE_MARKER = "DONE"


def run_dir(cfg) -> Path:
    return Path(cfg.results_dir) / cfg.cell_id()


def is_complete(cfg) -> bool:
    """Presence of the marker is what lets an interrupted spot instance resume."""
    return (run_dir(cfg) / DONE_MARKER).exists()


def prepare_run_dir(cfg, overwrite: bool = False) -> Path:
    path = run_dir(cfg)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / DONE_MARKER
    if overwrite and marker.exists():
        marker.unlink()
    (path / "config.json").write_text(cfg.to_json())
    return path


def mark_complete(path: Path, summary: dict[str, Any]) -> None:
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    # Written last, and only after summary.json lands, so a marker always
    # implies a readable result.
    (path / DONE_MARKER).write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")


class RunLogger:
    """Appends to `print.log` and `metrics.jsonl` inside one run directory.

    Line-buffered and flushed per record: a preempted instance keeps whatever it
    had written, which is the difference between losing a run and losing an epoch.
    """

    def __init__(self, path: Path, echo: bool = True):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._log = open(self.path / "print.log", "a", buffering=1)
        self._metrics = open(self.path / "metrics.jsonl", "a", buffering=1)

    def print(self, *parts: object) -> None:
        line = " ".join(str(p) for p in parts)
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        if self.echo:
            print(stamped, flush=True)
        self._log.write(stamped + "\n")

    def metric(self, record: dict[str, Any]) -> None:
        self._metrics.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        for handle in (self._log, self._metrics):
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
