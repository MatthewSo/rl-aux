"""Optional Weights & Biases mirror of the per-run metrics.

`metrics.jsonl` stays the source of truth. Everything downstream -- `aggregate`,
`curves`, `weights`, `sublabels` -- reads the local files and never wandb, so a
sweep whose tracking failed is still a complete, analysable experiment. This
module only mirrors.

That ordering drives the whole design. Tracking runs on a box billed by the hour
where the only unrecoverable loss is the training itself, so **no wandb failure
may ever propagate into the run**. Every call is wrapped; the first failure
disables the tracker for the rest of the run and training continues.

Three details are load-bearing for sweeps specifically:

- **In-thread start.** `start_method="thread"` keeps wandb inside the process
  instead of spawning a service. With several cells starting at once on a
  multi-GPU box, the service handshake is what times out.
- **Offline when there is no key.** Rather than blocking on a doomed handshake,
  a box without credentials records to disk. `wandb sync` uploads it later, so
  an unattended sweep is never silently untracked.
- **Deterministic run ids.** The id is derived from the cell id, so re-running a
  cell after a crash resumes that wandb run instead of creating a duplicate.
  A resumable sweep with random ids produces a project full of half-runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_DEFAULT = "wamal-lm"


def _has_credentials() -> bool:
    """Whether an online run could plausibly authenticate.

    Checked up front rather than by attempting a login, because the failure mode
    of a missing key is a hang until `init_timeout`, and five idle minutes per
    cell across a 400-cell sweep is real money.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        try:
            if path.exists() and "api.wandb.ai" in path.read_text():
                return True
        except OSError:
            continue
    return False


def run_id_for(cell_id: str) -> str:
    """Stable, wandb-safe id for a cell. Same cell -> same run, always."""
    return hashlib.sha1(cell_id.encode()).hexdigest()[:16]


def _settings_kwargs(version: str) -> dict[str, Any]:
    """Init settings, adjusted for what this wandb version still honours.

    `start_method="thread"` is the fix for the service handshake timing out when
    several cells start at once. wandb 0.18 reworked that machinery and made the
    setting non-functional, where passing it only produces a deprecation warning
    on every run -- noise multiplied by 400 cells.
    """
    kwargs: dict[str, Any] = {"init_timeout": 300}
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except (ValueError, TypeError):
        return kwargs
    if (major, minor) < (0, 18):
        kwargs["start_method"] = "thread"
    return kwargs


class NullTracker:
    """What every arm gets when tracking is off. Costs nothing."""

    enabled = False
    url = None

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        pass

    def summarise(self, summary: dict[str, Any]) -> None:
        pass

    def finish(self, exit_code: int = 0) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish(exit_code=0 if exc_type is None else 1)
        return False


class WandbTracker:
    def __init__(self, run, module, run_path: Path | None = None):
        self._run = run
        self._wandb = module
        self.enabled = True
        self.url = getattr(run, "url", None)

        # Write the id and URL next to the metrics so a local run directory can
        # be matched to its wandb run later. Without this the two views are only
        # joinable by guessing at names.
        if run_path is not None:
            try:
                (Path(run_path) / "wandb_run.json").write_text(json.dumps({
                    "id": getattr(run, "id", None),
                    "name": getattr(run, "name", None),
                    "url": self.url,
                    "project": getattr(run, "project", None),
                    "entity": getattr(run, "entity", None),
                    "mode": "offline" if self.url is None else "online",
                }, indent=2))
            except OSError:
                pass

    def _guard(self, action, *args, **kwargs) -> None:
        if not self.enabled:
            return
        try:
            action(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # One failure is enough. Retrying every epoch would turn a flaky
            # network into a per-epoch stall.
            self.enabled = False
            print(f"[tracking] disabled after error: {exc.__class__.__name__}: {exc}",
                  flush=True)

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        # Drop Nones so arms without a meta step don't create empty panels.
        clean = {k: v for k, v in record.items() if v is not None}
        self._guard(self._run.log, clean, step=step)

    def summarise(self, summary: dict[str, Any]) -> None:
        flat = {k: v for k, v in summary.items() if not isinstance(v, dict)}
        for key, value in summary.items():
            if isinstance(value, dict):
                flat.update({f"{key}.{k}": v for k, v in value.items()})
        self._guard(self._run.summary.update, flat)

    def finish(self, exit_code: int = 0) -> None:
        self._guard(self._run.finish, exit_code=exit_code)
        self.enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish(exit_code=0 if exc_type is None else 1)
        return False


def build_tracker(cfg, run_path: Path | None = None, logger=None):
    """-> a tracker for this run. Never raises; degrades to NullTracker."""
    def note(message: str) -> None:
        if logger is not None:
            logger.print(f"[tracking] {message}")
        else:
            print(f"[tracking] {message}", flush=True)

    if not getattr(cfg, "wandb", False):
        return NullTracker()

    try:
        import wandb
    except ImportError:
        note("wandb not installed; continuing with metrics.jsonl only "
             "(pip install wandb)")
        return NullTracker()

    mode = getattr(cfg, "wandb_mode", "auto")
    if mode == "auto":
        mode = "online" if _has_credentials() else "offline"
        if mode == "offline":
            note("no WANDB_API_KEY and no netrc entry; recording offline. "
                 "Upload later with: wandb sync <results>/wandb/offline-run-*")
    if mode == "disabled":
        return NullTracker()

    cell = cfg.cell_id()
    settings_kwargs = _settings_kwargs(getattr(wandb, "__version__", ""))
    init_kwargs = dict(
        project=getattr(cfg, "wandb_project", None) or PROJECT_DEFAULT,
        entity=getattr(cfg, "wandb_entity", None) or None,
        # Grouping is what makes a 400-cell sweep navigable: group by phase,
        # split by arm, so the comparison you actually want is one click.
        group=getattr(cfg, "wandb_group", "") or None,
        job_type=cfg.arm,
        name=cell,
        id=run_id_for(cell),
        resume="allow",
        tags=[cfg.dataset, cfg.model, cfg.arm, f"k{cfg.shots}", f"seed{cfg.seed}"],
        config=json.loads(cfg.to_json()),
        mode=mode,
        reinit=True,
        dir=str(run_path) if run_path else None,
    )

    try:
        try:
            settings = wandb.Settings(**settings_kwargs)
        except TypeError:
            # Removed outright rather than merely deprecated. The reworked
            # default makes it unnecessary rather than wrong.
            settings = wandb.Settings(init_timeout=300)
        run = wandb.init(settings=settings, **init_kwargs)
    except Exception as exc:  # noqa: BLE001
        note(f"init failed ({exc.__class__.__name__}: {exc}); "
             f"continuing with metrics.jsonl only")
        return NullTracker()

    if run is None:
        note("init returned no run; continuing with metrics.jsonl only")
        return NullTracker()

    tracker = WandbTracker(run, wandb, run_path)
    note(f"{mode} as {cell}" + (f" -> {tracker.url}" if tracker.url else ""))
    return tracker
