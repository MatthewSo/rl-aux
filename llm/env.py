"""Filesystem and cache locations.

On vast.ai the container filesystem is ephemeral but `/workspace` is usually a
mounted volume that survives instance teardown and can be reattached. Anything
expensive to reproduce (multi-GB model weights) or impossible to reproduce (run
results) therefore defaults to living under `/workspace`.

Both locations are env-overridable so the same code runs unchanged on a laptop.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path("/workspace")


def _persistent_root() -> Path:
    """`/workspace` when it looks like a real mounted volume, else the home dir."""
    if WORKSPACE.is_dir() and os.access(WORKSPACE, os.W_OK):
        return WORKSPACE
    return Path.home()


def hf_home() -> Path:
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return Path(explicit).expanduser()
    return _persistent_root() / ".cache" / "huggingface"


def results_dir() -> Path:
    explicit = os.environ.get("RESULTS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    root = _persistent_root()
    # A laptop shouldn't get results dumped in ~; keep those next to the repo.
    if root == WORKSPACE:
        return root / "results"
    return Path.cwd() / "results"


def data_dir() -> Path:
    explicit = os.environ.get("WAMAL_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    return hf_home() / "wamal_data"


# Rented boxes are shared containers with a capped pid budget. Left alone,
# OpenBLAS and OpenMP each spawn one thread per *host* core -- which on a
# 128-core machine where we were sold 8 is dozens of threads per process, times
# every dataloader worker. The container hits its pid ceiling and the symptom is
# a BrokenPipeError when a worker forks, which looks like a data bug and is not.
# Capping costs nothing here: the work is GPU-bound and our tensors are small.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
DEFAULT_THREADS = "4"


def configure(threads: str | None = None) -> dict[str, str]:
    """Export cache locations and thread caps, and create the directories.

    Must run before torch/numpy/transformers are imported anywhere that matters:
    the BLAS libraries read their thread counts once, at load time, and ignore
    later changes.
    """
    home = hf_home()
    for sub in ("hub", "datasets"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    results_dir().mkdir(parents=True, exist_ok=True)

    exported = {
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(home / "hub"),
        "HF_DATASETS_CACHE": str(home / "datasets"),
        # Dataloader workers fork after the tokenizer is built; the parallelism
        # warning is noise and the fast path is unaffected at our sequence
        # lengths.
        "TOKENIZERS_PARALLELISM": "false",
    }
    cap = threads or os.environ.get("WAMAL_NUM_THREADS", DEFAULT_THREADS)
    for var in THREAD_VARS:
        exported[var] = cap

    for key, value in exported.items():
        os.environ.setdefault(key, value)
    return exported


def describe() -> str:
    return (
        f"HF_HOME={hf_home()}\n"
        f"RESULTS_DIR={results_dir()}\n"
        f"DATA_DIR={data_dir()}\n"
        f"threads={os.environ.get('OMP_NUM_THREADS', '<unset>')}"
    )
