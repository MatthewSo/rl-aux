"""Resumable run matrix.

Rented instances get interrupted and every restart is billed, so completion is a
filesystem marker rather than in-memory state: rerunning the identical command
after an interruption picks up exactly where it stopped.

    python -m llm.scripts.sweep --phase phase1 --dry-run
    python -m llm.scripts.sweep --phase phase1
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass

from llm.config import ARMS, ExperimentConfig
from llm.paths import is_complete


@dataclass(frozen=True)
class Phase:
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    arms: tuple[str, ...]
    shots: tuple[int, ...]
    seeds: tuple[int, ...]
    note: str = ""


PHASES: dict[str, Phase] = {
    # LR calibration. Arms 1 and 4 only: if the extremes don't separate, the
    # middle arms tell us nothing, and this runs before we commit to a setting.
    "phase0": Phase(
        datasets=("clinc150",),
        models=("opt-125m",),
        arms=("single", "wamal"),
        shots=(100,),
        seeds=(0, 1),
        note="LR calibration; drive with --primary-lr overrides",
    ),
    "phase1": Phase(
        datasets=("clinc150",),
        models=("opt-125m",),
        arms=ARMS,
        shots=(5, 10, 25, 50, 100),
        seeds=tuple(range(10)),
        note="signal check; go/no-go is wamal > human at k=5 and k=10",
    ),
    "phase2": Phase(
        datasets=("clinc150", "newsgroups"),
        models=("granite-350m", "olmo2-1b"),
        arms=ARMS,
        shots=(5, 10, 25, 50, 100),
        seeds=tuple(range(5)),
        note="main result",
    ),
    "phase3": Phase(
        datasets=("clinc150",),
        models=("qwen2.5-0.5b", "opt-1.3b", "granite-h-350m"),
        arms=("single", "maxl", "wamal"),
        shots=(5, 25, 100),
        seeds=(0, 1, 2),
        note="robustness: cross-family, scale, architecture",
    ),
    "smoke": Phase(
        datasets=("clinc150",),
        models=("opt-125m",),
        arms=ARMS,
        shots=(5,),
        seeds=(0,),
        note="one cell per arm, minimal cost",
    ),
}


def enumerate_cells(phase: Phase, overrides: dict) -> list[ExperimentConfig]:
    cells = []
    # Seed is the outer loop so that an interrupted sweep still yields complete
    # low-seed groups: 3 seeds of everything supports an error bar, 10 seeds of
    # one arm supports nothing.
    for seed, dataset, model, shots, arm in itertools.product(
        phase.seeds, phase.datasets, phase.models, phase.shots, phase.arms
    ):
        cfg = ExperimentConfig(
            dataset=dataset, model=model, arm=arm, shots=shots, seed=seed,
            **overrides,
        ).resolve()
        cells.append(cfg)
    return cells


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="smoke", choices=sorted(PHASES))
    p.add_argument("--dry-run", action="store_true",
                   help="list the pending cells and stop; use this before spending")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw"))
    p.add_argument("--primary-lr", type=float, default=1e-2)
    p.add_argument("--gen-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N cells; useful for a timeboxed rental")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-going", action="store_true",
                   help="log failures and continue instead of aborting the sweep")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phase = PHASES[args.phase]

    overrides = dict(
        optimizer=args.optimizer,
        primary_lr=args.primary_lr,
        gen_lr=args.gen_lr,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=f"cuda:{args.gpu}",
        results_dir=args.results_dir,
    )

    cells = enumerate_cells(phase, overrides)
    pending = [c for c in cells if args.force or not is_complete(c)]
    if args.limit:
        pending = pending[: args.limit]

    print(f"phase {args.phase}: {phase.note}")
    print(f"{len(cells)} cells total, {len(cells) - len(pending)} already done, "
          f"{len(pending)} to run")

    if args.dry_run:
        for cfg in pending:
            print("  " + cfg.cell_id())
        return 0

    failures: list[tuple[str, str]] = []
    started = time.time()

    for i, cfg in enumerate(pending, 1):
        elapsed = time.time() - started
        eta = (elapsed / max(i - 1, 1)) * (len(pending) - i + 1) if i > 1 else 0
        print(f"\n[{i}/{len(pending)}] {cfg.cell_id()}  "
              f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)", flush=True)

        # Each cell runs in a fresh process. A CUDA OOM or a wedged allocator in
        # one cell would otherwise poison every cell after it, and losing a
        # whole sweep to cell 12 of 200 is the expensive failure mode here.
        cmd = [
            sys.executable, "-m", "llm.scripts.run_experiment",
            "--dataset", cfg.dataset, "--model", cfg.model, "--arm", cfg.arm,
            "--shots", str(cfg.shots), "--seed", str(cfg.seed),
            "--optimizer", cfg.optimizer,
            "--primary-lr", str(cfg.primary_lr), "--gen-lr", str(cfg.gen_lr),
            "--batch-size", str(cfg.batch_size), "--dtype", cfg.dtype,
            "--gpu", str(args.gpu),
        ]
        if cfg.results_dir:
            cmd += ["--results-dir", cfg.results_dir]
        if args.force:
            cmd += ["--force"]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((cfg.cell_id(), f"exit {result.returncode}"))
            print(f"  FAILED: exit {result.returncode}", file=sys.stderr)
            if not args.keep_going:
                break

    print(f"\nsweep finished in {(time.time() - started)/60:.1f}m")
    if failures:
        print(f"{len(failures)} failure(s):", file=sys.stderr)
        for cell, why in failures:
            print(f"  {cell}: {why}", file=sys.stderr)
        print(json.dumps({"failures": failures}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
