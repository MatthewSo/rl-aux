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
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from llm.config import ARMS, ExperimentConfig
from llm.paths import is_complete, run_dir


@dataclass(frozen=True)
class Phase:
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    arms: tuple[str, ...]
    shots: tuple[int, ...]
    seeds: tuple[int, ...]
    note: str = ""
    meta_holdout_frac: float = 0.0
    # Swept per cell, unlike meta_holdout_frac, because the ladder's whole point
    # is the shape of the curve across levels.
    corruptions: tuple[float, ...] = (0.0,)


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
    # Phase 1's deficit tracks how long the meta signal survives: the outer loss is
    # measured on the batch theta was just fitted on, so it reaches ~1e-5 halfway
    # through a k=5 run and phi stops learning, while at k=100 it stays alive for
    # 87% of the run -- which is the ordering of wamal's results. This measures the
    # outer loss on withheld data instead, at the two shot counts the go/no-go
    # rests on. All four arms, because the withheld examples leave theta's training
    # set for every arm and the baselines must stay matched.
    #
    # Read aux_recovery first, not accuracy: it is the direct test of whether phi
    # finds real sub-structure once its objective stops saturating, and it is not
    # confounded by the smaller training set.
    "phase1_holdout": Phase(
        datasets=("clinc150",),
        models=("opt-125m",),
        arms=ARMS,
        shots=(5, 10),
        seeds=tuple(range(5)),
        note="meta objective on withheld data; isolates the saturation confound",
        meta_holdout_frac=0.2,
    ),
    # The intent-level task. Phase 1's ceiling is the reason: an oracle hierarchy on
    # the 10-way domain task is worth only ~+1pp (human - single, paired), so no
    # label-generating method can clear it by much and a negative result there is
    # partly a statement about the task. All 150 intents as the primary task is far
    # harder and leaves real headroom, and the auxiliary hierarchy becomes invented
    # rather than recovered, which is MAXL's original setting.
    #
    # No `human` arm, because no ground truth exists below an intent -- the config
    # rejects it rather than letting it train on a constant target. The cost is that
    # sub-label recovery cannot be scored, so read inter-seed agreement instead:
    #   python -m llm.analysis.recovery --inter-seed --dataset clinc150_intent --psi 5
    "phase1_intent": Phase(
        datasets=("clinc150_intent",),
        models=("opt-125m",),
        arms=("single", "maxl", "wamal"),
        # k=100 is dropped, not forgotten: at 15000 examples it would cost more than
        # the rest of the phase combined, and the domain task already showed the
        # auxiliary effect shrinking monotonically with data. Headroom lives at low k.
        shots=(5, 10, 25),
        seeds=tuple(range(5)),
        note="150-way primary, 5 invented sub-classes; headroom the domain task lacks",
    ),
    # How good would learned labels have to be? Phase 1 says real hierarchical
    # labels are worth +0.8pp at k=5 while phi's are worth nothing, and a
    # permutation-invariant reading of the dumps puts phi at roughly 15-20% of the
    # true hierarchy -- not the "at chance" that the naive aux_recovery metric
    # implied. So the open question is whether phi's labels are simply below the
    # quality threshold where an auxiliary task starts paying.
    #
    # This corrupts the human labels toward chance-within-block and reads the gain
    # as a function of quality. Cheap: the human arm has no meta step, so these
    # cost ~200s each rather than wamal's ~500s. The 0.0 rung is deliberately
    # omitted -- phase 1's human cells already are it, and rerunning would only
    # risk disagreeing with them.
    "phase1_ladder": Phase(
        datasets=("clinc150",),
        models=("opt-125m",),
        arms=("human",),
        shots=(5, 10),
        seeds=tuple(range(5)),
        corruptions=(0.25, 0.5, 0.75, 1.0),
        note="label-quality ladder; how good must auxiliary labels be to help",
    ),
    # Phase 1 found wamal ~1pp below single at every k, on a task already at 90.6%
    # with k=5 and 97.7% at k=100 -- about 7pp of total range. This asks whether
    # the deficit is about headroom by going where accuracy is genuinely lower.
    # Same dataset, model, arms and seed count as phase 1, so the k grids join up
    # into one curve. A wamal deficit that persists at k=1 rules headroom out,
    # which is the more informative outcome of the two.
    "phase1_lowk": Phase(
        datasets=("clinc150",),
        models=("opt-125m",),
        arms=ARMS,
        shots=(1, 2),
        seeds=tuple(range(10)),
        note="headroom probe: extends phase 1's k grid down to 1 and 2 shots",
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
    for seed, dataset, model, shots, arm, corruption in itertools.product(
        phase.seeds, phase.datasets, phase.models, phase.shots, phase.arms,
        phase.corruptions,
    ):
        extra = dict(overrides)
        if corruption > 0:
            extra["label_corruption"] = corruption
        cfg = ExperimentConfig(
            dataset=dataset, model=model, arm=arm, shots=shots, seed=seed,
            **extra,
        ).resolve()
        cells.append(cfg)
    return cells


def measured_runtimes(results_root: Path) -> dict[str, float]:
    """Wall times from runs already on disk, keyed by (arm, shots).

    Used to price a phase from measured data rather than a guess. After phase 1
    this makes the phase 2 bill an extrapolation.
    """
    times: dict[str, list[float]] = {}
    for summary_path in results_root.rglob("summary.json"):
        try:
            record = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            continue
        wall = record.get("wall_time_s")
        if not wall:
            continue
        times.setdefault(f"{record.get('arm')}:{record.get('shots')}", []).append(wall)
    return {k: sum(v) / len(v) for k, v in times.items()}


def estimate_hours(cells: list, results_root: Path, workers: int) -> tuple[float, int]:
    """-> (estimated GPU-hours, number of cells we had a measurement for).

    Cells with no matching measurement are priced at the mean of whatever we do
    have; with no data at all the estimate is meaningless and the caller says so.
    """
    measured = measured_runtimes(results_root)
    if not measured:
        return 0.0, 0

    fallback = sum(measured.values()) / len(measured)
    total, known = 0.0, 0
    for cfg in cells:
        key = f"{cfg.arm}:{cfg.shots}"
        if key in measured:
            total += measured[key]
            known += 1
        else:
            total += fallback
    return total / 3600.0 / max(workers, 1), known


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="smoke", choices=sorted(PHASES))
    p.add_argument("--dry-run", action="store_true",
                   help="list the pending cells and stop; use this before spending")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU ids to run across, e.g. 0,1,2,3. "
                        "Overrides --gpu.")
    p.add_argument("--pack", type=int, default=1,
                   help="concurrent cells per GPU (default 1). These cells leave "
                        "the GPU mostly idle, so 2-4 is usually near-linear. "
                        "Watch VRAM and CPU: each cell is a full process.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="dataloader workers per cell. Lower it when packing, "
                        "since total processes are gpus x pack x (1 + workers).")
    p.add_argument("--meta-holdout-frac", type=float, default=None,
                   help="override the phase's meta holdout fraction. Writes to "
                        "separate run directories, so it cannot collide with or "
                        "be averaged into the base results.")
    p.add_argument("--wandb", action="store_true",
                   help="mirror every cell to Weights & Biases, grouped by phase")
    p.add_argument("--wandb-project", default="wamal-lm")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None,
                   help="defaults to the phase name, which is what you usually want")
    p.add_argument("--wandb-mode", default="auto",
                   choices=("auto", "online", "offline", "disabled"))
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


def cell_command(cfg, gpu: int, force: bool) -> list[str]:
    """Each cell runs in a fresh process.

    A CUDA OOM or a wedged allocator in one cell would otherwise poison every
    cell after it, and losing a 200-cell sweep to cell 12 is the expensive
    failure mode. Process isolation makes a bad cell cost one cell.
    """
    cmd = [
        sys.executable, "-m", "llm.scripts.run_experiment",
        "--dataset", cfg.dataset, "--model", cfg.model, "--arm", cfg.arm,
        "--shots", str(cfg.shots), "--seed", str(cfg.seed),
        "--optimizer", cfg.optimizer,
        "--primary-lr", str(cfg.primary_lr), "--gen-lr", str(cfg.gen_lr),
        "--batch-size", str(cfg.batch_size), "--dtype", cfg.dtype,
        "--gpu", str(gpu),
        "--num-workers", str(cfg.num_workers),
    ]
    # Only when non-default, so the base sweep's command lines stay identical to
    # the ones that produced the existing results.
    if cfg.meta_holdout_frac > 0:
        cmd += ["--meta-holdout-frac", str(cfg.meta_holdout_frac)]
    if cfg.label_corruption > 0:
        cmd += ["--label-corruption", str(cfg.label_corruption)]
    if cfg.results_dir:
        cmd += ["--results-dir", cfg.results_dir]
    if force:
        cmd += ["--force"]
    if getattr(cfg, "wandb", False):
        cmd += ["--wandb",
                "--wandb-project", cfg.wandb_project,
                "--wandb-mode", cfg.wandb_mode]
        if cfg.wandb_group:
            cmd += ["--wandb-group", cfg.wandb_group]
        if cfg.wandb_entity:
            cmd += ["--wandb-entity", cfg.wandb_entity]
    return cmd


def run_sequential(pending, gpu, force, keep_going, log_dir=None):
    failures, started = [], time.time()
    for i, cfg in enumerate(pending, 1):
        elapsed = time.time() - started
        eta = (elapsed / max(i - 1, 1)) * (len(pending) - i + 1) if i > 1 else 0
        print(f"\n[{i}/{len(pending)}] {cfg.cell_id()}  "
              f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)", flush=True)

        result = subprocess.run(cell_command(cfg, gpu, force))
        if result.returncode != 0:
            failures.append((cfg.cell_id(), f"exit {result.returncode}"))
            print(f"  FAILED: exit {result.returncode}", file=sys.stderr)
            if not keep_going:
                break
    return failures


def run_parallel(pending, slots, force, keep_going):
    """One worker thread per slot, each pulling from a shared queue.

    A slot is a place to run one cell; `slots` is a list of GPU ids with repeats,
    so [0,0,1,1] means two concurrent cells on each of GPUs 0 and 1. Packing pays
    off because these cells barely load the GPU: at OPT-125M with 64-token
    sequences the bottleneck is host-side kernel launch, not compute, so a single
    cell leaves the device idle most of the time. Concurrent cells interleave
    into those gaps.

    A queue rather than a static partition because cells vary a lot in cost --
    a k=100 wamal cell is an order of magnitude longer than k=5 single -- and a
    static split would leave slots idle at the end waiting on the unlucky one.

    Per-cell output goes to a log file instead of the terminal; interleaving
    several training runs on one stdout is unreadable.
    """
    work: queue.Queue = queue.Queue()
    for cfg in pending:
        work.put(cfg)

    failures: list[tuple[str, str]] = []
    lock = threading.Lock()
    stop = threading.Event()
    done = [0]
    started = time.time()

    def worker(gpu: int) -> None:
        while not stop.is_set():
            try:
                cfg = work.get_nowait()
            except queue.Empty:
                return

            log_path = run_dir(cfg) / "sweep.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as log:
                result = subprocess.run(cell_command(cfg, gpu, force),
                                        stdout=log, stderr=subprocess.STDOUT)

            with lock:
                done[0] += 1
                elapsed = time.time() - started
                rate = elapsed / max(done[0], 1)
                remaining = rate * (len(pending) - done[0])
                status = "ok" if result.returncode == 0 else f"FAIL {result.returncode}"
                print(f"[{done[0]}/{len(pending)}] gpu{gpu} {status} "
                      f"{cfg.cell_id()}  (elapsed {elapsed/60:.1f}m, "
                      f"eta {remaining/60:.1f}m)", flush=True)
                if result.returncode != 0:
                    failures.append((cfg.cell_id(), f"exit {result.returncode}, see {log_path}"))
                    if not keep_going:
                        stop.set()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def main(argv: list[str] | None = None) -> int:
    from llm import env

    args = build_parser().parse_args(argv)
    phase = PHASES[args.phase]

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip() != ""]
            if args.gpus else [args.gpu])
    if args.pack < 1:
        print("--pack must be at least 1", file=sys.stderr)
        return 2
    # Interleaved rather than grouped ([0,1,0,1] not [0,0,1,1]) so that a sweep
    # stopped early has spread its work evenly across devices.
    slots = [g for _ in range(args.pack) for g in gpus]

    overrides = dict(
        optimizer=args.optimizer,
        primary_lr=args.primary_lr,
        gen_lr=args.gen_lr,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=f"cuda:{gpus[0]}",
        results_dir=args.results_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        # Grouping by phase is what turns 400 runs into one comparable sweep.
        wandb_group=args.wandb_group if args.wandb_group is not None else args.phase,
        wandb_mode=args.wandb_mode,
    )
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.meta_holdout_frac is not None:
        overrides["meta_holdout_frac"] = args.meta_holdout_frac
    elif phase.meta_holdout_frac > 0:
        overrides["meta_holdout_frac"] = phase.meta_holdout_frac

    cells = enumerate_cells(phase, overrides)
    pending = [c for c in cells if args.force or not is_complete(c)]
    if args.limit:
        pending = pending[: args.limit]

    results_root = Path(args.results_dir) if args.results_dir else env.results_dir()

    workers_per_cell = overrides.get("num_workers", 2)
    print(f"phase {args.phase}: {phase.note}")
    print(f"{len(cells)} cells total, {len(cells) - len(pending)} already done, "
          f"{len(pending)} to run on GPU(s) {gpus}"
          + (f" x{args.pack} packed = {len(slots)} concurrent"
             if args.pack > 1 else ""))

    # Each cell is a process with its own dataloader workers. Oversubscribing
    # the container's pid or core budget is the failure the thread caps exist to
    # prevent, so say the number out loud before it bites.
    procs = len(slots) * (1 + workers_per_cell)
    cores = os.cpu_count() or 0
    if args.pack > 1:
        print(f"  {procs} processes ({len(slots)} cells x 1+{workers_per_cell} workers)"
              + (f", {cores} cores visible" if cores else ""))
        if cores and procs > cores:
            print(f"  WARNING: more processes than cores; consider "
                  f"--num-workers {max(0, cores // len(slots) - 1)} or a lower --pack")

    hours, known = estimate_hours(pending, results_root, len(slots))
    if known or hours:
        print(f"estimated {hours:.1f} h wall-clock across {len(slots)} slot(s) "
              f"({known}/{len(pending)} cells priced from measured runs)")
        if args.pack > 1:
            print("  assumes packing scales linearly; it will not exactly, since "
                  "packed cells contend for the host")
    else:
        print("no prior runs to estimate from; run --phase smoke first for a price")

    if args.dry_run:
        for cfg in pending:
            print("  " + cfg.cell_id())
        return 0

    if not pending:
        print("nothing to do")
        return 0

    started = time.time()
    if len(slots) > 1:
        failures = run_parallel(pending, slots, args.force, args.keep_going)
    else:
        failures = run_sequential(pending, slots[0], args.force, args.keep_going)

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
