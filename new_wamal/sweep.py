"""Run an experiment grid on a shared machine without hogging it.

    python -m new_wamal.sweep --exp E0 --gpus 0,1,3
    python -m new_wamal.sweep --exp E0 --dry-run

Guard rails (this runs on a shared lab server):
* at most MAX_GPUS (3) GPUs, one job per GPU;
* a GPU is only used if nobody else has more than OTHER_USER_MB on it;
* jobs run under `nice -n 10`, with 2 CPU threads and 2 dataloader workers;
* nothing new is launched while free disk is below MIN_FREE_GB;
* finished cells (DONE marker) are skipped, so the sweep is restartable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

from new_wamal.experiments import EXPERIMENTS, cell_dir

MAX_GPUS = 3
OTHER_USER_MB = 1500
MIN_FREE_GB = 8


def gpu_mem_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    used = {}
    for line in out.strip().splitlines():
        try:
            i, mb = [s.strip() for s in line.split(",")]
            used[int(i)] = int(mb)
        except ValueError:
            continue
    return used


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def to_args(c, out):
    args = []
    for k, v in c.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args + ["--out", out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gpus", default="0,1,3")
    ap.add_argument("--root", default="results_new")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--only", default="", help="substring filter on cell dir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gpus = [int(g) for g in a.gpus.split(",") if g != ""]
    if len(gpus) > MAX_GPUS:
        sys.exit(f"refusing to use {len(gpus)} GPUs; the limit on this machine is {MAX_GPUS}")

    cells = EXPERIMENTS[a.exp]()
    todo = []
    for c in cells:
        d = cell_dir(a.root, a.exp, c)
        if a.only and a.only not in d:
            continue
        if not os.path.exists(os.path.join(d, "DONE")):
            todo.append((d, c))
    print(f"{a.exp}: {len(cells)} cells, {len(todo)} to run, GPUs {gpus}")
    if a.dry_run:
        for d, _ in todo:
            print("  ", d)
        return

    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    running = {}   # gpu -> (Popen, dir)
    failed = []
    while todo or running:
        for g, (p, d) in list(running.items()):
            if p.poll() is not None:
                if p.returncode != 0:
                    failed.append(d); print(f"FAILED ({p.returncode}): {d}", flush=True)
                else:
                    print(f"done: {d}", flush=True)
                del running[g]
        used = gpu_mem_used()
        for g in gpus:
            if not todo or g in running:
                continue
            if used.get(g, 0) > OTHER_USER_MB:
                continue                       # someone else is on it
            if free_gb(".") < MIN_FREE_GB:
                print(f"low disk ({free_gb('.'):.1f} GB free); not launching", flush=True)
                break
            d, c = todo.pop(0)
            os.makedirs(d, exist_ok=True)
            cmd = ["nice", "-n", "10", sys.executable, "-m", "new_wamal.train", *to_args(c, d),
                   "--data-root", a.data_root, "--device", "cuda:0"]
            log = open(os.path.join(d, "stdout.log"), "a")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(g)))
            running[g] = (p, d)
            print(f"GPU {g}: {d}", flush=True)
        time.sleep(30)
    print(f"finished; {len(failed)} failed")
    for d in failed:
        print("  FAILED", d)


if __name__ == "__main__":
    main()
