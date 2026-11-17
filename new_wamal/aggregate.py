"""Summarise finished cells.

    python -m new_wamal.aggregate --exp E0

Reports, per (setup, method): n seeds, test accuracy at the best-validation
epoch (mean +- std), final-epoch test accuracy, seconds per epoch, peak memory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--root", default="results_new")
    a = ap.parse_args()

    rows = defaultdict(list)
    running = defaultdict(int)
    for d in sorted(glob.glob(os.path.join(a.root, a.exp, "*", "seed*"))):
        name = os.path.basename(os.path.dirname(d))
        s = os.path.join(d, "summary.json")
        if os.path.exists(s):
            rows[name].append(json.load(open(s)))
        elif os.path.exists(os.path.join(d, "metrics.jsonl")):
            running[name] += 1

    fmt = "{:<58} {:>2}  {:>15}  {:>15}  {:>8}  {:>7}"
    print(fmt.format("cell", "n", "test@best-val", "test final", "s/epoch", "mem MB"))
    for name in sorted(set(rows) | set(running)):
        r = rows.get(name, [])
        if not r:
            print(fmt.format(name, 0, f"({running[name]} running)", "", "", "")); continue
        tb = np.array([x["test_at_best_val"] for x in r]) * 100
        tf = np.array([x["test_final"] for x in r]) * 100
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        print(fmt.format(name, len(r), f"{tb.mean():.2f} ± {sd(tb):.2f}", f"{tf.mean():.2f} ± {sd(tf):.2f}",
                         f"{np.mean([x['sec_per_epoch'] for x in r]):.1f}",
                         f"{max(x['peak_mem_mb'] for x in r):.0f}")
              + (f"  (+{running[name]} running)" if running.get(name) else ""))


if __name__ == "__main__":
    main()
