"""Permutation-invariant sub-label recovery, from the dumped assignments.

`summary.json`'s `final_test_aux_vs_fine_acc` asks whether the auxiliary head's
argmax equals the true fine label. For the `human` arm that is exactly right --
its targets *are* the true fine labels, so the integer slots are anchored. For
`maxl` and `wamal` it is close to meaningless: phi's clusters are identified only
up to a permutation within each coarse block, so a phi that partitioned every
domain's intents perfectly would still score 1/psi, indistinguishable from noise.

That is not a hypothetical. Phase 1 read ~0.060 against a 1/15 = 0.067 chance
level at every k, which looked like "phi learned nothing". Measured this way, phi
recovers a weak but unambiguous partition -- about 6.6x the null mutual
information at k=100.

Two statistics, because they fail differently:

- **matched accuracy**, the best permutation of clusters onto intents. Interpretable
  (a fraction of examples) but it penalises a cluster that merges two intents as
  hard as one that is random, and it is biased upward when blocks are small.
- **mutual information**, which credits partial and merged structure. Also biased
  upward at small n.

Both biases are handled the same way: recompute on assignments shuffled within
each block, which preserves the marginals and destroys only the correspondence.
The reported `lift` and `mi_ratio` are against that null, so a value of 0 (or 1x)
means no recoverable structure regardless of block size or shot count.

    python -m llm.analysis.recovery                     # every arm with dumps
    python -m llm.analysis.recovery --arm wamal --shots 100
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_SHUFFLES = 20


def _match(table: np.ndarray) -> int:
    """Best one-to-one cluster/intent assignment; greedy if scipy is absent."""
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-table)
        return int(table[rows, cols].sum())
    except ImportError:
        pass

    total = 0
    used_r: set[int] = set()
    used_c: set[int] = set()
    work = table.copy()
    for _ in range(work.shape[0]):
        best, best_val = None, 0
        for i in range(work.shape[0]):
            if i in used_r:
                continue
            for j in range(work.shape[1]):
                if j in used_c:
                    continue
                if work[i, j] > best_val:
                    best, best_val = (i, j), work[i, j]
        if best is None:
            break
        total += best_val
        used_r.add(best[0])
        used_c.add(best[1])
    return total


def matched_accuracy(pred: np.ndarray, true: np.ndarray, psi: int) -> float:
    table = np.zeros((psi, psi), dtype=np.int64)
    for p, t in zip(pred, true):
        table[p, t] += 1
    return _match(table) / max(len(pred), 1)


def mutual_information(pred: np.ndarray, true: np.ndarray) -> float:
    """In nats. Zero only when the assignment and the intent are independent."""
    n = len(pred)
    if n == 0:
        return 0.0
    joint = Counter(zip(pred.tolist(), true.tolist()))
    pm, tm = Counter(pred.tolist()), Counter(true.tolist())
    total = 0.0
    for (p, t), count in joint.items():
        pxy = count / n
        total += pxy * np.log(pxy / ((pm[p] / n) * (tm[t] / n)))
    return float(total)


def analyse_dump(path: Path, psi: int, shuffles: int = DEFAULT_SHUFFLES,
                 seed: int = 0) -> dict:
    data = np.load(path)
    sub, fine, coarse = data["sublabels"], data["fine"], data["coarse"]
    # -1 marks examples the dump never covered.
    keep = sub >= 0
    sub, fine, coarse = sub[keep], fine[keep], coarse[keep]
    if len(sub) == 0:
        raise ValueError(f"{path} has no assigned sub-labels")

    rng = np.random.default_rng(seed)
    acc, mi, acc_null, mi_null, used = [], [], [], [], []

    for block in np.unique(coarse):
        mask = coarse == block
        # Both are global indices within block `block`; shift to [0, psi).
        pred = sub[mask] - block * psi
        true = fine[mask] - block * psi
        if pred.min() < 0 or pred.max() >= psi or true.min() < 0 or true.max() >= psi:
            raise ValueError(
                f"{path}: block {block} has labels outside its own window, so "
                "fine // psi == coarse does not hold and the mask is misaligned"
            )

        acc.append(matched_accuracy(pred, true, psi))
        mi.append(mutual_information(pred, true))

        a, m = [], []
        for _ in range(shuffles):
            shuffled = rng.permutation(pred)
            a.append(matched_accuracy(shuffled, true, psi))
            m.append(mutual_information(shuffled, true))
        acc_null.append(float(np.mean(a)))
        mi_null.append(float(np.mean(m)))
        used.append(len(set(pred.tolist())))

    mi_mean, mi_null_mean = float(np.mean(mi)), float(np.mean(mi_null))
    return {
        "naive": float((sub == fine).mean()),
        "matched": float(np.mean(acc)),
        "matched_null": float(np.mean(acc_null)),
        "lift": float(np.mean(acc) - np.mean(acc_null)),
        "mi": mi_mean,
        "mi_null": mi_null_mean,
        "mi_ratio": mi_mean / mi_null_mean if mi_null_mean > 1e-12 else float("inf"),
        "clusters_used": float(np.mean(used)),
        "psi": psi,
        "n": int(len(sub)),
    }


def inter_seed_agreement(paths: list[Path], psi: int,
                         shuffles: int = DEFAULT_SHUFFLES,
                         seed: int = 0) -> dict:
    """Do independent seeds invent the same partition? -> agreement vs a null.

    The only recovery evidence available when the auxiliary hierarchy has no ground
    truth, as on the intent-level task. Two runs that split the same test examples
    the same way found structure in the data; two that disagree fit noise. Compared
    over the test split because the k-shot training subset differs by seed.

    Reported as mutual information between the two assignments within each primary
    class, against a null that shuffles one of them -- so, as with the ground-truth
    version, the number is chance-corrected and comparable across shot counts.
    """
    loaded = []
    for path in paths:
        data = np.load(path)
        loaded.append((data["sublabels"], data["primary"]))

    rng = np.random.default_rng(seed)
    mis, nulls, pairs = [], [], 0

    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            (sub_a, prim_a), (sub_b, prim_b) = loaded[i], loaded[j]
            if len(sub_a) != len(sub_b) or not np.array_equal(prim_a, prim_b):
                raise ValueError(
                    "assignments describe different test sets; they can only be "
                    "compared when both runs saw identical examples in order"
                )
            keep = (sub_a >= 0) & (sub_b >= 0)
            pairs += 1
            for block in np.unique(prim_a[keep]):
                mask = keep & (prim_a == block)
                if mask.sum() < 2:
                    continue
                a = sub_a[mask] - block * psi
                b = sub_b[mask] - block * psi
                mis.append(mutual_information(a, b))
                nulls.append(float(np.mean([
                    mutual_information(rng.permutation(a), b)
                    for _ in range(shuffles)
                ])))

    mi_mean = float(np.mean(mis)) if mis else 0.0
    null_mean = float(np.mean(nulls)) if nulls else 0.0
    return {
        "mi": mi_mean,
        "mi_null": null_mean,
        "mi_ratio": mi_mean / null_mean if null_mean > 1e-12 else float("inf"),
        "pairs": pairs,
        "runs": len(loaded),
    }


def find_assignments(results_dir: Path, arm: str | None = None,
                     shots: int | None = None,
                     dataset: str | None = None) -> list[Path]:
    out = []
    for path in sorted(results_dir.glob("*/*/*/k*/seed*/test_assignment.npz")):
        cell = path.parent
        if dataset and cell.parts[-5] != dataset:
            continue
        if arm and cell.parent.parent.name.split("__")[0] != arm:
            continue
        if shots is not None and cell.parent.name != f"k{shots}":
            continue
        out.append(path)
    return out


def find_dumps(results_dir: Path, arm: str | None = None,
               shots: int | None = None) -> list[Path]:
    """Latest dump per cell: the assignment at the end of training."""
    out = []
    for cell in sorted(results_dir.glob("*/*/*/k*/seed*")):
        if arm and cell.parent.parent.name.split("__")[0] != arm:
            continue
        if shots is not None and cell.parent.name != f"k{shots}":
            continue
        dumps = sorted(cell.glob("dump_epoch*.npz"))
        if dumps:
            out.append(dumps[-1])
    return out


def main(argv: list[str] | None = None) -> int:
    from llm import env

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--arm", default=None)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--psi", type=int, default=15)
    p.add_argument("--shuffles", type=int, default=DEFAULT_SHUFFLES)
    p.add_argument("--dataset", default=None)
    p.add_argument("--inter-seed", action="store_true",
                   help="compare phi's test-split partition across seeds instead of "
                        "against ground truth. The only recovery evidence available "
                        "where the auxiliary hierarchy is invented.")
    args = p.parse_args(argv)

    root = Path(args.results_dir or env.results_dir())

    if args.inter_seed:
        groups: dict[tuple, list[Path]] = {}
        for path in find_assignments(root, args.arm, args.shots, args.dataset):
            cell = path.parent
            groups.setdefault(cell.parts[-5:-1], []).append(path)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if not groups:
            print(f"no cell under {root} has 2+ runs with test_assignment.npz")
            return 1

        print(f"psi={args.psi}. Agreement between independent seeds on the same "
              f"test examples.\n")
        header = f"{'cell':<44} {'seeds':>6} {'MI':>8} {'null':>8} {'MI/null':>9}"
        print(header)
        print("-" * len(header))
        for key, paths in sorted(groups.items()):
            res = inter_seed_agreement(paths, args.psi, args.shuffles)
            ratio = ("inf" if res["mi_ratio"] == float("inf")
                     else f"{res['mi_ratio']:.2f}x")
            print(f"{'/'.join(key):<44} {res['runs']:>6} {res['mi']:>8.4f} "
                  f"{res['mi_null']:>8.4f} {ratio:>9}")
        print("\nMI/null near 1x means seeds invent unrelated partitions, i.e. phi is")
        print("fitting noise. Clearly above 1x means independent runs found the same")
        print("structure in the data, which is real even with no label for it.")
        return 0
    dumps = find_dumps(root, args.arm, args.shots)
    if not dumps:
        print(f"no dumps under {root}"
              f"{f' for arm {args.arm}' if args.arm else ''}")
        return 1

    print(f"psi={args.psi}, naive chance = 1/{args.psi} = {1/args.psi:.4f}\n")
    header = (f"{'cell':<42} {'naive':>7} {'matched':>8} {'null':>7} {'lift':>7} "
              f"{'MI':>7} {'MI/null':>8} {'used':>5}")
    print(header)
    print("-" * len(header))

    rows = []
    for dump in dumps:
        cell = dump.parent
        label = "/".join(cell.parts[-4:])
        try:
            res = analyse_dump(dump, args.psi, args.shuffles)
        except ValueError as exc:
            print(f"{label:<42} SKIPPED: {exc}")
            continue
        rows.append((label, res))
        ratio = "inf" if res["mi_ratio"] == float("inf") else f"{res['mi_ratio']:.2f}x"
        print(f"{label:<42} {res['naive']:>7.4f} {res['matched']:>8.4f} "
              f"{res['matched_null']:>7.4f} {res['lift']:>+7.4f} {res['mi']:>7.4f} "
              f"{ratio:>8} {res['clusters_used']:>5.1f}")

    if rows:
        print(f"\n{len(rows)} cell(s). `lift` and `MI/null` are the chance-corrected")
        print("numbers; `naive` is what summary.json reports and is only meaningful")
        print("for the human arm, whose targets are the true fine labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
