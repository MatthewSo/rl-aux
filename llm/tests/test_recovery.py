"""Calibration for the permutation-invariant recovery statistics.

The point of `llm/analysis/recovery.py` is to distinguish two worlds the naive
`aux_vs_fine_acc` metric cannot: phi assigning sub-labels at random, versus phi
recovering the intents exactly under a relabeling. Both score ~1/psi naively.

These build dumps with known structure at CLINC150's geometry and check the
statistics respond monotonically, and -- more importantly -- that they read *zero*
on random input at every block size. A statistic that drifted with sample size
would be worse than useless here, since it is compared across shot counts.

    python -m llm.tests.test_recovery
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np

from llm.analysis.recovery import analyse_dump, matched_accuracy, mutual_information

PSI, COARSE = 15, 10


def write_dump(directory: Path, assign, k: int, seed: int = 0) -> Path:
    """A dump with `k` examples per fine class, labelled by `assign`."""
    rng = np.random.default_rng(seed)
    sub, fine, coarse = [], [], []
    for block in range(COARSE):
        true_local = np.repeat(np.arange(PSI), k)
        pred_local = assign(true_local, rng)
        sub.extend((pred_local + block * PSI).tolist())
        fine.extend((true_local + block * PSI).tolist())
        coarse.extend([block] * len(true_local))

    path = directory / "dump_epoch000.npz"
    np.savez_compressed(
        path,
        weights=np.zeros(len(sub), dtype=np.float32),
        sublabels=np.array(sub, dtype=np.int64),
        fine=np.array(fine, dtype=np.int64),
        coarse=np.array(coarse, dtype=np.int64),
    )
    return path


def perfect(true, rng):
    return rng.permutation(PSI)[true]


def random_assign(true, rng):
    return rng.integers(0, PSI, size=len(true))


def collapsed(true, rng):
    return np.zeros(len(true), dtype=np.int64)


def partial(frac):
    def f(true, rng):
        out = rng.permutation(PSI)[true].copy()
        flip = rng.random(len(true)) > frac
        out[flip] = rng.integers(0, PSI, size=int(flip.sum()))
        return out
    return f


def measure(assign, k):
    with tempfile.TemporaryDirectory() as tmp:
        path = write_dump(Path(tmp), assign, k)
        return analyse_dump(path, PSI, shuffles=10)


# --------------------------------------------------------------------------- #


def test_perfect_recovery_is_invisible_to_the_naive_metric():
    """The bug this module exists for."""
    res = measure(perfect, k=20)
    assert res["naive"] < 0.15, (
        f"naive metric scored {res['naive']:.3f} on a perfect permuted recovery; "
        "the premise of this analysis is that it cannot see one"
    )
    assert res["matched"] > 0.99, res["matched"]
    assert res["lift"] > 0.8, res["lift"]
    print(f"  perfect recovery: naive {res['naive']:.4f} (blind) vs "
          f"matched {res['matched']:.4f}, lift {res['lift']:+.4f}, "
          f"MI {res['mi_ratio']:.1f}x null")


def test_random_assignment_reads_zero_at_every_block_size():
    """The bias correction has to hold across k, since k varies across the sweep."""
    for k in (5, 25, 100):
        res = measure(random_assign, k)
        assert abs(res["lift"]) < 0.02, (
            f"k={k}: random assignment gave lift {res['lift']:+.4f}, expected ~0"
        )
        assert 0.9 < res["mi_ratio"] < 1.1, f"k={k}: MI ratio {res['mi_ratio']:.3f}"
        # The uncorrected numbers are wildly k-dependent, which is the whole
        # reason the null is needed: 0.29 at k=5 against 0.11 at k=100.
        print(f"  k={k:>3}: matched {res['matched']:.4f} (uncorrected, varies) -> "
              f"lift {res['lift']:+.4f}, MI {res['mi_ratio']:.2f}x")


def test_statistics_increase_with_real_structure():
    lifts, ratios = [], []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        res = measure(partial(frac), k=50)
        lifts.append(res["lift"])
        ratios.append(res["mi_ratio"])
        print(f"  {frac:>5.0%} correct: lift {res['lift']:+.4f}, "
              f"MI {res['mi_ratio']:.2f}x")
    for a, b in zip(lifts, lifts[1:]):
        assert b > a - 0.02, f"lift not monotone: {lifts}"
    assert lifts[-1] - lifts[0] > 0.5, lifts
    assert ratios[-1] > ratios[0] * 5, ratios


def test_collapse_is_distinguished_from_random():
    """Collapse and randomness both fail, but they are different failures."""
    col = measure(collapsed, k=50)
    rnd = measure(random_assign, k=50)
    assert col["clusters_used"] == 1.0, col["clusters_used"]
    assert rnd["clusters_used"] > PSI * 0.8, rnd["clusters_used"]
    assert col["mi"] < 1e-9, col["mi"]
    print(f"  collapsed: {col['clusters_used']:.0f} cluster, MI {col['mi']:.2e} | "
          f"random: {rnd['clusters_used']:.1f} clusters, MI {rnd['mi']:.4f}")


def test_misaligned_hierarchy_is_reported_not_averaged_away():
    """If fine // psi != coarse the mask is wrong and every number is garbage."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dump_epoch000.npz"
        np.savez_compressed(
            path,
            weights=np.zeros(4, dtype=np.float32),
            sublabels=np.array([0, 1, 2, 3], dtype=np.int64),
            fine=np.array([0, 1, 2, 99], dtype=np.int64),   # 99 // 15 = 6, not 0
            coarse=np.array([0, 0, 0, 0], dtype=np.int64),
        )
        try:
            analyse_dump(path, PSI, shuffles=2)
        except ValueError as exc:
            assert "fine // psi" in str(exc), str(exc)
            print(f"  raised: {str(exc).split(':')[-1].strip()[:60]}...")
            return
    raise AssertionError("a misaligned hierarchy was silently accepted")


def test_helpers_agree_with_hand_computed_values():
    pred = np.array([0, 0, 1, 1])
    true = np.array([0, 0, 1, 1])
    assert matched_accuracy(pred, true, PSI) == 1.0
    # Two balanced clusters perfectly aligned carry ln(2) nats.
    assert abs(mutual_information(pred, true) - np.log(2)) < 1e-9
    # Independent: cluster says nothing about intent.
    assert abs(mutual_information(np.array([0, 1, 0, 1]), true)) < 1e-9
    print(f"  matched=1.0, MI=ln2={np.log(2):.4f}, independent MI=0")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for test in tests:
        print(f"{test.__name__} ...")
        try:
            test()
        except Exception:  # noqa: BLE001
            failures.append(test.__name__)
            traceback.print_exc()
    print("-" * 72)
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
