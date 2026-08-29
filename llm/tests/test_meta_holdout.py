"""The held-out meta objective, and the bookkeeping that keeps it comparable.

Phase 1's meta objective is the primary loss on the batch theta was just fitted
on, so it decays to ~1e-5 partway through a low-k run and phi stops learning.
`meta_holdout_frac` measures that loss on withheld data instead.

Two classes of failure are worth testing, and the second is the dangerous one:

- The split itself. It must be deterministic and stratified, or arms within a
  cell train on different data and any gap between them confounds the method
  with the draw -- the exact bug `fewshot.py` exists to prevent.
- Result bookkeeping. A holdout run shares `(arm, shots, seed)` with its base
  counterpart, and `paired_deltas` pivots on `arm` using `pivot_table`, whose
  default aggfunc is `mean`. Without `variant` as a key, a holdout run is
  silently averaged into the base result with no error and no warning.

    python -m llm.tests.test_meta_holdout
"""

from __future__ import annotations

import traceback

from llm.analysis.aggregate import variant_of
from llm.config import ExperimentConfig


class FakeSubset:
    """Positions carrying fine labels, k per class, which is all the split reads."""

    def __init__(self, num_fine: int, k: int):
        self._fine = [c for c in range(num_fine) for _ in range(k)]

    def __len__(self):
        return len(self._fine)

    def __getitem__(self, idx):
        return ("text", self._fine[idx] // 15, self._fine[idx])

    @property
    def fine_labels(self):
        return self._fine


def cfg(**kw):
    base = dict(dataset="clinc150", model="opt-125m", arm="wamal", shots=5, seed=0)
    base.update(kw)
    return ExperimentConfig(**base).resolve()


# --------------------------------------------------------------------------- #


def test_split_is_stratified_over_every_fine_class():
    """Both sides need every class: the query side is what phi gets feedback on."""
    from llm.datasets.fewshot import stratified_holdout

    subset = FakeSubset(num_fine=150, k=5)
    keep, held = stratified_holdout(subset, 0.2, seed=0, dataset_name="clinc150")

    fine = subset.fine_labels
    kept_classes = {fine[i] for i in keep}
    held_classes = {fine[i] for i in held}
    assert kept_classes == set(range(150)), "a class vanished from the train split"
    assert held_classes == set(range(150)), "a class vanished from the query split"
    assert len(keep) + len(held) == len(subset), "split lost or duplicated examples"
    assert not set(keep) & set(held), "an example is in both splits"
    assert len(held) == 150, f"expected 1 per class at k=5, got {len(held)}"
    print(f"  k=5 x 150 classes -> {len(keep)} train / {len(held)} query, "
          f"all 150 classes on both sides")


def test_split_is_deterministic_across_calls():
    """Arms in one cell must get byte-identical splits."""
    from llm.datasets.fewshot import stratified_holdout

    subset = FakeSubset(num_fine=20, k=10)
    a = stratified_holdout(subset, 0.2, seed=3, dataset_name="clinc150")
    b = stratified_holdout(subset, 0.2, seed=3, dataset_name="clinc150")
    assert a == b, "same coordinates produced different splits"

    c = stratified_holdout(subset, 0.2, seed=4, dataset_name="clinc150")
    assert a != c, "different seeds produced the same split"
    print("  identical for equal (dataset, seed, frac), different across seeds")


def test_holdout_runs_get_their_own_directory():
    """Otherwise a holdout run overwrites or is skipped by its base counterpart."""
    base, held = cfg(), cfg(meta_holdout_frac=0.2)
    assert base.cell_id() != held.cell_id(), (
        f"both variants resolve to {base.cell_id()} and would share a directory"
    )
    assert base.variant() == "base" and held.variant() == "holdout0.2"
    # Existing phase-1 paths must not move, or 200 completed runs look unfinished.
    assert base.cell_id() == "clinc150/opt-125m/wamal/k5/seed0", base.cell_id()
    print(f"  base -> {base.cell_id()}\n  holdout -> {held.cell_id()}")


def test_variants_are_never_averaged_together():
    """The silent-corruption case: pivot_table means duplicate keys."""
    import pandas as pd

    from llm.analysis.aggregate import paired_deltas, summarise, to_frame

    def row(arm, variant, acc, seed=0):
        return {"dataset": "clinc150", "model": "opt-125m", "shots": 5,
                "arm": arm, "seed": seed, "variant": variant,
                "test_acc_at_best_val": acc, "wall_time_s": 1.0,
                "final_test_aux_vs_fine_acc": 0.06}

    # Same (arm, shots, seed) in both variants, with very different accuracies.
    rows = []
    for seed in range(3):
        rows += [row("human", "base", 0.90, seed), row("wamal", "base", 0.88, seed),
                 row("human", "holdout0.2", 0.80, seed),
                 row("wamal", "holdout0.2", 0.84, seed)]
    frame = to_frame(rows)

    table = summarise(frame)
    assert "variant" in table.columns, "variant is not a grouping key"
    wamal = table[table["arm"] == "wamal"].set_index("variant")["acc_mean"]
    assert abs(wamal["base"] - 0.88) < 1e-9, (
        f"base wamal reported {wamal['base']}, expected 0.88 -- averaged with holdout"
    )
    assert abs(wamal["holdout0.2"] - 0.84) < 1e-9, wamal["holdout0.2"]

    deltas = paired_deltas(frame, baseline="human", treatment="wamal")
    assert "variant" in deltas.columns, "paired deltas ignore variant"
    by_variant = deltas.set_index("variant")["delta_mean"]
    # base: wamal loses by 2pp. holdout: wamal wins by 4pp. Opposite signs, so
    # any averaging would show up as a near-zero delta in both rows.
    assert abs(by_variant["base"] - (-0.02)) < 1e-9, by_variant["base"]
    assert abs(by_variant["holdout0.2"] - 0.04) < 1e-9, by_variant["holdout0.2"]
    assert (deltas["n_pairs"] == 3).all(), (
        f"expected 3 pairs per variant, got {deltas['n_pairs'].tolist()}"
    )
    print(f"  base delta {by_variant['base']:+.3f}, "
          f"holdout delta {by_variant['holdout0.2']:+.3f}, kept separate")


def test_old_runs_resolve_to_base():
    """The 200 phase-1 runs have no such field and must still aggregate."""
    assert variant_of({}) == "base"
    assert variant_of({"meta_holdout_frac": None}) == "base"
    assert variant_of({"meta_holdout_frac": 0.0}) == "base"
    assert variant_of({"meta_holdout_frac": 0.2}) == "holdout0.2"
    print("  absent/None/0.0 -> base; 0.2 -> holdout0.2")


def test_existing_base_results_aggregate_unchanged():
    """Regression guard for the 200 completed phase-1 runs.

    `variant` became a grouping key in `summarise` and an index level in
    `paired_deltas`. Runs written before the field existed have no
    `meta_holdout_frac` in their config, so this walks a synthetic tree shaped
    exactly like those and checks they still land in one row per (shots, arm)
    with the accuracy they recorded.
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    from llm.analysis.aggregate import load_runs, paired_deltas, summarise, to_frame

    tmp = Path(tempfile.mkdtemp(prefix="wamal_agg_"))
    try:
        accs = {"human": 0.910, "wamal": 0.892}
        for arm, acc in accs.items():
            for seed in range(3):
                d = tmp / "clinc150" / "opt-125m" / arm / "k5" / f"seed{seed}"
                d.mkdir(parents=True)
                (d / "summary.json").write_text(json.dumps({
                    "dataset": "clinc150", "model": "opt-125m", "arm": arm,
                    "shots": 5, "seed": seed, "test_acc_at_best_val": acc,
                    "final_test_aux_vs_fine_acc": 0.06, "wall_time_s": 200.0,
                }))
                # No meta_holdout_frac, exactly as the existing runs were written.
                (d / "config.json").write_text(json.dumps({
                    "optimizer": "sgd", "primary_lr": 0.01, "total_epoch": 40,
                }))
                (d / "DONE").write_text("x")

        rows = load_runs(tmp)
        assert len(rows) == 6, f"read {len(rows)} runs, expected 6"
        assert {r["variant"] for r in rows} == {"base"}, "old runs are not `base`"

        frame = to_frame(rows)
        table = summarise(frame)
        assert len(table) == 2, f"expected one row per arm, got {len(table)}"
        for _, row in table.iterrows():
            assert abs(row["acc_mean"] - accs[row["arm"]]) < 1e-9, row.to_dict()
            assert row["seeds"] == 3, row.to_dict()

        deltas = paired_deltas(frame, baseline="human", treatment="wamal")
        assert len(deltas) == 1 and deltas.iloc[0]["n_pairs"] == 3
        assert abs(deltas.iloc[0]["delta_mean"] - (0.892 - 0.910)) < 1e-9
        print(f"  6 legacy runs -> 2 rows, all `base`, "
              f"paired delta {deltas.iloc[0]['delta_mean']:+.3f} over 3 seeds")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corruption_is_fixed_per_example_not_resampled():
    """Resampling each epoch would average the noise away and measure nothing."""
    from llm.datasets.fewshot import CorruptFineLabels

    base = FakeSubset(num_fine=150, k=5)
    a = CorruptFineLabels(base, 0.5, seed=0, psi=15, dataset_name="clinc150")
    b = CorruptFineLabels(base, 0.5, seed=0, psi=15, dataset_name="clinc150")
    assert a.fine_labels == b.fine_labels, "corruption is not deterministic"
    # Stable across repeated access, which is what "not resampled" means in
    # practice: the loader reads every example once per epoch.
    assert [a[i][2] for i in range(20)] == [a[i][2] for i in range(20)]

    c = CorruptFineLabels(base, 0.5, seed=1, psi=15, dataset_name="clinc150")
    assert a.fine_labels != c.fine_labels, "different seeds gave the same corruption"
    print(f"  deterministic in seed; {a.n_corrupted}/{len(base)} examples corrupted")


def test_corruption_stays_inside_the_coarse_block():
    """Outside its block the label is illegal under the mask, and the coarse
    signal the primary task needs would be damaged too -- a different experiment."""
    from llm.datasets.fewshot import CorruptFineLabels

    base = FakeSubset(num_fine=150, k=5)
    true = base.fine_labels
    for frac in (0.25, 1.0):
        d = CorruptFineLabels(base, frac, seed=0, psi=15, dataset_name="clinc150")
        got = d.fine_labels
        for i, (t, g) in enumerate(zip(true, got)):
            assert t // 15 == g // 15, (
                f"example {i}: {t} -> {g} crossed from block {t//15} to {g//15}"
            )
        changed = sum(1 for t, g in zip(true, got) if t != g)
        # At frac=1.0 every example is redrawn uniformly over psi slots, so ~1/15
        # of them land back on the truth. Full corruption means chance, not wrong.
        expected = frac * (1 - 1 / 15)
        assert abs(changed / len(true) - expected) < 0.06, (
            f"frac={frac}: {changed/len(true):.3f} changed, expected ~{expected:.3f}"
        )
        print(f"  frac={frac}: {changed/len(true):.1%} differ from truth, "
              f"all inside their block")


def test_ladder_variants_are_separate_from_base_and_from_each_other():
    from llm.analysis.aggregate import variant_of

    rungs = [cfg(arm="human", label_corruption=c) for c in (0.25, 0.5, 1.0)]
    ids = {c.cell_id() for c in rungs} | {cfg(arm="human").cell_id()}
    assert len(ids) == 4, f"ladder rungs collide: {sorted(ids)}"
    assert rungs[0].variant() == "corrupt0.25"
    assert variant_of({"label_corruption": 0.5}) == "corrupt0.5"
    # Both knobs at once must still be distinguishable.
    both = cfg(arm="human", label_corruption=0.5, meta_holdout_frac=0.2)
    assert both.variant() == "holdout0.2_corrupt0.5", both.variant()
    print(f"  {sorted(v.variant() for v in rungs)} all distinct from base")


def test_corruption_is_rejected_on_arms_that_ignore_fine_labels():
    """A run directory saying corrupt0.5 must mean corruption actually happened."""
    for arm in ("single", "maxl", "wamal"):
        try:
            cfg(arm=arm, label_corruption=0.5)
        except ValueError:
            continue
        raise AssertionError(f"{arm} accepted label_corruption, which it ignores")
    assert cfg(arm="human", label_corruption=0.5).label_corruption == 0.5
    print("  rejected for single/maxl/wamal, accepted for human")


def test_impossible_fractions_are_rejected():
    """A silent 0-example training split would be far worse than an error."""
    for shots, frac, why in [(1, 0.2, "k=1 leaves nothing to train on"),
                             (5, 0.5, "half or more is a different experiment"),
                             (5, -0.1, "negative")]:
        try:
            cfg(shots=shots, meta_holdout_frac=frac)
        except ValueError:
            continue
        raise AssertionError(f"accepted shots={shots} frac={frac} ({why})")

    for shots, frac in [(2, 0.2), (5, 0.2), (10, 0.4)]:
        c = cfg(shots=shots, meta_holdout_frac=frac)
        assert c.variant().startswith("holdout")
    print("  rejects k=1, frac>=0.5 and negatives; accepts k=2 upward")


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
