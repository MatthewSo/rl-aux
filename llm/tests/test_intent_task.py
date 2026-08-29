"""The intent-level task, where the auxiliary hierarchy is invented.

Phase 1 ran CLINC150 as 10 domains with 150 human intents as the auxiliary task.
An oracle hierarchy there is worth only ~+1pp, which is too low a ceiling for a
label-generating method to clear by much. `clinc150_intent` makes all 150 intents
the primary task with 5 invented sub-classes each: much harder, real headroom, and
MAXL's original setting where nothing anchors the sub-labels.

The hazard is that `fine` becomes a placeholder. Anything that scores against it
now produces a number that looks meaningful and is not, and the `human` arm would
train on a constant target while still reporting plausible accuracy. These tests
pin the guards that make that impossible rather than merely unlikely.

    python -m llm.tests.test_intent_task
"""

from __future__ import annotations

import traceback

import numpy as np

from llm.config import DATASET_DEFAULTS, ExperimentConfig


def cfg(**kw):
    base = dict(dataset="clinc150_intent", model="opt-125m", arm="wamal",
                shots=5, seed=0)
    base.update(kw)
    return ExperimentConfig(**base).resolve()


# --------------------------------------------------------------------------- #


def test_task_shape_is_150_way_with_5_invented_subclasses():
    c = cfg()
    assert c.num_primary == 150, c.num_primary
    assert c.psi == 5, c.psi
    assert c.num_auxiliary == 750, c.num_auxiliary
    # The domain-level task is untouched, so phase 1 stays reproducible.
    d = ExperimentConfig(dataset="clinc150", arm="wamal", shots=5).resolve()
    assert (d.num_primary, d.psi, d.num_auxiliary) == (10, 15, 150)
    print(f"  intent task: {c.num_primary}-way primary, psi={c.psi}, "
          f"{c.num_auxiliary} auxiliary | domain task unchanged at "
          f"{d.num_primary}/{d.psi}")


def test_human_arm_is_rejected_rather_than_silently_degenerate():
    """Its targets would be slot 0 of every block: a constant, not a hierarchy."""
    try:
        cfg(arm="human")
    except ValueError as exc:
        assert "placeholder" in str(exc), str(exc)
        print(f"  rejected: {str(exc).split('--')[0].strip()[:70]}...")
    else:
        raise AssertionError("the human arm was accepted on an invented hierarchy")

    for arm in ("single", "maxl", "wamal"):
        assert cfg(arm=arm).arm == arm, f"{arm} should be allowed"


def test_metrics_scored_against_fine_are_suppressed():
    assert cfg().has_fine_hierarchy is False
    assert ExperimentConfig(dataset="clinc150", arm="wamal").resolve() \
        .has_fine_hierarchy is True
    # Also gated: label_corruption needs the human arm, which is already barred.
    try:
        cfg(arm="human", label_corruption=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("corruption accepted on a task with no fine labels")
    print("  has_fine_hierarchy False -> aux_vs_fine_acc suppressed, human barred")


def test_few_shot_stratifies_per_intent_not_per_invented_subclass():
    """750 auxiliary slots exist but only 150 distinct `fine` values do.

    Passing num_auxiliary to the sampler would make it demand 750 classes in the
    pool and abort. Stratifying on the 150 placeholders is what yields k per intent.
    """
    c = cfg()
    assert c.num_stratify_classes == 150, c.num_stratify_classes
    d = ExperimentConfig(dataset="clinc150", arm="wamal").resolve()
    assert d.num_stratify_classes == d.num_auxiliary == 150
    print(f"  intent task stratifies on {c.num_stratify_classes} classes "
          f"(not {c.num_auxiliary}), so k means k per intent")


def test_placeholder_fine_labels_keep_the_mask_invariant():
    """`fine // psi == coarse` is what keeps each target inside its own window."""
    psi = DATASET_DEFAULTS["clinc150_intent"]["psi"]
    for intent in (0, 7, 149):
        fine = intent * psi
        assert fine // psi == intent
        assert 0 <= fine < 150 * psi
    print(f"  placeholder fine = intent*{psi} satisfies fine // psi == coarse")


def test_intent_runs_cannot_collide_with_domain_runs():
    assert "clinc150_intent" in cfg().cell_id()
    assert cfg().cell_id() != ExperimentConfig(
        dataset="clinc150", arm="wamal", shots=5, seed=0
    ).resolve().cell_id()
    print(f"  {cfg().cell_id()}")


def test_inter_seed_agreement_separates_shared_structure_from_noise():
    """The substitute for a recovery metric when there is no ground truth."""
    import tempfile
    from pathlib import Path

    from llm.analysis.recovery import inter_seed_agreement

    psi, blocks, per_block = 5, 20, 40
    rng = np.random.default_rng(0)
    primary = np.repeat(np.arange(blocks), per_block)

    def write(directory, sublabels, name):
        path = Path(directory) / name
        np.savez_compressed(path, sublabels=sublabels, primary=primary)
        return path

    with tempfile.TemporaryDirectory() as tmp:
        # Shared structure: both seeds recover the same partition under different
        # relabelings, which is exactly what a real discovery looks like.
        truth = rng.integers(0, psi, size=len(primary))
        perm_a, perm_b = rng.permutation(psi), rng.permutation(psi)
        a = write(tmp, primary * psi + perm_a[truth], "a.npz")
        b = write(tmp, primary * psi + perm_b[truth], "b.npz")
        shared = inter_seed_agreement([a, b], psi, shuffles=10)

        # Independent noise: no shared partition to find.
        c = write(tmp, primary * psi + rng.integers(0, psi, len(primary)), "c.npz")
        d = write(tmp, primary * psi + rng.integers(0, psi, len(primary)), "d.npz")
        noise = inter_seed_agreement([c, d], psi, shuffles=10)

    assert shared["mi_ratio"] > 3.0, shared
    assert noise["mi_ratio"] < 2.0, noise
    assert shared["mi_ratio"] > noise["mi_ratio"] * 2, (shared, noise)
    print(f"  shared partition {shared['mi_ratio']:.1f}x null vs "
          f"independent noise {noise['mi_ratio']:.2f}x")


def test_mismatched_test_sets_are_refused():
    """Comparing assignments over different examples would be meaningless."""
    import tempfile
    from pathlib import Path

    from llm.analysis.recovery import inter_seed_agreement

    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.npz"
        b = Path(tmp) / "b.npz"
        np.savez_compressed(a, sublabels=np.array([0, 1]), primary=np.array([0, 0]))
        np.savez_compressed(b, sublabels=np.array([0, 1]), primary=np.array([0, 1]))
        try:
            inter_seed_agreement([a, b], 5, shuffles=2)
        except ValueError as exc:
            assert "different test sets" in str(exc), str(exc)
            print("  refused assignments over different test sets")
            return
    raise AssertionError("mismatched test sets were compared anyway")


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
