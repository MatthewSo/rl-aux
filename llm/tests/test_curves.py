"""The training-curve diagnostics, especially WAMAL-vs-MAXL degeneration.

`aggregate` reports which arm won. If WAMAL has silently collapsed into MAXL --
every sample receiving the same auxiliary weight -- it still reports a plausible
accuracy, and the only thing that can reveal it is the per-sample weight
dispersion. So the detector has to fire on a collapsed run and stay quiet on a
healthy one; a check that does neither reliably is worse than none, because it
will be trusted.

    python -m llm.tests.test_curves
"""

from __future__ import annotations

import traceback

from llm.analysis.curves import diagnose


def make_run(arm: str, n: int = 20, *, mean=0.4, std=0.05,
             include_std: bool = True, val_from=0.5, val_to=0.9) -> dict:
    """A synthetic run whose only interesting axis is the weight dispersion."""
    epochs = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        record = {
            "epoch": i,
            "val_acc": val_from + (val_to - val_from) * frac,
            "train_acc_primary": 0.6 + 0.35 * frac,
            "train_loss_primary": 1.0 * (1 - 0.9 * frac),
            "train_weight_mean": mean(i) if callable(mean) else mean,
        }
        if include_std:
            record["train_weight_std"] = std(i) if callable(std) else std
        epochs.append(record)
    return {"arm": arm, "epochs": epochs, "dataset": "clinc150",
            "model": "opt-125m", "shots": 5, "seed": 0}


def collapse_notes(run) -> list[str]:
    return [n for n in diagnose(run)
            if "constant" in n or "MAXL" in n or "train_weight_std" in n]


def test_constant_per_sample_weights_are_flagged():
    run = make_run("wamal", std=0.0)
    notes = collapse_notes(run)
    assert notes, "zero dispersion is exactly MAXL and must be flagged"
    assert "MAXL" in notes[0], f"unhelpful message: {notes[0]}"
    print(f"  flagged: {notes[0]}")


def test_healthy_dispersion_is_not_flagged():
    run = make_run("wamal", mean=0.4, std=0.05)   # 12.5% of the mean
    assert not collapse_notes(run), (
        f"a run with real per-sample spread was flagged: {collapse_notes(run)}"
    )
    print("  mean 0.40 +/- 0.05 passes clean")


def test_drifting_mean_with_constant_weights_is_still_flagged():
    """The bug the old check had.

    Judging collapse by whether the *mean* moves across epochs passes a run in
    which every sample gets an identical weight that happens to drift. That is
    pure MAXL, so dispersion is the only thing worth testing.
    """
    run = make_run("wamal", mean=lambda i: 0.30 + 0.01 * i, std=0.0)
    notes = collapse_notes(run)
    assert notes, "a drifting but per-sample-constant weight is still MAXL"
    print(f"  drifting mean, zero spread -> {notes[0][:60]}...")


def test_missing_dispersion_field_is_reported_not_ignored():
    """Runs predating the field must say so rather than silently pass."""
    run = make_run("wamal", include_std=False)
    notes = collapse_notes(run)
    assert notes, "a run with no std must not be silently treated as healthy"
    assert "train_weight_std" in notes[0], f"unclear message: {notes[0]}"
    print(f"  reported: {notes[0]}")


def test_other_arms_are_never_flagged():
    """single and human have no weight head; maxl has one it never trains."""
    for arm in ("single", "human", "maxl"):
        run = make_run(arm, std=0.0, include_std=False)
        assert not collapse_notes(run), f"{arm} was flagged for weight collapse"
    print("  single/human/maxl exempt, as they must be")


def test_short_runs_are_not_flagged():
    """Early epochs are legitimately flat while the label network is near init."""
    run = make_run("wamal", n=5, std=0.0)
    assert not collapse_notes(run), "a 5-epoch run is too short to call collapsed"
    print("  5-epoch run not flagged")


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
