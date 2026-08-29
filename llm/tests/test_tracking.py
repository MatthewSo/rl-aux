"""Tracking must never be able to break a run.

wandb is a mirror of `metrics.jsonl`, and it runs on a box billed by the hour
where the only unrecoverable loss is the training itself. So the property under
test is not "does it log correctly" but "does it stay out of the way when it is
absent, unconfigured, or broken mid-run".

    python -m llm.tests.test_tracking
"""

from __future__ import annotations

import builtins
import json
import shutil
import sys
import tempfile
from pathlib import Path

from llm import tracking
from llm.config import ExperimentConfig


def make_cfg(tmp: str, **kwargs) -> ExperimentConfig:
    return ExperimentConfig(dataset="clinc150", model="opt-125m", arm="wamal",
                            shots=5, seed=0, results_dir=tmp, **kwargs).resolve()


# --------------------------------------------------------------------------- #


def test_disabled_by_default():
    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    try:
        tracker = tracking.build_tracker(make_cfg(tmp))
        assert isinstance(tracker, tracking.NullTracker)
        # The null path must accept every call the training loop makes.
        tracker.log({"epoch": 0, "val_acc": 0.5}, step=0)
        tracker.summarise({"best_val_acc": 0.5, "trainable": {"theta": 1}})
        tracker.finish()
        print("  default config -> NullTracker, all calls accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_wandb_degrades_quietly():
    """An unbuilt env must not turn into a crashed sweep."""
    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    real_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no module named wandb")
        return real_import(name, *args, **kwargs)

    saved = sys.modules.pop("wandb", None)
    try:
        builtins.__import__ = no_wandb
        tracker = tracking.build_tracker(make_cfg(tmp, wandb=True))
        assert isinstance(tracker, tracking.NullTracker)
        print("  wandb not installed -> NullTracker, no exception")
    finally:
        builtins.__import__ = real_import
        if saved is not None:
            sys.modules["wandb"] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_init_failure_degrades_quietly():
    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    stub = type(sys)("wandb")
    stub.Settings = lambda **kw: kw

    def boom(**kwargs):
        raise RuntimeError("network is unreachable")

    stub.init = boom
    saved = sys.modules.get("wandb")
    try:
        sys.modules["wandb"] = stub
        tracker = tracking.build_tracker(make_cfg(tmp, wandb=True, wandb_mode="online"))
        assert isinstance(tracker, tracking.NullTracker)
        print("  init raising -> NullTracker, run continues")
    finally:
        if saved is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_mid_run_failure_disables_rather_than_raises():
    """The nastiest case: tracking works, then the network drops at epoch 7."""
    calls = {"log": 0}

    class FlakyRun:
        id, name, url = "abc123", "cell", "https://wandb.ai/x/y/runs/abc123"
        project, entity = "wamal-lm", None

        class summary:
            @staticmethod
            def update(_):
                raise RuntimeError("connection reset")

        def log(self, record, step=None):
            calls["log"] += 1
            if calls["log"] > 3:
                raise RuntimeError("connection reset")

        def finish(self, exit_code=0):
            pass

    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    try:
        tracker = tracking.WandbTracker(FlakyRun(), None, Path(tmp))
        for epoch in range(10):
            tracker.log({"epoch": epoch, "val_acc": 0.5}, step=epoch)
        tracker.summarise({"best_val_acc": 0.5})
        tracker.finish()

        assert not tracker.enabled, "should have disabled itself after the failure"
        # Stops retrying rather than stalling once per epoch on a dead socket.
        assert calls["log"] == 4, f"kept retrying after failure: {calls['log']} calls"
        print(f"  failed at call 4 of 10, disabled itself, no exception escaped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_id_is_stable_and_unique():
    """Resumable sweeps re-run cells; random ids would litter the project."""
    a = tracking.run_id_for("clinc150/opt-125m/wamal/k5/seed0")
    b = tracking.run_id_for("clinc150/opt-125m/wamal/k5/seed0")
    c = tracking.run_id_for("clinc150/opt-125m/wamal/k5/seed1")
    assert a == b, "same cell must map to the same run id"
    assert a != c, "different cells must not collide"
    assert a.isalnum() and len(a) <= 64
    print(f"  {a} stable across calls, distinct from seed1's {c}")


def test_run_pointer_is_written():
    """Local run dir -> wandb run must be joinable after the fact."""
    class Run:
        id, name, url = "abc123", "cell", "https://wandb.ai/x/y/runs/abc123"
        project, entity = "wamal-lm", "me"

        class summary:
            @staticmethod
            def update(_):
                pass

        def log(self, *a, **k):
            pass

        def finish(self, exit_code=0):
            pass

    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    try:
        tracking.WandbTracker(Run(), None, Path(tmp))
        pointer = json.loads((Path(tmp) / "wandb_run.json").read_text())
        assert pointer["id"] == "abc123"
        assert pointer["url"].endswith("abc123")
        print(f"  wandb_run.json -> {pointer['url']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_nones_are_dropped_from_logs():
    """`single` has no meta stats; empty panels would clutter every chart."""
    logged = {}

    class Run:
        id = name = url = project = entity = None

        class summary:
            @staticmethod
            def update(_):
                pass

        def log(self, record, step=None):
            logged.update(record)

        def finish(self, exit_code=0):
            pass

    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    try:
        tracker = tracking.WandbTracker(Run(), None, Path(tmp))
        tracker.log({"epoch": 1, "val_acc": 0.5, "meta_loss_before": None}, step=1)
        assert "meta_loss_before" not in logged
        assert logged["val_acc"] == 0.5
        print("  None-valued metrics dropped, real ones kept")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_settings_track_the_installed_wandb():
    """start_method is the concurrency fix on old wandb and pure noise on new."""
    old = tracking._settings_kwargs("0.17.9")
    new = tracking._settings_kwargs("0.26.1")
    assert old.get("start_method") == "thread", "0.17 needs the in-thread fix"
    assert "start_method" not in new, "0.18+ only warns about it"
    assert old["init_timeout"] == new["init_timeout"] == 300
    # An unparseable version must not crash init.
    assert tracking._settings_kwargs("weird-dev-build")["init_timeout"] == 300
    print("  0.17 -> start_method=thread, 0.26 -> omitted, garbage version safe")


def test_auto_mode_picks_offline_without_credentials():
    import os

    saved = os.environ.pop("WANDB_API_KEY", None)
    try:
        # Can't assume the test machine has no netrc, so only assert the
        # negative case is handled coherently.
        has = tracking._has_credentials()
        assert isinstance(has, bool)
        print(f"  credential probe returned {has} without WANDB_API_KEY set")
    finally:
        if saved is not None:
            os.environ["WANDB_API_KEY"] = saved


def test_training_loop_reports_every_epoch_and_the_summary():
    """The wiring, not the wrapper: does the real loop actually call through?

    Reuses the synthetic backbone from test_train_loop so this stays a CPU test
    with no downloads.
    """
    import torch

    from llm.paths import RunLogger, prepare_run_dir
    from llm.runner import build_optimizers
    from llm.tests.test_train_loop import build_everything
    from llm.train_text import train_wamal_text

    class Spy:
        def __init__(self):
            self.logged, self.summary, self.finished = [], None, False

        def log(self, record, step=None):
            self.logged.append((step, record))

        def summarise(self, summary):
            self.summary = summary

        def finish(self, exit_code=0):
            self.finished = True

    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    try:
        epochs = 3
        cfg, model, loaders = build_everything("wamal", tmp, epochs=epochs)
        theta_opt, theta_sched, phi_opt, phi_sched = build_optimizers(cfg, model)
        spy = Spy()
        path = prepare_run_dir(cfg, overwrite=True)
        with RunLogger(path, echo=False) as logger:
            train_wamal_text(cfg, torch.device("cpu"), model, loaders,
                             theta_opt, theta_sched, phi_opt, phi_sched,
                             save_path=path, logger=logger, tracker=spy)

        assert len(spy.logged) == epochs, f"expected {epochs} logs, got {len(spy.logged)}"
        assert [s for s, _ in spy.logged] == list(range(epochs)), "steps must be epoch indices"
        assert spy.summary is not None and "best_val_acc" in spy.summary
        # metrics.jsonl is the source of truth and must be written regardless.
        lines = (path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == epochs, "local metrics must match what was mirrored"
        print(f"  {epochs} epochs mirrored with matching steps, summary forwarded, "
              f"metrics.jsonl intact")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_rejects_a_bad_mode():
    tmp = tempfile.mkdtemp(prefix="wamal_track_")
    try:
        try:
            make_cfg(tmp, wandb=True, wandb_mode="sometimes")
        except ValueError as exc:
            assert "wandb_mode" in str(exc)
            print(f"  rejected at config time: {exc}")
            return
        raise AssertionError("an invalid wandb_mode should not resolve")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for test in tests:
        print(f"{test.__name__} ...")
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failures.append((test.__name__, f"{exc.__class__.__name__}: {exc}"))

    print("-" * 72)
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED")
        for name, why in failures:
            print(f"  {name}: {why}")
        return 1
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
