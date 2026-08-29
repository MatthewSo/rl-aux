"""Sweep orchestration: cell enumeration, resumption, cost estimates, and the
multi-GPU work queue.

The parallel path is worth testing rather than eyeballing: it is the one place
with threads, and its failure modes (a GPU running two cells at once, a cell
dropped from the queue, `--keep-going` ignored) would show up as a corrupted or
incomplete sweep hours into a paid run.

Real subprocesses, fake payload -- `cell_command` is swapped for a trivial
python one-liner so the mechanics are exercised without training anything.

    python -m llm.tests.test_sweep
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from llm.config import ARMS, ExperimentConfig
from llm.paths import mark_complete, prepare_run_dir
from llm.scripts import sweep


def make_cells(tmp: str, n: int) -> list:
    return [
        ExperimentConfig(dataset="clinc150", model="opt-125m", arm=ARMS[i % len(ARMS)],
                         shots=5, seed=i, results_dir=tmp).resolve()
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #


def test_phase_sizes_match_the_spec():
    expected = {"phase1": 200, "phase2": 400, "smoke": 4}
    for name, want in expected.items():
        cells = sweep.enumerate_cells(sweep.PHASES[name], {})
        assert len(cells) == want, f"{name}: {len(cells)} cells, spec says {want}"
    print("  " + ", ".join(f"{k}={v}" for k, v in expected.items()))


def test_cells_are_unique():
    """A duplicated cell would silently halve the seeds behind an error bar."""
    for name in sweep.PHASES:
        cells = sweep.enumerate_cells(sweep.PHASES[name], {})
        ids = [c.cell_id() for c in cells]
        assert len(ids) == len(set(ids)), f"{name} has duplicate cells"
    print(f"  all {len(sweep.PHASES)} phases enumerate distinct cells")


def test_seed_is_the_outer_loop():
    """Interrupted sweeps should leave complete low-seed groups.

    Three seeds of every arm supports an error bar; ten seeds of one arm and
    none of the others supports nothing.
    """
    cells = sweep.enumerate_cells(sweep.PHASES["phase1"], {})
    first_block = cells[: len(ARMS) * 5]
    assert {c.seed for c in first_block} == {0}, "seed is not the outermost loop"
    print(f"  first {len(first_block)} cells are all seed 0, covering "
          f"{len({c.arm for c in first_block})} arms x {len({c.shots for c in first_block})} shot values")


def test_completed_cells_are_skipped():
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    try:
        cells = make_cells(tmp, 6)
        for cfg in cells[:2]:
            path = prepare_run_dir(cfg, overwrite=True)
            mark_complete(path, {"cell": cfg.cell_id(), "wall_time_s": 12.0,
                                 "arm": cfg.arm, "shots": cfg.shots})

        from llm.paths import is_complete

        pending = [c for c in cells if not is_complete(c)]
        assert len(pending) == 4, f"expected 4 pending, got {len(pending)}"
        print(f"  2 of 6 marked DONE -> {len(pending)} pending")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_estimate_scales_with_workers():
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    try:
        cells = make_cells(tmp, 8)
        for cfg in cells[:4]:
            path = prepare_run_dir(cfg, overwrite=True)
            mark_complete(path, {"cell": cfg.cell_id(), "arm": cfg.arm,
                                 "shots": cfg.shots, "wall_time_s": 3600.0})

        one, known = sweep.estimate_hours(cells, Path(tmp), 1)
        four, _ = sweep.estimate_hours(cells, Path(tmp), 4)

        assert abs(one - 8.0) < 0.01, f"expected 8 GPU-hours, got {one}"
        assert abs(four - 2.0) < 0.01, f"4 workers should quarter it, got {four}"
        assert known == 8, f"all cells should price from measurements, got {known}"
        print(f"  8 x 1h cells -> {one:.1f}h on 1 gpu, {four:.1f}h on 4")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_estimate_is_honest_with_no_data():
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    try:
        hours, known = sweep.estimate_hours(make_cells(tmp, 4), Path(tmp), 1)
        assert hours == 0.0 and known == 0, (
            "with no measurements the estimate must report nothing rather than guess"
        )
        print("  no prior runs -> no estimate, as intended")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stub_command(marker_dir: Path, fail_seeds: set[int]):
    """Replacement for cell_command: records the GPU it ran on, then exits."""
    def build(cfg, gpu, force):
        marker = marker_dir / f"{cfg.seed}.json"
        code = (
            "import json,os,sys,time\n"
            f"time.sleep(0.05)\n"
            f"json.dump({{'gpu': {gpu}, 'pid': os.getpid(), 'start': time.time()}}, "
            f"open({str(marker)!r}, 'w'))\n"
            f"sys.exit({1 if cfg.seed in fail_seeds else 0})\n"
        )
        return [sys.executable, "-c", code]
    return build


def test_parallel_runs_every_cell_exactly_once():
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    markers = Path(tmp) / "markers"
    markers.mkdir(parents=True)
    original = sweep.cell_command
    try:
        cells = make_cells(tmp, 12)
        sweep.cell_command = _stub_command(markers, set())

        failures = sweep.run_parallel(cells, [0, 1, 2], force=False, keep_going=False)

        assert not failures, f"unexpected failures: {failures}"
        produced = sorted(int(p.stem) for p in markers.glob("*.json"))
        assert produced == list(range(12)), f"cells run: {produced}"

        gpus_used = {json.loads(p.read_text())["gpu"] for p in markers.glob("*.json")}
        assert gpus_used == {0, 1, 2}, f"expected all 3 GPUs busy, saw {gpus_used}"
        print(f"  12 cells across GPUs {sorted(gpus_used)}, each run exactly once")
    finally:
        sweep.cell_command = original
        shutil.rmtree(tmp, ignore_errors=True)


def _record_spans(tmp: str, slots: list[int], n_cells: int):
    """Run n_cells over `slots` with stub subprocesses that record wall intervals.

    -> {gpu: [(start, end), ...]}. Observing real intervals rather than counting
    around the call: a counter would only restate the code's own structure,
    whereas timestamps show what concurrency actually happened.
    """
    spans = Path(tmp) / "spans"
    spans.mkdir(parents=True, exist_ok=True)
    original = sweep.cell_command

    def build(cfg, gpu, force):
        code = (
            "import json,time\n"
            "start = time.time()\n"
            "time.sleep(0.05)\n"
            f"json.dump({{'gpu': {gpu}, 'start': start, 'end': time.time()}}, "
            f"open({str(spans / f'{cfg.seed}.json')!r}, 'w'))\n"
        )
        return [sys.executable, "-c", code]

    try:
        sweep.cell_command = build
        sweep.run_parallel(make_cells(tmp, n_cells), slots, force=False, keep_going=True)
    finally:
        sweep.cell_command = original

    by_gpu: dict[int, list[tuple[float, float]]] = {}
    for path in spans.glob("*.json"):
        rec = json.loads(path.read_text())
        by_gpu.setdefault(rec["gpu"], []).append((rec["start"], rec["end"]))
    assert len(list(spans.glob("*.json"))) == n_cells, "not every cell ran"
    return by_gpu


def peak_concurrency(intervals: list[tuple[float, float]]) -> int:
    """Most cells alive at once on one GPU, by sweeping the interval endpoints."""
    events = [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals]
    events.sort()
    peak = live = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


def test_unpacked_runs_one_cell_per_gpu():
    """Default packing must not overlap: cells would otherwise race for VRAM."""
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    try:
        by_gpu = _record_spans(tmp, [0, 1, 2, 3], 16)
        for gpu, intervals in by_gpu.items():
            peak = peak_concurrency(intervals)
            assert peak == 1, f"gpu{gpu} ran {peak} cells at once with pack=1"
        # The opposite failure: serialising onto one GPU would satisfy the
        # overlap check while wasting the other three.
        assert len(by_gpu) == 4, f"expected work on 4 GPUs, saw {sorted(by_gpu)}"
        print(f"  16 cells / 4 GPUs unpacked, peak concurrency 1 everywhere; "
              f"counts {[len(v) for _, v in sorted(by_gpu.items())]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_packing_runs_several_cells_per_gpu_but_respects_the_cap():
    """pack=2 must actually overlap, and must not exceed 2 per GPU."""
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    try:
        slots = [g for _ in range(2) for g in (0, 1)]   # [0,1,0,1]
        by_gpu = _record_spans(tmp, slots, 12)

        peaks = {gpu: peak_concurrency(iv) for gpu, iv in by_gpu.items()}
        for gpu, peak in peaks.items():
            assert peak <= 2, f"gpu{gpu} exceeded the pack cap with {peak} cells"
        assert max(peaks.values()) == 2, (
            f"packing never actually overlapped; peaks {peaks}. "
            f"pack=2 that runs serially is a silent no-op."
        )
        assert set(by_gpu) == {0, 1}, f"expected both GPUs used, saw {sorted(by_gpu)}"
        print(f"  12 cells / 2 GPUs x pack 2: peak concurrency {peaks}, cap respected")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_slot_list_is_interleaved():
    """[0,1,0,1] not [0,0,1,1], so an early stop spreads work across devices."""
    gpus, pack = [0, 1, 2], 2
    slots = [g for _ in range(pack) for g in gpus]
    assert slots == [0, 1, 2, 0, 1, 2], slots
    assert len(slots) == len(gpus) * pack
    print(f"  gpus={gpus} pack={pack} -> slots={slots}")


def test_parallel_reports_failures_and_honours_keep_going():
    tmp = tempfile.mkdtemp(prefix="wamal_sweep_")
    markers = Path(tmp) / "markers"
    markers.mkdir(parents=True)
    original = sweep.cell_command
    try:
        cells = make_cells(tmp, 8)
        sweep.cell_command = _stub_command(markers, fail_seeds={2, 5})

        failures = sweep.run_parallel(cells, [0, 1], force=False, keep_going=True)

        assert len(failures) == 2, f"expected 2 failures, got {failures}"
        assert all("exit 1" in why for _, why in failures)
        # Every cell should still have been attempted.
        assert len(list(markers.glob("*.json"))) == 8
        print(f"  2 failing cells reported, remaining 6 still ran: "
              f"{[c for c, _ in failures]}")
    finally:
        sweep.cell_command = original
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
