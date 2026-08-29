"""End-to-end exercise of the training loop on a synthetic dataset.

No network, no GPU, no HuggingFace: a tiny random backbone and a fake tokenizer
stand in for the real ones. The point is to prove the loop mechanics -- all four
arms, the per-sample dumps, the DONE marker, validation-based selection -- rather
than to learn anything.

Catching a loop bug here costs seconds. Catching it on a rented GPU costs an hour
and a sweep.

    python -m llm.tests.test_train_loop
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from llm.config import ExperimentConfig
from llm.datasets.collate import IndexedDataset
from llm.networks.lora import inject_dual_lora
from llm.networks.wamal_lm import WamalLM
from llm.paths import RunLogger, is_complete, prepare_run_dir
from llm.runner import build_optimizers
from llm.tests.test_lora_functional import HIDDEN, VOCAB, TinyBackbone
from llm.train_text import train_wamal_text

NUM_PRIMARY, PSI = 3, 4
NUM_FINE = NUM_PRIMARY * PSI
SEQ = 6


class FakeTextDataset(Dataset):
    """Synthetic (text, coarse, fine) with the fine//psi == coarse invariant."""

    def __init__(self, per_class: int, seed: int = 0):
        rng = torch.Generator().manual_seed(seed)
        self.texts, self.coarse, self.fine = [], [], []
        for fine in range(NUM_FINE):
            for _ in range(per_class):
                tokens = torch.randint(0, VOCAB, (SEQ,), generator=rng).tolist()
                self.texts.append(" ".join(str(t) for t in tokens))
                self.coarse.append(fine // PSI)
                self.fine.append(fine)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.coarse[idx], self.fine[idx]

    @property
    def fine_labels(self):
        return self.fine


class FakeCollator:
    def __call__(self, batch):
        ids = torch.tensor([[int(t) for t in b[0].split()] for b in batch])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "coarse": torch.tensor([b[1] for b in batch]),
            "fine": torch.tensor([b[2] for b in batch]),
            "index": torch.tensor([b[3] for b in batch]),
        }


def build_everything(arm: str, results_dir: str, epochs: int = 2):
    torch.manual_seed(0)
    cfg = ExperimentConfig(
        dataset="clinc150", model="opt-125m", arm=arm, shots=4, seed=0,
        num_primary=NUM_PRIMARY, psi=PSI, total_epoch=epochs, batch_size=8,
        device="cpu", results_dir=results_dir, num_workers=0,
        save_checkpoint=False, dump_every=1, primary_lr=0.05, gen_lr=0.05,
        log_cosine=True,
    ).resolve()

    base = TinyBackbone()
    for param in base.parameters():
        param.requires_grad_(False)
    inject_dual_lora(base, targets=("q_proj", "v_proj"), r=4, alpha=8)
    model = WamalLM(base=base, feature_dim=HIDDEN, num_primary=NUM_PRIMARY,
                    psi=PSI, model_slug="tiny")

    collator = FakeCollator()
    train = FakeTextDataset(per_class=4, seed=0)
    evalset = FakeTextDataset(per_class=2, seed=1)
    loaders = {
        "train": DataLoader(IndexedDataset(train), batch_size=8, shuffle=True,
                            collate_fn=collator),
        "val": DataLoader(IndexedDataset(evalset), batch_size=8, collate_fn=collator),
        "test": DataLoader(IndexedDataset(evalset), batch_size=8, collate_fn=collator),
        "train_size": len(train),
    }
    return cfg, model, loaders


def build_with_holdout(arm: str, results_dir: str, frac: float = 0.25):
    """Same as `build_everything` but with the meta query split carved out.

    Exercises the real split helper rather than a hand-made one, so the test
    fails if the stratification or the loader wiring changes underneath it.
    """
    from llm.datasets.fewshot import IndexSubset, stratified_holdout

    cfg, model, loaders = build_everything(arm, results_dir)
    cfg.meta_holdout_frac = frac

    train = FakeTextDataset(per_class=4, seed=0)
    keep, held = stratified_holdout(train, frac, seed=0, dataset_name="clinc150")
    collator = FakeCollator()
    loaders["train"] = DataLoader(
        IndexedDataset(IndexSubset(train, keep)), batch_size=8, shuffle=True,
        collate_fn=collator,
    )
    loaders["meta"] = DataLoader(
        IndexedDataset(IndexSubset(train, held)), batch_size=8, shuffle=True,
        collate_fn=collator,
    )
    loaders["train_size"] = len(keep)
    return cfg, model, loaders, len(keep), len(held)


def run_arm(arm: str, results_dir: str) -> dict:
    cfg, model, loaders = build_everything(arm, results_dir)
    theta_opt, theta_sched, phi_opt, phi_sched = build_optimizers(cfg, model)
    save_path = prepare_run_dir(cfg, overwrite=True)
    with RunLogger(save_path, echo=False) as logger:
        summary = train_wamal_text(
            cfg, torch.device("cpu"), model, loaders,
            theta_opt, theta_sched, phi_opt, phi_sched, save_path, logger,
        )
    return cfg, summary, save_path


# --------------------------------------------------------------------------- #


def test_all_four_arms_run_and_complete():
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        for arm in ("single", "human", "maxl", "wamal"):
            cfg, summary, path = run_arm(arm, tmp)

            assert is_complete(cfg), f"{arm}: DONE marker missing"
            assert (path / "summary.json").exists()
            assert (path / "config.json").exists()

            acc = summary["test_acc_at_best_val"]
            assert acc == acc, f"{arm}: test accuracy is NaN"
            assert 0.0 <= acc <= 1.0, f"{arm}: accuracy {acc} out of range"

            records = [json.loads(line)
                       for line in (path / "metrics.jsonl").read_text().splitlines()]
            assert len(records) == cfg.total_epoch, (
                f"{arm}: {len(records)} metric rows for {cfg.total_epoch} epochs"
            )
            for record in records:
                for key in ("train_loss_primary", "val_acc", "test_acc"):
                    value = record[key]
                    assert value == value, f"{arm}: {key} is NaN at epoch {record['epoch']}"

            print(f"  {arm:7s} ok  test={acc:.3f}  {summary['wall_time_s']}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_meta_step_runs_only_for_generated_label_arms():
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        for arm, expected in [("single", False), ("human", False),
                              ("maxl", True), ("wamal", True)]:
            _, _, path = run_arm(arm, tmp)
            first = json.loads((path / "metrics.jsonl").read_text().splitlines()[0])
            ran = first["meta_loss_after"] is not None
            assert ran == expected, (
                f"{arm}: meta step ran={ran}, expected {expected}"
            )
            print(f"  {arm:7s} meta step ran={ran} (expected {expected})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_only_wamal_records_weights():
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        import numpy as np

        for arm, expects_weights in [("human", None), ("maxl", False), ("wamal", True)]:
            _, _, path = run_arm(arm, tmp)
            dumps = sorted(path.glob("dump_epoch*.npz"))
            if expects_weights is None:
                assert not dumps, f"{arm}: should not dump (no label network)"
                print(f"  {arm:7s} no dumps, as expected")
                continue

            assert dumps, f"{arm}: expected per-sample dumps"
            data = np.load(dumps[-1])
            assert (data["sublabels"] >= 0).all(), f"{arm}: unfilled sub-label slots"
            # Every generated sub-label must sit inside its own class window.
            assert (data["sublabels"] // PSI == data["coarse"]).all(), (
                f"{arm}: generated sub-label escaped the hierarchy mask"
            )

            has_weights = not np.all(np.isnan(data["weights"]))
            assert has_weights == expects_weights, (
                f"{arm}: weights recorded={has_weights}, expected {expects_weights}"
            )
            print(f"  {arm:7s} dumps ok, weights={has_weights}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wamal_logs_per_sample_weight_dispersion():
    """The only field that can tell WAMAL apart from MAXL.

    A per-sample weight that is the same for every sample is MAXL with a rescaled
    auxiliary loss. The mean cannot show that; the std across samples can, so it
    has to be in every epoch record for the arms that learn weights.
    """
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        for arm, learns_weights in [("single", False), ("human", False),
                                    ("maxl", False), ("wamal", True)]:
            _, _, path = run_arm(arm, tmp)
            records = [json.loads(line)
                       for line in (path / "metrics.jsonl").read_text().splitlines()]

            for key in ("train_weight_mean", "train_weight_std",
                        "train_weight_min", "train_weight_max"):
                assert key in records[0], f"{arm}: {key} missing from the record"

            got = records[-1]
            if not learns_weights:
                assert got["train_weight_std"] is None, (
                    f"{arm}: does not train the weight head, so its output must "
                    f"not be reported as if it meant something"
                )
                print(f"  {arm:7s} no weight stats, as expected")
                continue

            mean, std = got["train_weight_mean"], got["train_weight_std"]
            lo, hi = got["train_weight_min"], got["train_weight_max"]
            assert None not in (mean, std, lo, hi), f"{arm}: incomplete weight stats"
            assert std >= 0.0, f"{arm}: negative std {std}"
            assert std > 0.0, (
                f"{arm}: std is exactly 0, so every sample got weight {mean} -- "
                f"either the weight head collapsed or it is not per-sample at all"
            )
            assert lo <= mean <= hi, f"{arm}: mean {mean} outside [{lo}, {hi}]"
            assert std <= (hi - lo), f"{arm}: std {std} exceeds the range {hi - lo}"
            print(f"  {arm:7s} weight {mean:.4f} +/- {std:.4f}, range [{lo:.4f}, {hi:.4f}]")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rerunning_a_cell_replaces_rather_than_appends_metrics():
    """An interrupted cell restarts from epoch 0, so its metrics must too.

    Appending left two runs concatenated in one file with the epoch counter
    jumping backwards partway through, which every epoch-indexed analysis then
    read as a single implausible curve.
    """
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        cfg, _, path = run_arm("wamal", tmp)
        first = (path / "metrics.jsonl").read_text()
        assert first.strip(), "no metrics written on the first run"

        _, _, path_again = run_arm("wamal", tmp)
        assert path_again == path, "re-run wrote to a different directory"
        records = [json.loads(line)
                   for line in (path / "metrics.jsonl").read_text().splitlines()]

        assert len(records) == cfg.total_epoch, (
            f"{len(records)} records for {cfg.total_epoch} epochs -- the previous "
            f"attempt was appended to instead of replaced"
        )
        epochs = [r["epoch"] for r in records]
        assert epochs == sorted(set(epochs)), (
            f"epoch sequence {epochs} is not strictly increasing, so this file "
            f"holds more than one run"
        )
        assert epochs[0] == 0, f"first record is epoch {epochs[0]}, expected 0"

        # Replaced, not destroyed: the previous attempt is the only evidence of
        # why it died, which is worth keeping on a preemptible box.
        prev = path / "metrics.jsonl.prev"
        assert prev.exists(), "previous attempt was discarded rather than kept"
        assert prev.read_text() == first, "metrics.jsonl.prev is not the old file"
        print(f"  re-run left {len(records)} records (epochs {epochs[0]}..{epochs[-1]}), "
              f"old file preserved as metrics.jsonl.prev")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_meta_step_can_measure_its_loss_on_withheld_data():
    """The whole point of the holdout variant: an outer loss that can't saturate.

    Under the paper-faithful path the outer loss is measured on the batch theta
    was just fitted on, so it decays toward zero and phi stops learning. Here it
    must be measured on the query split instead, which theta never trains on.

    Asserted structurally rather than by watching the loss decay -- two epochs on
    random data is far too short for saturation to show up, so a decay-based check
    would pass whether or not the query split was being used at all.
    """
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        cfg, model, loaders, n_train, n_held = build_with_holdout("wamal", tmp)
        assert n_held > 0 and n_train > 0, "degenerate split"
        assert n_train + n_held == NUM_FINE * 4, "split lost examples"

        seen = {"support": 0, "query": 0}
        real_forward = model.forward_primary

        def spy(input_ids, attention_mask, params=None):
            # The inner step runs at theta on support; the outer loss runs with
            # fast weights on whichever split the query loader supplies.
            key = "query" if params is not None else "support"
            seen[key] += input_ids.size(0)
            return real_forward(input_ids, attention_mask, params=params)

        model.forward_primary = spy
        theta_opt, theta_sched, phi_opt, phi_sched = build_optimizers(cfg, model)
        save_path = prepare_run_dir(cfg, overwrite=True)
        with RunLogger(save_path, echo=False) as logger:
            summary = train_wamal_text(
                cfg, torch.device("cpu"), model, loaders,
                theta_opt, theta_sched, phi_opt, phi_sched, save_path, logger,
            )
        model.forward_primary = real_forward

        assert seen["query"] > 0, "the outer loss never ran on a query batch"
        # One query example per support example, since the stream cycles the
        # smaller split to cover the larger one.
        per_epoch = seen["query"] / cfg.total_epoch
        assert per_epoch >= n_train * 0.8, (
            f"only {per_epoch:.0f} query examples per epoch for {n_train} support "
            f"examples -- the query split is not covering the support split"
        )

        records = [json.loads(line)
                   for line in (save_path / "metrics.jsonl").read_text().splitlines()]
        assert len(records) == cfg.total_epoch
        for r in records:
            assert r["meta_loss_before"] is not None, "meta step did not run"
            assert r["meta_loss_after"] is not None

        # The run must land in its own directory, never the base one.
        assert "holdout" in str(save_path), save_path
        assert 0.0 <= summary["test_acc_at_best_val"] <= 1.0
        print(f"  {n_train} support / {n_held} query, "
              f"{per_epoch:.0f} query examples per epoch, "
              f"wrote to .../{save_path.parent.parent.name}/...")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_holdout_shrinks_training_data_for_every_arm():
    """Baselines must lose the withheld examples too, or the comparison is rigged.

    If only the meta arms gave up training data, wamal would be measured against
    baselines that saw more of it, and the deficit under test would be confounded
    by the split rather than explained by it.
    """
    from llm.datasets.fewshot import stratified_holdout

    train = FakeTextDataset(per_class=4, seed=0)
    keep, held = stratified_holdout(train, 0.25, seed=0, dataset_name="clinc150")

    # The split depends only on (dataset, seed, frac) -- not on the arm -- so
    # every arm in a cell trains on an identical subset.
    for arm in ("single", "human", "maxl", "wamal"):
        again = stratified_holdout(train, 0.25, seed=0, dataset_name="clinc150")
        assert again == (keep, held), f"{arm} would get a different split"

    fine = train.fine_labels
    assert {fine[i] for i in keep} == set(range(NUM_FINE)), "class lost from train"
    assert len(keep) == NUM_FINE * 3 and len(held) == NUM_FINE * 1
    print(f"  split is arm-independent: {len(keep)} train / {len(held)} query "
          f"for all four arms")


def test_selection_uses_validation_not_test():
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        cfg, summary, path = run_arm("wamal", tmp)
        records = [json.loads(line)
                   for line in (path / "metrics.jsonl").read_text().splitlines()]

        best_by_val = max(records, key=lambda r: r["val_acc"])
        assert summary["best_epoch"] == best_by_val["epoch"], (
            "selected epoch does not match the best validation epoch"
        )
        assert summary["test_acc_at_best_val"] == best_by_val["test_acc"]

        # The reported number must be the test accuracy at the val-selected
        # epoch, which is only coincidentally the max test accuracy.
        best_test = max(r["test_acc"] for r in records)
        print(f"  selected epoch {summary['best_epoch']} by val; "
              f"test there = {summary['test_acc_at_best_val']:.3f}, "
              f"max test over epochs = {best_test:.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_completed_runs_are_skipped():
    """The property the whole resumable-sweep story rests on."""
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        cfg, summary, _ = run_arm("single", tmp)
        assert is_complete(cfg)

        fresh = ExperimentConfig(**{
            k: v for k, v in cfg.to_dict().items()
            if k not in ("derived",)
        })
        fresh.resolve()
        assert is_complete(fresh), "a fresh config for the same cell was not seen as done"
        print(f"  {cfg.cell_id()} recognised as complete from a fresh config")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_single_arm_ignores_the_auxiliary_head():
    """Arm 1 must be a genuine single-task baseline.

    `vit_coco_single_dense.py` was configured identically to its WAMAL twin, so
    the COCO baseline was silently running WAMAL. This asserts the analogous
    thing cannot happen here: with arm=single the auxiliary head must never
    receive a gradient.
    """
    tmp = tempfile.mkdtemp(prefix="wamal_loop_")
    try:
        cfg, model, loaders = build_everything("single", tmp, epochs=1)
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters() if "auxiliary_head" in n}

        theta_opt, theta_sched, phi_opt, phi_sched = build_optimizers(cfg, model)
        save_path = prepare_run_dir(cfg, overwrite=True)
        with RunLogger(save_path, echo=False) as logger:
            train_wamal_text(cfg, torch.device("cpu"), model, loaders,
                             theta_opt, theta_sched, phi_opt, phi_sched,
                             save_path, logger)

        after = dict(model.named_parameters())
        for name, original in before.items():
            assert torch.allclose(original, after[name].detach()), (
                f"arm=single moved {name}; the auxiliary task is leaking in"
            )
        print(f"  auxiliary head unchanged across {len(before)} tensors")
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
