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
