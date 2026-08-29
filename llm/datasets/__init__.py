"""Hierarchical text datasets and few-shot sampling."""

from __future__ import annotations

from torch.utils.data import DataLoader

from llm import env
from llm.datasets.collate import (
    IndexedDataset,
    IndexedTextCollator,
    TextCollator,
    build_tokenizer,
)
from llm.datasets.fewshot import FewShotSubset, IndexSubset, stratified_holdout

__all__ = [
    "build_dataset",
    "build_loaders",
    "build_tokenizer",
    "FewShotSubset",
    "IndexSubset",
    "stratified_holdout",
    "TextCollator",
    "IndexedTextCollator",
    "IndexedDataset",
]


def build_dataset(name: str, split: str, seed: int = 0):
    cache = str(env.data_dir())
    if name == "clinc150":
        from llm.datasets.clinc150 import CLINC150Hierarchical

        return CLINC150Hierarchical(split=split, cache_dir=cache)
    if name == "newsgroups":
        from llm.datasets.newsgroups import NewsgroupsHierarchical

        # seed=0 always: the train/val carve-out must not move between cells, or
        # model selection would be comparing against different validation sets.
        return NewsgroupsHierarchical(split=split, cache_dir=cache, seed=0)
    raise KeyError(f"unknown dataset {name!r}")


def build_loaders(cfg, tokenizer) -> dict[str, DataLoader]:
    """train/val/test loaders. Only train is subsampled to k shots."""
    from llm.seeding import dataloader_generator, worker_init_fn

    collator = IndexedTextCollator(tokenizer, cfg.max_len)

    train_base = build_dataset(cfg.dataset, "train", cfg.seed)
    train = FewShotSubset(
        train_base,
        k=cfg.shots,
        seed=cfg.seed,
        dataset_name=cfg.dataset,
        num_fine=cfg.num_auxiliary,
    )

    # The withheld split leaves theta's training set for every arm, so that the
    # baselines wamal is compared against saw exactly the same examples. Only the
    # arms with a meta step get a loader over the remainder.
    meta_subset = None
    if cfg.meta_holdout_frac > 0:
        keep, held = stratified_holdout(
            train, cfg.meta_holdout_frac, cfg.seed, cfg.dataset
        )
        meta_subset = IndexSubset(train, held)
        train = IndexSubset(train, keep)

    loaders = {
        "train": DataLoader(
            IndexedDataset(train),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            generator=dataloader_generator(cfg.seed),
            worker_init_fn=worker_init_fn,
            drop_last=False,
            pin_memory=True,
        )
    }
    if meta_subset is not None and cfg.needs_meta_step:
        loaders["meta"] = DataLoader(
            IndexedDataset(meta_subset),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collator,
            # Deliberately 0: this loader is re-iterated several times per epoch
            # (it is much smaller than train), and respawning workers each cycle
            # would cost more than the loading it saves.
            num_workers=0,
            generator=dataloader_generator(cfg.seed + 1),
            drop_last=False,
        )
        loaders["meta_size"] = len(meta_subset)
    for split in ("val", "test"):
        data = build_dataset(cfg.dataset, split, cfg.seed)
        loaders[split] = DataLoader(
            IndexedDataset(data),
            # Evaluation has no backward pass, so a bigger batch is free.
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    loaders["train_size"] = len(train)
    return loaders
