"""Deterministic k-per-fine-class subsampling.

`dataset_loaders/capped_dataset.py` samples with an unseeded `random.sample`, so
in the vision experiments two arms at the same nominal cap trained on different
examples. Any accuracy gap then confounds the method with the draw. Here the
subset is a pure function of `(seed, dataset, k)` and is therefore byte-identical
across all four arms in a cell.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from llm.seeding import derived_seed


class FewShotSubset(Dataset):
    """k examples per FINE class, chosen deterministically.

    Sampling on the fine label rather than the coarse one keeps every intent
    represented. Sampling k per *domain* would let the label network see domains
    whose 15 intents are covered unevenly, which would confound the sub-label
    recovery analysis.
    """

    def __init__(self, base: Dataset, k: int | None, seed: int,
                 dataset_name: str = "", num_fine: int | None = None):
        self.base = base
        self.k = k

        if k is None:
            self.indices = list(range(len(base)))
            self.per_class_count = None
            return

        fine = np.asarray(getattr(base, "fine_labels", None) if
                          hasattr(base, "fine_labels") else
                          [base[i][2] for i in range(len(base))])

        # Keyed on the cell coordinates, not on a global RNG that earlier calls
        # could have advanced.
        rng = np.random.default_rng(derived_seed("fewshot", dataset_name, seed, k))

        classes = np.unique(fine)
        if num_fine is not None and len(classes) != num_fine:
            raise ValueError(
                f"expected {num_fine} fine classes in the pool, found {len(classes)}"
            )

        chosen: list[int] = []
        shortfalls: list[tuple[int, int]] = []
        for cls in classes:
            pool = np.flatnonzero(fine == cls)
            if len(pool) < k:
                shortfalls.append((int(cls), len(pool)))
                take = pool
            else:
                take = rng.choice(pool, size=k, replace=False)
            chosen.extend(int(i) for i in take)

        if shortfalls:
            raise ValueError(
                f"k={k} exceeds the available examples for {len(shortfalls)} "
                f"class(es), e.g. {shortfalls[:3]}. A partially-filled subset "
                "would make per-class counts differ between cells."
            )

        chosen.sort()
        self.indices = chosen
        self.per_class_count = k

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]

    @property
    def fine_labels(self) -> list[int]:
        base_fine = self.base.fine_labels
        return [base_fine[i] for i in self.indices]


class IndexSubset(Dataset):
    """A positional view onto another dataset, preserving `fine_labels`."""

    def __init__(self, base: Dataset, positions: list[int]):
        self.base = base
        self.positions = list(positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        return self.base[self.positions[idx]]

    @property
    def fine_labels(self) -> list[int]:
        base_fine = self.base.fine_labels
        return [base_fine[i] for i in self.positions]


class CorruptFineLabels(Dataset):
    """Replaces a fraction of fine labels with a random one from the same block.

    For the label-quality ladder. Phase 1 established that real hierarchical
    labels help (+0.8pp at k=5) while phi's learned ones do not, and phi's labels
    recover only ~15-20% of the true hierarchy. This interpolates between the two
    so the gain can be read as a function of label quality, which turns "wamal
    does not help" into a statement about how good learned labels would need to be.

    Corruption is fixed once per example, not resampled per epoch: resampling
    would average the noise away over a run and measure something else entirely.
    Replacements stay inside the example's coarse block, so the corrupted label is
    still a legal target under the hierarchy mask and only the *within-class*
    information degrades -- the coarse signal the primary task needs is untouched.
    """

    def __init__(self, base: Dataset, frac: float, seed: int, psi: int,
                 dataset_name: str = ""):
        self.base = base
        self.frac = frac
        self.psi = psi

        fine = np.asarray(base.fine_labels)
        rng = np.random.default_rng(
            derived_seed("label_corruption", dataset_name, seed,
                         int(round(frac * 1000)))
        )
        corrupt = rng.random(len(fine)) < frac
        block = fine // psi
        # Uniform over the psi slots in the block, including the true one, so the
        # ladder's endpoint is chance-within-block rather than guaranteed-wrong.
        replacement = block * psi + rng.integers(0, psi, size=len(fine))
        self._fine = np.where(corrupt, replacement, fine).astype(np.int64)
        self.n_corrupted = int(corrupt.sum())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        text, coarse, _ = self.base[idx]
        return text, coarse, int(self._fine[idx])

    @property
    def fine_labels(self) -> list[int]:
        return self._fine.tolist()

    @property
    def true_fine_labels(self) -> list[int]:
        return list(self.base.fine_labels)


def stratified_holdout(subset: Dataset, frac: float, seed: int,
                       dataset_name: str = "") -> tuple[list[int], list[int]]:
    """Split positions of `subset` into (train, holdout), stratified by fine class.

    Stratified rather than a flat random cut: a flat cut of a k=5 subset would
    leave some of the 150 intents absent from one side, and both sides need every
    intent -- the training side to keep the primary task balanced, the holdout
    side because it is the meta objective's query set and a missing intent there
    is a class phi never gets feedback on.

    Deterministic in `(dataset, seed, frac)` for the same reason the few-shot
    sampler is: two arms in one cell must see byte-identical splits, or any gap
    between them confounds the method with the draw.
    """
    fine = np.asarray(subset.fine_labels)
    rng = np.random.default_rng(derived_seed("meta_holdout", dataset_name, seed,
                                            int(round(frac * 1000))))

    train: list[int] = []
    holdout: list[int] = []
    for cls in np.unique(fine):
        pool = np.flatnonzero(fine == cls)
        held = max(1, round(len(pool) * frac))
        if len(pool) - held < 1:
            raise ValueError(
                f"class {int(cls)} has {len(pool)} example(s); withholding "
                f"{held} would leave nothing to train on"
            )
        picked = rng.choice(pool, size=held, replace=False)
        held_set = set(int(i) for i in picked)
        holdout.extend(sorted(held_set))
        train.extend(int(i) for i in pool if int(i) not in held_set)

    train.sort()
    holdout.sort()
    return train, holdout
