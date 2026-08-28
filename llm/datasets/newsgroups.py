"""20 Newsgroups as a 4 x 4 hierarchy.

Second dataset, chosen for a different failure mode than CLINC150: long documents
instead of short utterances, a naturally-occurring rather than
product-manager-designed hierarchy, and a 1990s corpus that predates every
model in the ladder.

Only 16 of the 20 groups are used. Four superclasses of exactly four groups keeps
`psi` a scalar, matching the vision code's assumption. `comp.os.ms-windows.misc`
is dropped to bring `comp.*` from five groups to four, and `misc.forsale`,
`alt.atheism` and `soc.religion.christian` are dropped because their
superclasses cannot be filled to four without merging unrelated topics.
"""

from __future__ import annotations

from torch.utils.data import Dataset

NUM_PRIMARY = 4
PSI = 4
NUM_FINE = NUM_PRIMARY * PSI

HIERARCHY: dict[str, list[str]] = {
    "comp": [
        "comp.graphics",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    ],
    "rec": [
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
    ],
    "sci": [
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
    ],
    "talk": [
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ],
}

CATEGORIES = [g for groups in HIERARCHY.values() for g in groups]

# Headers carry the newsgroup name outright, and quoted replies plus signature
# footers leak it almost as reliably. Without this the task is near-trivial and
# the few-shot comparison measures nothing.
REMOVE = ("headers", "footers", "quotes")

# Fraction of train held out for model selection. 20NG ships only train/test.
VAL_FRACTION = 0.1


def _fine_index() -> tuple[dict[str, int], dict[str, int], list[str]]:
    group_to_fine: dict[str, int] = {}
    group_to_coarse: dict[str, int] = {}
    for coarse, (_super, groups) in enumerate(HIERARCHY.items()):
        assert len(groups) == PSI
        for offset, group in enumerate(groups):
            group_to_fine[group] = coarse * PSI + offset
            group_to_coarse[group] = coarse
    return group_to_fine, group_to_coarse, list(HIERARCHY.keys())


class NewsgroupsHierarchical(Dataset):
    """Yields `(text, coarse, fine)` with `fine // PSI == coarse`."""

    NUM_PRIMARY = NUM_PRIMARY
    NUM_FINE = NUM_FINE
    PSI = PSI

    def __init__(self, split: str = "train", cache_dir: str | None = None,
                 seed: int = 0):
        import numpy as np
        from sklearn.datasets import fetch_20newsgroups

        sk_subset = "test" if split == "test" else "train"
        raw = fetch_20newsgroups(
            subset=sk_subset,
            categories=CATEGORIES,
            remove=REMOVE,
            data_home=cache_dir,
            shuffle=False,
        )

        group_to_fine, group_to_coarse, _ = _fine_index()
        # sklearn renumbers targets against its own sorted category list, so map
        # back through the names rather than assuming index alignment.
        names = list(raw.target_names)

        texts, coarse, fine = [], [], []
        for text, target in zip(raw.data, raw.target):
            body = text.strip()
            # Stripping headers/quotes empties a nontrivial number of posts.
            if not body:
                continue
            group = names[target]
            texts.append(body)
            coarse.append(group_to_coarse[group])
            fine.append(group_to_fine[group])

        if split in ("train", "val", "validation"):
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(texts))
            cut = int(len(texts) * VAL_FRACTION)
            keep = order[cut:] if split == "train" else order[:cut]
            keep = sorted(keep.tolist())
            texts = [texts[i] for i in keep]
            coarse = [coarse[i] for i in keep]
            fine = [fine[i] for i in keep]

        self.split = split
        self.texts, self.coarse, self.fine = texts, coarse, fine

        for c, f in zip(self.coarse, self.fine):
            assert f // PSI == c, "fine//psi == coarse invariant violated"

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int, int]:
        return self.texts[idx], self.coarse[idx], self.fine[idx]

    @property
    def fine_labels(self) -> list[int]:
        return self.fine

    def label_names(self) -> tuple[list[str], list[str]]:
        group_to_fine, _, supers = _fine_index()
        names = [""] * NUM_FINE
        for group, fine in group_to_fine.items():
            names[fine] = group
        return names, supers
