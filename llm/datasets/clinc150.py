"""CLINC150 as a two-level hierarchy: 10 domains x 15 intents.

Why this dataset carries the main result: the hierarchy is exactly balanced, so
`psi` is a single integer and the masked-softmax machinery ports from the vision
code untouched. Domains are the primary task, intents are the human auxiliary
task, and 100 examples per intent means the few-shot ladder reaches k=100
without subsampling artifacts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from torch.utils.data import Dataset

DOMAINS_JSON = Path(__file__).with_name("clinc150_domains.json")

NUM_PRIMARY = 10
PSI = 15
NUM_FINE = NUM_PRIMARY * PSI

# HF exposes no config named "full"; "plus" is data_full.json with a larger OOS
# pool (100 train / 20 val / 30 test per in-scope intent, which is what we want).
HF_CONFIG = "plus"
OOS_LABEL = "oos"


@lru_cache(maxsize=1)
def load_hierarchy() -> tuple[list[str], dict[str, int], dict[str, int]]:
    """-> (domain_names, intent->fine_index, intent->coarse_index).

    Fine indices are assigned so that `fine // PSI == coarse`, which is the
    invariant that lets the human intent labels be used directly as auxiliary
    targets under the hierarchy mask with no remapping at train time.
    """
    raw = json.loads(DOMAINS_JSON.read_text())["domains"]

    if len(raw) != NUM_PRIMARY:
        raise ValueError(f"expected {NUM_PRIMARY} domains, found {len(raw)}")

    domain_names = list(raw.keys())
    intent_to_fine: dict[str, int] = {}
    intent_to_coarse: dict[str, int] = {}

    for coarse, domain in enumerate(domain_names):
        intents = raw[domain]
        if len(intents) != PSI:
            raise ValueError(
                f"domain {domain!r} has {len(intents)} intents, expected {PSI}. "
                "The hierarchy must be exactly balanced for a scalar psi."
            )
        for offset, intent in enumerate(intents):
            if intent in intent_to_fine:
                raise ValueError(f"intent {intent!r} appears in more than one domain")
            intent_to_fine[intent] = coarse * PSI + offset
            intent_to_coarse[intent] = coarse

    assert len(intent_to_fine) == NUM_FINE
    return domain_names, intent_to_fine, intent_to_coarse


def verify_against_dataset(label_names: list[str]) -> None:
    """Fail loudly if the asset and the dataset disagree on intent names.

    This is the check that makes a transcribed-from-a-PDF hierarchy safe to rely
    on. It cannot detect a correctly-spelled intent filed under the wrong domain
    (see SPEC D7), but it does catch every typo and omission.
    """
    _, intent_to_fine, _ = load_hierarchy()
    dataset_intents = {n for n in label_names if n != OOS_LABEL}
    ours = set(intent_to_fine)

    if dataset_intents != ours:
        missing = sorted(dataset_intents - ours)
        extra = sorted(ours - dataset_intents)
        raise ValueError(
            "clinc150_domains.json does not match the dataset's intent labels.\n"
            f"  in dataset but not in the asset ({len(missing)}): {missing}\n"
            f"  in the asset but not in dataset ({len(extra)}): {extra}\n"
            "Fix llm/datasets/clinc150_domains.json before running anything."
        )


class CLINC150Hierarchical(Dataset):
    """Yields `(text, coarse, fine)` with `fine // PSI == coarse`."""

    NUM_PRIMARY = NUM_PRIMARY
    NUM_FINE = NUM_FINE
    PSI = PSI

    def __init__(self, split: str = "train", include_oos: bool = False,
                 cache_dir: str | None = None):
        from datasets import load_dataset

        hf_split = {"train": "train", "val": "validation",
                    "validation": "validation", "test": "test"}[split]

        # No trust_remote_code: datasets>=4.0 dropped loading scripts, and the
        # hub serves clinc_oos as parquet.
        raw = load_dataset("clinc_oos", HF_CONFIG, split=hf_split,
                           cache_dir=cache_dir)

        label_names = raw.features["intent"].names
        verify_against_dataset(label_names)
        _, intent_to_fine, intent_to_coarse = load_hierarchy()

        self.split = split
        self.texts: list[str] = []
        self.coarse: list[int] = []
        self.fine: list[int] = []
        self.oos_texts: list[str] = []

        for text, label_id in zip(raw["text"], raw["intent"]):
            name = label_names[label_id]
            if name == OOS_LABEL:
                # Retained separately for the optional robustness probe; never
                # part of the classification problem itself.
                if include_oos:
                    self.oos_texts.append(text)
                continue
            self.texts.append(text)
            self.coarse.append(intent_to_coarse[name])
            self.fine.append(intent_to_fine[name])

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
        domain_names, intent_to_fine, _ = load_hierarchy()
        names = [""] * NUM_FINE
        for intent, fine in intent_to_fine.items():
            names[fine] = intent
        return names, domain_names
