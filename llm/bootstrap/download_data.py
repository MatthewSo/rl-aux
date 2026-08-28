"""Prefetch datasets and validate the CLINC150 hierarchy asset.

Runs on the cheap part of the rental, before any GPU work, so a bad hierarchy
file or a dataset-hub outage costs seconds rather than a sweep.

    python -m llm.bootstrap.download_data
"""

from __future__ import annotations

import argparse
import sys

from llm import env


def fetch_clinc150() -> tuple[bool, str]:
    from llm.datasets.clinc150 import CLINC150Hierarchical

    sizes = {}
    for split in ("train", "val", "test"):
        # The constructor runs verify_against_dataset, so this doubles as the
        # hierarchy check.
        data = CLINC150Hierarchical(split=split, cache_dir=str(env.data_dir()))
        sizes[split] = len(data)

    expected = {"train": 15000, "val": 3000, "test": 4500}
    for split, want in expected.items():
        if sizes[split] != want:
            return False, (
                f"{split} has {sizes[split]} in-scope examples, expected {want}. "
                "Wrong HF config, or OOS filtering changed."
            )
    return True, f"train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}"


def fetch_newsgroups() -> tuple[bool, str]:
    from llm.datasets.newsgroups import NewsgroupsHierarchical

    sizes = {}
    for split in ("train", "val", "test"):
        data = NewsgroupsHierarchical(split=split, cache_dir=str(env.data_dir()))
        sizes[split] = len(data)
    if min(sizes.values()) == 0:
        return False, f"empty split: {sizes}"
    return True, f"train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=None,
                        choices=["clinc150", "newsgroups"])
    args = parser.parse_args(argv)

    env.configure()
    print(f"[data] cache: {env.data_dir()}")

    wanted = args.dataset or ["clinc150", "newsgroups"]
    fetchers = {"clinc150": fetch_clinc150, "newsgroups": fetch_newsgroups}

    failures = []
    for name in wanted:
        print(f"[data] {name} ...", flush=True)
        try:
            ok, detail = fetchers[name]()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{exc.__class__.__name__}: {exc}"
        if ok:
            print(f"[data]   ok  {detail}")
        else:
            print(f"[data]   FAIL {detail}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"\n[data] FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("[data] all datasets cached and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
