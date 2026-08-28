"""Prefetch backbone weights into the persistent HF cache.

Separated from training so that (a) a re-rented instance with the same volume
skips the download entirely, and (b) a network failure surfaces before any GPU
time is spent rather than 20 minutes into a sweep.

    python -m llm.bootstrap.download_models --tier phase1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from llm import env
from llm.models_registry import TIERS, ModelSpec, tier_models


WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pth")


def has_weights(path: str) -> bool:
    root = Path(path)
    return any(any(root.rglob(pattern)) for pattern in WEIGHT_GLOBS)


def download_one(spec: ModelSpec) -> tuple[bool, str]:
    from huggingface_hub import snapshot_download

    started = time.time()
    try:
        # Prefer safetensors and skip the redundant .bin twin where a repo ships
        # both, which halves the transfer on the larger checkpoints.
        path = snapshot_download(
            repo_id=spec.hf_id,
            ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5", "*.onnx"],
        )

        # Repos that ship *only* .bin (facebook/opt-125m among them) come back
        # from the call above with config and tokenizer but no weights, and no
        # error. Verifying rather than trusting the exit status is the
        # difference between finding out here and finding out mid-sweep.
        if not has_weights(path):
            path = snapshot_download(repo_id=spec.hf_id,
                                     ignore_patterns=["*.msgpack", "*.h5", "*.onnx"])
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"

    if not has_weights(path):
        return False, f"no weight files under {path} after download"

    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return True, f"{size/1024**2:.0f}MB  ({time.time() - started:.1f}s)  {path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="phase1", choices=sorted(TIERS))
    parser.add_argument("--model", action="append", default=None,
                        help="Explicit model slug; repeatable. Overrides --tier.")
    args = parser.parse_args(argv)

    env.configure()

    if args.model:
        from llm.models_registry import get
        specs = [get(s) for s in args.model]
    else:
        specs = tier_models(args.tier)

    print(f"[models] cache: {env.hf_home()}")
    print(f"[models] fetching {len(specs)} checkpoint(s)")

    failures = []
    for spec in specs:
        print(f"[models] {spec.slug:<16} {spec.hf_id}")
        ok, detail = download_one(spec)
        if ok:
            print(f"[models]   ok  {detail}")
        else:
            print(f"[models]   FAIL {detail}", file=sys.stderr)
            failures.append(spec.slug)

    if failures:
        print(f"\n[models] FAILED: {', '.join(failures)}", file=sys.stderr)
        print("[models] gated repos need `huggingface-cli login` first.", file=sys.stderr)
        return 1

    print("[models] all checkpoints cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
