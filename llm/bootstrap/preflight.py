"""The gate between renting a box and spending money on it.

Everything here is chosen to fail in seconds on the cheapest model. Check 5 is
the one that matters: a real end-to-end WAMAL step exercises the dual-adapter
forward, the second-order inner update and the writer, which is where the
implementation risk is concentrated.

    python -m llm.bootstrap.preflight --tier phase1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import traceback

from llm import env
from llm.models_registry import TIERS, tier_models

CHECKS: list[tuple[str, str]] = []


def _report(name: str, ok: bool, detail: str) -> bool:
    status = " ok " if ok else "FAIL"
    print(f"[{status}] {name}: {detail}", flush=True)
    CHECKS.append((name, detail if ok else f"FAILED: {detail}"))
    return ok


def check_torch_and_gpu() -> bool:
    import torch

    detail = f"torch {torch.__version__}"
    if not torch.cuda.is_available():
        return _report("gpu", False,
                       f"{detail}, but CUDA is not available. On a GPU rental "
                       "this usually means a driver/torch mismatch in the image.")
    props = torch.cuda.get_device_properties(0)
    bf16 = torch.cuda.is_bf16_supported()
    detail += (f", cuda {torch.version.cuda}, {torch.cuda.device_count()}x "
               f"{props.name}, {props.total_memory/1024**3:.1f}GB, "
               f"bf16={'yes' if bf16 else 'no (will use fp32)'}")
    return _report("gpu", True, detail)


def check_models_cached(tier: str) -> bool:
    """Load from cache with the hub disabled.

    Offline is the point: it proves download_models actually populated the cache
    rather than the model quietly re-downloading later, mid-sweep, on a box
    whose network may have changed.
    """
    import os

    import torch

    prior = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    ok = True
    try:
        for spec in tier_models(tier):
            try:
                from transformers import AutoConfig, AutoTokenizer

                from llm.bootstrap.download_models import has_weights

                cache = str(env.hf_home() / "hub")
                AutoConfig.from_pretrained(spec.hf_id, cache_dir=cache,
                                           trust_remote_code=spec.trust_remote_code)
                AutoTokenizer.from_pretrained(spec.hf_id, cache_dir=cache)

                # Config and tokenizer caching happily without weights is a real
                # failure mode, not a hypothetical: a download filtered to
                # safetensors leaves a .bin-only repo in exactly that state.
                repo_dir = (env.hf_home() / "hub" /
                            f"models--{spec.hf_id.replace('/', '--')}")
                if not has_weights(str(repo_dir)):
                    raise FileNotFoundError(
                        f"config and tokenizer are cached but no weight file is "
                        f"present under {repo_dir}"
                    )
                _report(f"model:{spec.slug}", True, f"{spec.hf_id} cached with weights")
            except Exception as exc:  # noqa: BLE001
                ok = _report(f"model:{spec.slug}", False,
                             f"{exc.__class__.__name__}: {exc}. "
                             "Run llm.bootstrap.download_models first.") and ok
    finally:
        if prior is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior
    del torch
    return ok


def check_datasets() -> bool:
    ok = True
    try:
        from llm.datasets.clinc150 import CLINC150Hierarchical

        data = CLINC150Hierarchical(split="train", cache_dir=str(env.data_dir()))
        ok = _report("data:clinc150", True,
                     f"{len(data)} train examples, hierarchy validated") and ok
    except Exception as exc:  # noqa: BLE001
        ok = _report("data:clinc150", False, f"{exc.__class__.__name__}: {exc}") and ok
    return ok


def check_end_to_end(gpu: int) -> bool:
    """A real two-epoch WAMAL run at k=5 on the smallest model."""
    import torch

    from llm.config import ExperimentConfig
    from llm.runner import run

    tmp = tempfile.mkdtemp(prefix="wamal_preflight_")
    try:
        cfg = ExperimentConfig(
            dataset="clinc150",
            model="opt-125m",
            arm="wamal",
            shots=5,
            seed=0,
            total_epoch=2,
            batch_size=8,
            device=f"cuda:{gpu}",
            results_dir=tmp,
            num_workers=0,
            save_checkpoint=False,
            dump_every=1,
        ).resolve()

        started = time.time()
        summary = run(cfg, force=True)
        elapsed = time.time() - started

        acc = summary["test_acc_at_best_val"]
        if acc != acc:  # NaN
            return _report("end_to_end", False,
                           "test accuracy is NaN; the focal loss diverged")

        peak = summary.get("peak_memory_mb")
        report = summary["trainable"]
        # ~1x base resident, not 2x: the shared-backbone invariant in memory
        # terms. Two copies of OPT-125M in bf16 would be ~500MB rather than ~250.
        base_mb = report["frozen_base"] * 2 / 1024**2
        detail = (f"{elapsed:.1f}s, test acc {acc:.3f}, peak {peak}MB, "
                  f"base {base_mb:.0f}MB resident (1x), "
                  f"theta {report['theta_trainable']/1e6:.2f}M / "
                  f"phi {report['phi_trainable']/1e6:.2f}M trainable")

        if peak is not None and peak > 4 * base_mb + 2048:
            return _report("end_to_end", False,
                           f"peak memory {peak}MB is far above the ~1x base "
                           f"budget ({base_mb:.0f}MB); the backbone may be "
                           "duplicated. " + detail)

        del torch
        return _report("end_to_end", True, detail)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _report("end_to_end", False, f"{exc.__class__.__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="phase1", choices=sorted(TIERS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true",
                        help="skip the end-to-end check (not recommended)")
    args = parser.parse_args(argv)

    env.configure()
    print("preflight")
    print(env.describe())
    print("-" * 72)

    ok = check_torch_and_gpu()
    ok = check_models_cached(args.tier) and ok
    ok = check_datasets() and ok
    if not args.skip_train:
        ok = check_end_to_end(args.gpu) and ok

    print("-" * 72)
    if ok:
        print("PREFLIGHT PASSED - safe to start a sweep")
        return 0
    print("PREFLIGHT FAILED - fix the above before spending GPU hours",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
