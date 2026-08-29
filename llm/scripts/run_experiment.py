"""Single entry point for one cell of the run matrix.

One script with an `--arm` flag, rather than the repo's one-file-per-cell
convention. `wamal/dense_scripts/vit_coco_single_dense.py` and its WAMAL twin
are byte-identical in configuration, so the COCO "single-task" baseline was
actually running WAMAL. Sharing the code path makes that class of mistake
impossible.

    python -m llm.scripts.run_experiment --arm wamal --shots 10 --seed 0
"""

from __future__ import annotations

import argparse
import json

from llm.config import ARMS, DATASETS, ExperimentConfig
from llm.models_registry import MODELS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    cell = p.add_argument_group("cell")
    cell.add_argument("--dataset", default="clinc150", choices=DATASETS)
    cell.add_argument("--model", default="opt-125m", choices=sorted(MODELS))
    cell.add_argument("--arm", default="wamal", choices=ARMS)
    cell.add_argument("--shots", type=int, default=100,
                      help="examples per FINE class; omit --shots for the full split")
    cell.add_argument("--seed", type=int, default=0)

    opt = p.add_argument_group("optimization")
    opt.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw"))
    opt.add_argument("--primary-lr", type=float, default=1e-2)
    opt.add_argument("--gen-lr", type=float, default=1e-3)
    opt.add_argument("--batch-size", type=int, default=32)
    opt.add_argument("--epochs", type=int, default=None, dest="total_epoch",
                     help="default is k-dependent; see config.EPOCHS_BY_SHOTS")
    opt.add_argument("--grad-clip", type=float, default=1.0)

    meta = p.add_argument_group("wamal")
    meta.add_argument("--val-range", type=float, default=5.0)
    meta.add_argument("--entropy-loss-factor", type=float, default=0.2)
    meta.add_argument("--inner-lr", type=float, default=None)
    meta.add_argument("--normalize-batch-weights", action="store_true")
    meta.add_argument("--first-order", action="store_true",
                      help="drop the second-order term; ablation and OOM escape hatch")

    net = p.add_argument_group("backbone")
    net.add_argument("--lora-r", type=int, default=16)
    net.add_argument("--lora-alpha", type=int, default=32)
    net.add_argument("--lora-dropout", type=float, default=0.0)
    net.add_argument("--pooling", default="last", choices=("last", "mean"))
    net.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    net.add_argument("--max-len", type=int, default=None)

    run = p.add_argument_group("runtime")
    run.add_argument("--gpu", type=int, default=0, help="matches wamal/argparse.py")
    run.add_argument("--results-dir", default=None)
    run.add_argument("--num-workers", type=int, default=2)
    run.add_argument("--dump-every", type=int, default=10)
    run.add_argument("--log-cosine", action="store_true",
                     help="2 extra backward passes per step; diagnostic only")
    run.add_argument("--no-checkpoint", action="store_true")
    run.add_argument("--force", action="store_true",
                     help="rerun even if a DONE marker exists")
    run.add_argument("--tag", default="")

    track = p.add_argument_group("tracking (wandb; metrics.jsonl is always written)")
    track.add_argument("--wandb", action="store_true",
                       help="mirror per-epoch metrics to Weights & Biases")
    track.add_argument("--wandb-project", default="wamal-lm")
    track.add_argument("--wandb-entity", default=None,
                       help="team or username; defaults to your wandb default")
    track.add_argument("--wandb-group", default="",
                       help="groups runs in the UI; sweep.py sets this to the phase")
    track.add_argument("--wandb-mode", default="auto",
                       choices=("auto", "online", "offline", "disabled"),
                       help="auto goes online when a key is present, else offline")
    return p


def config_from_args(args) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        arm=args.arm,
        shots=args.shots,
        seed=args.seed,
        optimizer=args.optimizer,
        primary_lr=args.primary_lr,
        gen_lr=args.gen_lr,
        batch_size=args.batch_size,
        total_epoch=args.total_epoch,
        grad_clip=args.grad_clip,
        val_range=args.val_range,
        entropy_loss_factor=args.entropy_loss_factor,
        inner_lr=args.inner_lr,
        normalize_batch_weights=args.normalize_batch_weights,
        first_order=args.first_order,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pooling=args.pooling,
        dtype=args.dtype,
        max_len=args.max_len,
        device=f"cuda:{args.gpu}",
        results_dir=args.results_dir,
        num_workers=args.num_workers,
        dump_every=args.dump_every,
        log_cosine=args.log_cosine,
        save_checkpoint=not args.no_checkpoint,
        tag=args.tag,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
    ).resolve()


def main(argv: list[str] | None = None) -> int:
    from llm import env

    # Before importing anything that pulls in torch or numpy. The BLAS libraries
    # read their thread counts once at load time, so setting the caps after the
    # import silently does nothing.
    env.configure()

    from llm.runner import run

    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    summary = run(cfg, force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
