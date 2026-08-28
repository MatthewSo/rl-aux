"""Turns an `ExperimentConfig` into a completed run."""

from __future__ import annotations

import torch

from llm import env
from llm.paths import RunLogger, is_complete, prepare_run_dir, run_dir
from llm.seeding import seed_everything


def build_optimizers(cfg, model):
    """Two optimizers over disjoint parameter groups.

    phi carries weight decay and theta does not, matching the paper's setup
    (label-weight network decay 5e-4).
    """
    theta_params = model.theta_parameters()
    phi_params = model.phi_parameters()

    if cfg.optimizer == "sgd":
        theta_opt = torch.optim.SGD(theta_params, lr=cfg.primary_lr,
                                    momentum=cfg.momentum)
        phi_opt = torch.optim.SGD(phi_params, lr=cfg.gen_lr,
                                  momentum=cfg.momentum,
                                  weight_decay=cfg.gen_weight_decay)
    else:
        theta_opt = torch.optim.AdamW(theta_params, lr=cfg.primary_lr,
                                      weight_decay=0.0)
        phi_opt = torch.optim.AdamW(phi_params, lr=cfg.gen_lr,
                                    weight_decay=cfg.gen_weight_decay)

    theta_sched = torch.optim.lr_scheduler.StepLR(
        theta_opt, step_size=cfg.step_size, gamma=cfg.gamma)
    phi_sched = torch.optim.lr_scheduler.StepLR(
        phi_opt, step_size=cfg.step_size, gamma=cfg.gamma)
    return theta_opt, theta_sched, phi_opt, phi_sched


def run(cfg, force: bool = False) -> dict:
    from llm.datasets import build_loaders, build_tokenizer
    from llm.models_registry import get
    from llm.networks import build_model
    from llm.train_text import train_wamal_text

    cfg.resolve()
    env.configure()

    if is_complete(cfg) and not force:
        import json
        path = run_dir(cfg)
        return json.loads((path / "summary.json").read_text())

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    save_path = prepare_run_dir(cfg, overwrite=True)
    with RunLogger(save_path) as logger:
        logger.print(f"config: {cfg.to_json()}")
        logger.print(f"device: {device}")

        spec = get(cfg.model)
        tokenizer = build_tokenizer(spec.hf_id, cache_dir=str(env.hf_home() / "hub"))
        loaders = build_loaders(cfg, tokenizer)
        logger.print(f"data: {loaders['train_size']} train, "
                     f"{len(loaders['val'].dataset)} val, "
                     f"{len(loaders['test'].dataset)} test")

        model = build_model(cfg, device)
        report = model.memory_report()
        logger.print(
            f"params: base {report['frozen_base']/1e6:.1f}M frozen | "
            f"theta {report['theta_trainable']/1e6:.2f}M | "
            f"phi {report['phi_trainable']/1e6:.2f}M | "
            f"{report['lora_modules']} adapted linears"
        )

        theta_opt, theta_sched, phi_opt, phi_sched = build_optimizers(cfg, model)

        return train_wamal_text(
            cfg=cfg,
            device=device,
            model=model,
            loaders=loaders,
            theta_optimizer=theta_opt,
            theta_scheduler=theta_sched,
            phi_optimizer=phi_opt,
            phi_scheduler=phi_sched,
            save_path=save_path,
            logger=logger,
        )
