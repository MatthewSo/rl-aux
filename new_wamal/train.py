"""Train one cell.

    python -m new_wamal.train --dataset cifar100_20 --arch vgg16 --method wamal --seed 0

Differences from wamal/train_network.py (all deliberate):

* Model selection on a held-out validation split. ``summary.json`` reports
  test accuracy at the best-validation epoch, the final epoch, and the mean of
  the last 5 epochs. Max-over-epochs test accuracy is recorded only as
  ``test_max_DO_NOT_REPORT`` for comparison with the old numbers.
* Accuracy is correct/total (the old code divided by the nominal batch size).
* No gradient cosine-similarity logging (it cost two extra backward passes per
  step, in every arm including single-task).
* Single-task runs build no generator and do no meta step, so timing is fair.
* The generator is evaluated under ``no_grad`` in the inner step (its graph was
  built and discarded there before; the update to theta is identical).
* Per-epoch wall-clock (inner / meta / eval) and peak GPU memory are logged.
* Per-sample weights, losses and auxiliary assignments are dumped (float16)
  at a few epochs for the weight / noise analyses.

The bi-level update itself, losses (focal), entropy term, weight scaling
2**(2*r*u - r) and LR schedule follow the original implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.optim as optim

from new_wamal import data as D
from new_wamal.models import build_models, METHODS
from wamal.networks.utils import model_fit, model_entropy


@dataclass
class Config:
    dataset: str = "cifar100_20"
    arch: str = "vgg16"
    method: str = "wamal"
    seed: int = 0
    # optimisation (defaults = paper's VGG16 / CIFAR setting)
    epochs: int = 200
    batch_size: int = 100
    lr: float = 0.02
    gen_lr: float = 1e-3
    gen_wd: float = 5e-4
    step_size: int = 50
    gamma: float = 0.5
    optimizer: str = "sgd"           # sgd | adamw
    wd: float = 0.0
    # WAMAL
    psi: int = 5
    r: float = 5.0
    entropy: float = 0.2
    label_arch: str = ""            # generator backbone; default = arch
    pretrained: bool = False
    # data
    data_root: str = "./data"
    val_frac: float = 0.1
    split_seed: int = 0
    train_frac: float = 1.0
    noise_type: str = "none"         # none | sym | asym
    noise_rate: float = 0.0
    img224: bool = False
    cluster_labels: str = ""         # .npy, required by weights_only
    # bookkeeping
    out: str = "results_new/scratch"
    workers: int = 2
    threads: int = 2
    dump_epochs: str = "1,mid,last"
    max_batches: int = 0             # >0: truncate epochs (smoke tests only)
    device: str = "cuda"


def scale(u, r):
    return torch.pow(2.0, 2 * r * u - r)


def inner_sgd_update(model, loss, lr):
    fast = OrderedDict(model.named_parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    return OrderedDict((n, w if g is None else w - lr * g) for (n, w), g in zip(fast.items(), grads))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(model, loader, device, n_cls, max_batches=0):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for b, (x, y, _) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            p, _ = model(x)
            loss_sum += model_fit(p, y, device, pri=True, num_output=n_cls).sum().item()
            correct += (p.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1), loss_sum / max(total, 1)


def run(cfg: Config):
    os.makedirs(cfg.out, exist_ok=True)
    if os.path.exists(os.path.join(cfg.out, "DONE")):
        print("already done:", cfg.out); return
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    m = METHODS[cfg.method]

    bundle = D.build(cfg)
    n_cls = bundle.num_primary
    primary, gen, n_aux = build_models(cfg, n_cls, bundle.image_shape, device)

    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.workers, collate_fn=bundle.collate,
              pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0)
    g = torch.Generator(); g.manual_seed(cfg.seed)
    dl_train = torch.utils.data.DataLoader(bundle.train, shuffle=True, generator=g, **kw)
    dl_val = torch.utils.data.DataLoader(bundle.val, shuffle=False, **kw)
    dl_test = torch.utils.data.DataLoader(bundle.test, shuffle=False, **kw)

    if cfg.optimizer == "sgd":
        opt = optim.SGD(primary.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    else:
        opt = optim.AdamW(primary.parameters(), lr=cfg.lr, weight_decay=cfg.wd or 0.05)
    sched = optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    gen_opt = gen_sched = None
    if gen is not None:
        gen_opt = optim.SGD(gen.parameters(), lr=cfg.gen_lr, weight_decay=cfg.gen_wd)
        gen_sched = optim.lr_scheduler.StepLR(gen_opt, step_size=cfg.step_size, gamma=cfg.gamma)

    # ---- fixed per-sample quantities --------------------------------------
    N = len(bundle.train)
    fixed_labels = None
    if m["labels"] == "fixed_human":
        fixed_labels = torch.as_tensor(bundle.fine_labels, device=device)
    elif m["labels"] == "fixed_cluster":
        cl = np.load(cfg.cluster_labels)
        assert len(cl) == N, f"cluster labels ({len(cl)}) do not match train split ({N})"
        fixed_labels = torch.as_tensor(cl.astype(np.int64), device=device)
    rand_u = None
    if m["weights"] == "random":
        rng = np.random.RandomState(1000 + cfg.seed)
        rand_u = torch.as_tensor(rng.rand(N), dtype=torch.float32, device=device)

    def aux_targets(x, y, idx, grad: bool):
        """returns (soft/one-hot aux labels, scaled weights or None, raw u or None)"""
        labels = u = None
        if gen is not None:
            with torch.set_grad_enabled(grad):
                labels, u = gen(x, y, need_labels=(m["labels"] == "learned"))
        if fixed_labels is not None:
            labels = torch.nn.functional.one_hot(fixed_labels[idx], n_aux).float()
        if m["weights"] == "random":
            u = rand_u[idx]
        if u is None:
            return labels, None, None
        if m["weights"] == "shuffled":
            u = u[torch.randperm(u.shape[0], device=u.device)]
        return labels, scale(u, cfg.r), u

    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump(dict(asdict(cfg), n_aux=n_aux, **bundle.meta,
                       argv=sys.argv, cuda=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                       n_params_primary=sum(p.numel() for p in primary.parameters()),
                       n_params_gen=0 if gen is None else sum(p.numel() for p in gen.parameters())), f, indent=1)
    np.save(os.path.join(cfg.out, "noise_mask.npy"), bundle.noise_mask) if bundle.noise_mask.any() else None

    E = cfg.epochs
    dump_at = set()
    for t in cfg.dump_epochs.split(","):
        dump_at.add({"last": E, "mid": E // 2}.get(t, None) or int(t))
    inner_lr = cfg.lr
    log = open(os.path.join(cfg.out, "metrics.jsonl"), "w")
    hist = []
    t_start = time.time()

    for ep in range(1, E + 1):
        if ep > 1 and (ep - 1) % cfg.step_size == 0:
            inner_lr *= cfg.gamma
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dump = ep in dump_at
        rec_idx, rec_u, rec_l1, rec_l2, rec_aux = [], [], [], [], []

        # ---------------- inner step: update theta -------------------------
        primary.train()
        if gen is not None:
            gen.train()
        t0 = time.time()
        tl = tc = tn = 0.0
        for b, (x, y, idx) in enumerate(dl_train):
            if cfg.max_batches and b >= cfg.max_batches:
                break
            x, y, idx = x.to(device, non_blocking=True), y.to(device, non_blocking=True), idx.to(device)
            p1, p2 = primary(x)
            l1 = model_fit(p1, y, device, pri=True, num_output=n_cls)
            loss = l1.mean()
            if m["labels"] != "none":
                labels, w, u = aux_targets(x, y, idx, grad=False)
                l2 = model_fit(p2, labels, device, pri=False, num_output=n_aux)
                loss = loss + (l2 * w).mean() if w is not None else loss + l2.mean()
                if dump:
                    rec_idx.append(idx.cpu()); rec_l1.append(l1.detach().cpu()); rec_l2.append(l2.detach().cpu())
                    rec_aux.append(labels.argmax(1).cpu())
                    rec_u.append((u if u is not None else torch.full_like(l1, 0.5)).detach().cpu())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += l1.sum().item(); tc += (p1.argmax(1) == y).sum().item(); tn += y.numel()
        t_inner = time.time() - t0

        # ---------------- outer step: update the generator -----------------
        t0 = time.time()
        meta_before = meta_after = 0.0; mb = 0
        trains_gen = gen is not None and any(p.requires_grad for p in gen.parameters()) and \
            not (m["labels"] != "learned" and m["weight_mode"] == "none")
        if trains_gen:
            for b, (x, y, idx) in enumerate(dl_train):
                if cfg.max_batches and b >= cfg.max_batches:
                    break
                x, y, idx = x.to(device, non_blocking=True), y.to(device, non_blocking=True), idx.to(device)
                p1, p2 = primary(x)
                labels, w, _ = aux_targets(x, y, idx, grad=True)
                l1 = model_fit(p1, y, device, pri=True, num_output=n_cls)
                l2 = model_fit(p2, labels, device, pri=False, num_output=n_aux)
                loss = l1.mean() + ((l2 * w).mean() if w is not None else l2.mean())
                fast = inner_sgd_update(primary, loss, inner_lr)
                p1f, _ = primary(x, fast)
                l1f = model_fit(p1f, y, device, pri=True, num_output=n_cls).mean()
                meta_loss = l1f
                if m["labels"] == "learned":
                    meta_loss = meta_loss + cfg.entropy * model_entropy(labels)
                gen_opt.zero_grad(set_to_none=True)
                opt.zero_grad(set_to_none=True)
                meta_loss.backward()
                gen_opt.step()
                opt.zero_grad(set_to_none=True)
                meta_before += l1.mean().item(); meta_after += l1f.item(); mb += 1
        t_meta = time.time() - t0

        # ---------------- evaluation ----------------------------------------
        t0 = time.time()
        val_acc, val_loss = evaluate(primary, dl_val, device, n_cls, cfg.max_batches)
        test_acc, test_loss = evaluate(primary, dl_test, device, n_cls, cfg.max_batches)
        t_eval = time.time() - t0

        sched.step()
        if gen_sched is not None:
            gen_sched.step()

        rec = dict(epoch=ep, train_loss=tl / max(tn, 1), train_acc=tc / max(tn, 1),
                   val_acc=val_acc, val_loss=val_loss, test_acc=test_acc, test_loss=test_loss,
                   meta_loss_before=meta_before / max(mb, 1), meta_loss_after=meta_after / max(mb, 1),
                   t_inner=t_inner, t_meta=t_meta, t_eval=t_eval,
                   peak_mem_mb=(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0,
                   lr=opt.param_groups[0]["lr"])
        if dump and rec_u:
            U = torch.cat(rec_u); W = scale(U, cfg.r)
            rec.update(w_mean=W.mean().item(), w_std=W.std().item(), w_min=W.min().item(), w_max=W.max().item())
            np.savez_compressed(os.path.join(cfg.out, f"dump_epoch{ep:03d}.npz"),
                                idx=torch.cat(rec_idx).numpy().astype(np.int32),
                                u=U.numpy().astype(np.float16),
                                loss_pri=torch.cat(rec_l1).numpy().astype(np.float16),
                                loss_aux=torch.cat(rec_l2).numpy().astype(np.float16),
                                aux_label=torch.cat(rec_aux).numpy().astype(np.int16))
        hist.append(rec)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{cfg.method} s{cfg.seed}] ep {ep:3d} train {rec['train_acc']:.4f} val {val_acc:.4f} "
              f"test {test_acc:.4f} | {t_inner:.0f}s+{t_meta:.0f}s+{t_eval:.0f}s {rec['peak_mem_mb']:.0f}MB",
              flush=True)

    log.close()
    best = max(hist, key=lambda r: (r["val_acc"], -r["epoch"]))
    last5 = hist[-5:]
    summary = dict(
        test_at_best_val=best["test_acc"], best_val=best["val_acc"], best_val_epoch=best["epoch"],
        test_final=hist[-1]["test_acc"], test_last5_mean=float(np.mean([r["test_acc"] for r in last5])),
        test_max_DO_NOT_REPORT=max(r["test_acc"] for r in hist),
        sec_per_epoch=float(np.mean([r["t_inner"] + r["t_meta"] for r in hist])),
        sec_per_epoch_inner=float(np.mean([r["t_inner"] for r in hist])),
        sec_per_epoch_meta=float(np.mean([r["t_meta"] for r in hist])),
        peak_mem_mb=max(r["peak_mem_mb"] for r in hist),
        wall_clock_s=time.time() - t_start, epochs=E)
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    open(os.path.join(cfg.out, "DONE"), "w").close()
    print("summary:", json.dumps(summary))


def parse(argv=None) -> Config:
    ap = argparse.ArgumentParser()
    for fld in fields(Config):
        t = fld.type if not isinstance(fld.type, str) else eval(fld.type)
        if t is bool:
            ap.add_argument(f"--{fld.name.replace('_', '-')}", dest=fld.name, type=lambda s: s.lower() in ("1", "true", "yes"),
                            default=fld.default)
        else:
            ap.add_argument(f"--{fld.name.replace('_', '-')}", dest=fld.name, type=t, default=fld.default)
    return Config(**vars(ap.parse_args(argv)))


if __name__ == "__main__":
    run(parse())
