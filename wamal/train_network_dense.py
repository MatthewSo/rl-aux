# wamal/train_network_dense.py

from __future__ import annotations
from typing import Optional, Tuple
import os
import pickle
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from train.model.performance import EpochPerformance
from utils.log import log_print
from wamal.networks.utils import model_fit, model_entropy  # reuse your focal-like loss + entropy. :contentReference[oaicite:9]{index=9}


def inner_sgd_update(model: torch.nn.Module, loss: torch.Tensor, lr: float):
    # identical logic to your dense-agnostic implementation in train_wamal_network. :contentReference[oaicite:10]{index=10}
    fast = {n: p for n, p in model.named_parameters()}
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return {n: w - lr * g for (n, w), g in zip(fast.items(), grads)}


def _flatten_valid(pr: torch.Tensor,
                   tgt: torch.Tensor,
                   ignore_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pr  : [B, C, H, W] probabilities (softmaxed)
    tgt : [B, H, W] long labels OR [B, C, H, W] soft labels
    returns flattened (N, C) and labels as (N,) or (N, C) with ignore filtered
    """
    B, C, H, W = pr.shape
    if tgt.dim() == 3:  # hard labels
        valid = (tgt != ignore_index)
        if valid.sum() == 0:
            # avoid empty batch
            return pr.new_zeros((0, C)), tgt.new_zeros((0,), dtype=torch.long)
        pr_f = pr.permute(0, 2, 3, 1)[valid]     # (N, C)
        y_f  = tgt[valid]                        # (N,)
        return pr_f, y_f
    else:              # soft labels: [B, C, H, W]
        # assume soft labels already zeroed-out at ignored pixels (user-controlled if needed)
        pr_f = pr.permute(0, 2, 3, 1).reshape(-1, C)
        y_f  = tgt.permute(0, 2, 3, 1).reshape(-1, C)
        return pr_f, y_f


def _pixel_accuracy(pred_softmax: torch.Tensor, y: torch.Tensor, ignore_index: int) -> float:
    """
    pred_softmax: [B, C, H, W], y: [B, H, W]
    """
    pred = pred_softmax.argmax(dim=1)
    if pred.shape[-2:] != y.shape[-2:]:
        y = F.interpolate(y.unsqueeze(1).float(), size=pred.shape[-2:], mode="nearest").long().squeeze(1)
    valid = (y != ignore_index)
    total = valid.sum().item()
    if total == 0:
        return 0.0
    correct = (pred[valid] == y[valid]).sum().item()
    return correct / total


def train_network_dense(
        device: torch.device,
        dataloader_train: DataLoader,
        dataloader_test:  DataLoader,
        total_epoch: int,
        batch_size: int,
        model: torch.nn.Module,             # WamalDenseWrapper (θ)
        label_network: torch.nn.Module,     # LabelWeightDenseWrapper (φ)
        optimizer: torch.optim.Optimizer,   # for θ
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gen_optimizer: torch.optim.Optimizer,      # for φ
        gen_scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_primary_classes: int,
        num_auxiliary_classes: int,
        save_path: str,
        use_learned_weights: bool,
        model_lr: float,
        val_range: float,
        use_auxiliary_set: bool,
        aux_split: float,
        skip_mal: bool = False,
        normalize_batch_weights: bool = False,
        batch_frac: Optional[float] = None,
        ignore_index: int = 255,
        entropy_loss_factor: float = 0.2,
):
    """
    Dense WAMAL training loop (segmentation).
    Mirrors the classification training (wamal) but over pixels.
    - Primary loss: focal-like on primary per-pixel probabilities (via model_fit).
    - Auxiliary loss: focal-like against generated soft labels per pixel (via model_fit).
    - Meta step: inner SGD update for model parameters, outer step for label_network with entropy reg.
    References: your wamal classification trainer and utils. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
    """
    os.makedirs(save_path, exist_ok=True)

    if use_auxiliary_set:
        src_collate = getattr(dataloader_train, "collate_fn", None)
        full_ds = dataloader_train.dataset
        aux_len = int(aux_split * len(full_ds))
        train_len = len(full_ds) - aux_len
        train_ds, aux_ds = torch.utils.data.random_split(
            full_ds, [train_len, aux_len],
            generator=torch.Generator().manual_seed(42))

        dataloader_train = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0),
            **({"collate_fn": src_collate} if src_collate is not None else {})
        )
        dataloader_aux = torch.utils.data.DataLoader(
            aux_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0),
            **({"collate_fn": src_collate} if src_collate is not None else {})
        )
    else:
        dataloader_aux = dataloader_train

    train_batch = len(dataloader_train)
    aux_batch   = len(dataloader_aux)
    test_batch  = len(dataloader_test)

    epoch_performances = []
    best_acc = 0.0

    for epoch in range(total_epoch):
        eff_train_batches = train_batch
        eff_aux_batches   = aux_batch
        if batch_frac is not None:
            eff_train_batches = max(1, ceil(train_batch * batch_frac))
            eff_aux_batches   = max(1, ceil(aux_batch   * batch_frac))

        if (epoch + 1) % 50 == 0:
            model_lr *= 0.5

        model.train()
        label_network.train()

        # ---------- PRIMARY STEP (θ) ----------
        train_iter = iter(dataloader_train)
        running_pri_loss = 0.0
        running_acc      = 0.0

        for _ in range(eff_train_batches):
            try:
                imgs, y = next(train_iter)   # imgs: [B,3,H,W], y: [B,H,W]
            except StopIteration:
                train_iter = iter(dataloader_train)
                imgs, y = next(train_iter)

            imgs = imgs.to(device)
            y    = y.long().to(device)

            optimizer.zero_grad()

            pri_probs, aux_probs = model(imgs)                       # softmax already applied
            gen_labels, aux_w    = label_network(imgs, y)            # per-pixel soft labels + weights

            # flatten valid pixels
            pri_flat, y_flat = _flatten_valid(pri_probs, y, ignore_index)
            aux_flat, gen_flat = _flatten_valid(aux_probs, gen_labels, ignore_index)

            if pri_flat.numel() == 0:
                # batch contained only ignore pixels; skip to avoid NaNs
                continue

            pri_loss_vec = model_fit(pri_flat, y_flat, device, pri=True,  num_output=num_primary_classes)
            aux_loss_vec = model_fit(aux_flat, gen_flat, device, pri=False, num_output=num_auxiliary_classes)

            if use_learned_weights:
                # per-pixel weights -> broadcast to flattened positions
                # bring weights to input shape of aux_probs already handled in label_network
                w = aux_w
                if w.shape[-2:] != aux_probs.shape[-2:]:
                    w = F.interpolate(w, size=aux_probs.shape[-2:], mode="bilinear", align_corners=False)
                w_flat = w.permute(0, 2, 3, 1).reshape(-1, 1)
                # keep only positions consistent with aux_flat/gen_flat
                # (when using ignore_index, they remain aligned because we didn't drop positions for soft labels)
                weight_factors = torch.pow(2.0, 2 * val_range * w_flat - val_range)
                if normalize_batch_weights and weight_factors.numel() > 0:
                    weight_factors = weight_factors / (weight_factors.mean() + 1e-12)
                aux_loss = (aux_loss_vec * weight_factors.squeeze(1)) .mean()
            else:
                aux_loss = aux_loss_vec.mean()

            if skip_mal:
                loss = pri_loss_vec.mean()
            else:
                loss = pri_loss_vec.mean() + aux_loss

            loss.backward()
            optimizer.step()

            running_pri_loss += pri_loss_vec.mean().item() / eff_train_batches
            running_acc      += _pixel_accuracy(pri_probs.detach(), y, ignore_index) / eff_train_batches

        # ---------- META STEP (φ) ----------
        if not skip_mal:
            aux_iter = iter(dataloader_aux)
            for _ in range(eff_aux_batches):
                try:
                    imgs_aux, y_aux = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(dataloader_aux)
                    imgs_aux, y_aux = next(aux_iter)

                imgs_aux = imgs_aux.to(device)
                y_aux    = y_aux.long().to(device)

                optimizer.zero_grad()
                gen_optimizer.zero_grad()

                pri_probs, aux_probs = model(imgs_aux)
                gen_labels, aux_w    = label_network(imgs_aux, y_aux)

                # flatten
                pri_flat, y_flat = _flatten_valid(pri_probs, y_aux, ignore_index)
                aux_flat, gen_flat = _flatten_valid(aux_probs, gen_labels, ignore_index)
                if pri_flat.numel() == 0:
                    continue

                pri_loss_vec = model_fit(pri_flat, y_flat, device, pri=True,  num_output=num_primary_classes)
                aux_loss_vec = model_fit(aux_flat, gen_flat, device, pri=False, num_output=num_auxiliary_classes)

                if use_learned_weights:
                    w = aux_w
                    if w.shape[-2:] != aux_probs.shape[-2:]:
                        w = F.interpolate(w, size=aux_probs.shape[-2:], mode="bilinear", align_corners=False)
                    w_flat = w.permute(0, 2, 3, 1).reshape(-1, 1)
                    weight_factors = torch.pow(2.0, 2 * val_range * w_flat - val_range)
                    if normalize_batch_weights and weight_factors.numel() > 0:
                        weight_factors = weight_factors / (weight_factors.mean() + 1e-12)
                    aux_loss = (aux_loss_vec * weight_factors.squeeze(1)).mean()
                else:
                    aux_loss = aux_loss_vec.mean()

                joint_loss = pri_loss_vec.mean() + aux_loss

                # inner step to get fast weights (θ⁺)
                fast_weights = inner_sgd_update(model, joint_loss, model_lr)

                # evaluate primary loss after one model step, keep graph for φ
                pri_probs_fast, _ = model.forward(imgs_aux, params=fast_weights)
                pri_fast_flat, y_fast_flat = _flatten_valid(pri_probs_fast, y_aux, ignore_index)
                if pri_fast_flat.numel() == 0:
                    continue

                pri_loss_fast = model_fit(pri_fast_flat, y_fast_flat, device, pri=True, num_output=num_primary_classes)

                # encourage informative auxiliary partitioning with entropy reg (over all pixels)
                # flatten gen_labels to [N, CK] and reuse your model_entropy
                gen_flat_all = gen_labels.permute(0, 2, 3, 1).reshape(-1, num_auxiliary_classes)  # (N, CK)
                ent_loss = model_entropy(gen_flat_all)

                (pri_loss_fast.mean() + entropy_loss_factor * ent_loss).backward()
                gen_optimizer.step()

        # ---------- EVAL ----------
        model.eval()
        label_network.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_acc  = 0.0
            it = iter(dataloader_test)
            for _ in range(test_batch):
                imgs, y = next(it)
                imgs = imgs.to(device)
                y    = y.long().to(device)

                pri_probs, _ = model(imgs)
                pri_flat, y_flat = _flatten_valid(pri_probs, y, ignore_index)
                if pri_flat.numel() > 0:
                    loss_vec = model_fit(pri_flat, y_flat, device, pri=True, num_output=num_primary_classes)
                    test_loss += loss_vec.mean().item() / test_batch

                test_acc  += _pixel_accuracy(pri_probs, y, ignore_index) / test_batch

        scheduler.step()
        gen_scheduler.step()

        # save best (by pixel accuracy)
        if test_acc > best_acc:
            best_acc = test_acc
            model.save(os.path.join(save_path, "best_primary_model"))
            label_network.save(os.path.join(save_path, "best_label_model"))

        perf = EpochPerformance(
            epoch=epoch,
            train_loss_primary=running_pri_loss,
            train_loss_auxiliary=0.0,
            train_accuracy_primary=running_acc,
            train_accuracy_auxiliary=0.0,
            test_loss_primary=test_loss,
            test_loss_auxiliary=0.0,
            test_accuracy_primary=test_acc,
            test_accuracy_auxiliary=0.0,
            batch_id=eff_train_batches * (epoch + 1),
        )
        epoch_performances.append(perf)
        with open(os.path.join(save_path, "epoch_performances.pkl"), "wb") as f:
            pickle.dump(epoch_performances, f)

        log_print(perf)
        log_print(
            f"EPOCH {epoch:03d} | TrainLoss {running_pri_loss:.4f} "
            f"TrainPixAcc {running_acc:.4f} | TestLoss {test_loss:.4f} "
            f"TestPixAcc {test_acc:.4f}"
        )
