import os
import pickle
from math import ceil
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

import torch
import numpy as np
import torch.nn.functional as F
from auxilearn.optim import MetaOptimizer
from collections import deque
import pickle
import os
from math import ceil

from train.model.performance import EpochPerformance
from utils.log import log_print
# from wamal.networks.utils import model_fit

def train_meta_l1_network(
        device: torch.device,
        dataloader_train: DataLoader,
        dataloader_val:   DataLoader,
        dataloader_test:  DataLoader,
        total_epoch: int,
        batch_size: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gamma_optimizer: torch.optim.Optimizer,
        gamma_scheduler,
        num_primary_classes: int,
        save_path: str,
        gamma_params,
        init_gamma: float = 0.0,
        learned_range: float = 2.0,
        aux_split: float = 0.2,
        batch_frac = None,
        skip_meta: bool = False,
        skip_regularization: bool = False,
        reg_weight: float = 1.0,
):


    if dataloader_val is None and aux_split !=0:
        src_collate = getattr(dataloader_train, "collate_fn", None)
        full_ds     = dataloader_train.dataset
        val_len     = int(aux_split * len(full_ds))
        train_len   = len(full_ds) - val_len
        train_ds, val_ds = random_split(
            full_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(42))
        dataloader_train = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0),
            **({"collate_fn": src_collate} if src_collate is not None else {}))
        dataloader_val = DataLoader(
            val_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0),
            **({"collate_fn": src_collate} if src_collate is not None else {}))

    train_batches = len(dataloader_train)
    val_batches   = len(dataloader_val)
    test_batches  = len(dataloader_test)

    meta_optimizer = MetaOptimizer(
        gamma_optimizer,
        hpo_lr=gamma_optimizer.param_groups[0]["lr"],
    )

    def _task_loss(logits: torch.Tensor, y: torch.Tensor):
        if y.dtype in (torch.long, torch.int64):
            return F.cross_entropy(logits, y, reduction="mean")
        return F.mse_loss(logits, y.float(), reduction="mean")

    def _l1_reg() -> torch.Tensor:
        reg = 0.0
        
        for g_raw, p in zip(gamma_params, model.parameters()):
            g   = F.softplus(g_raw)          # element-wise γ, same shape as p
            reg = reg + (g * p).abs().mean() # accumulate a single scalar
        return reg/len([p for p in model.parameters()])

    def batch_losses(x: torch.Tensor, y: torch.Tensor):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        task   = _task_loss(logits, y)
        reg    = _l1_reg()
        train_loss = task + reg
        # check if skip meta and skip regularization
        if skip_regularization:
            train_loss = task
        return train_loss, task, logits

    os.makedirs(save_path, exist_ok=True)
    epoch_performances: List[EpochPerformance] = []
    best_score = -np.inf

    for epoch in range(total_epoch):
        model.train()
        running = {"task": 0.0, "reg": 0.0, "acc": 0.0}

        eff_train_batches = train_batches
        eff_val_batches   = val_batches
        if batch_frac is not None:
            eff_train_batches = max(1, ceil(train_batches * batch_frac))
            eff_val_batches   = max(1, ceil(val_batches   * batch_frac))

        train_iter = iter(dataloader_train)
        val_iter   = iter(dataloader_val)

        for _ in range(eff_train_batches):
            try:
                x_tr, y_tr = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader_train)
                x_tr, y_tr = next(train_iter)

            optimizer.zero_grad()
            train_loss, task, logits = batch_losses(x_tr, y_tr)
            train_loss.backward()
            optimizer.step()

            running["task"] += task.item() / eff_train_batches
            running["reg"]  += (_l1_reg().item()) / eff_train_batches
            if y_tr.dtype in (torch.long, torch.int64):
                running["acc"] += (logits.argmax(dim=1) == y_tr.to(device)).float().mean().item() / eff_train_batches

        if not skip_meta:
            for _ in range(eff_val_batches):
                try:
                    x_val, y_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(dataloader_val)
                    x_val, y_val = next(val_iter)

                try:
                    x_tr2, y_tr2 = next(train_iter)
                except StopIteration:
                    train_iter = iter(dataloader_train)
                    x_tr2, y_tr2 = next(train_iter)

                inner_loss, _, _ = batch_losses(x_tr2, y_tr2)
                _,        val_task_loss, _ = batch_losses(x_val, y_val)

                meta_optimizer.step(
                    val_loss   = val_task_loss,
                    train_loss = inner_loss,
                    aux_params = list(gamma_params),
                    parameters = list(model.parameters()),
                )

        model.eval()
        with torch.no_grad():
            test_task = 0.0
            test_acc  = 0.0
            for x_te, y_te in dataloader_test:
                x_te, y_te = x_te.to(device), y_te.to(device)
                logits = model(x_te)
                task_loss = _task_loss(logits, y_te)
                test_task += task_loss.item() / test_batches
                if y_te.dtype in (torch.long, torch.int64):
                    test_acc += (logits.argmax(dim=1) == y_te).float().mean().item() / test_batches

        scheduler.step()
        gamma_scheduler.step()

        score = test_acc if not np.isnan(test_acc) else -test_task
        if score > best_score:
            best_score = score
            def save_model(model, file_path: str):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                torch.save(model.state_dict(), file_path)
            save_model(model,os.path.join(save_path, "best_primary_model"))
            torch.save({"gamma_raw": [g.detach().cpu() for g in gamma_params]},
                       os.path.join(save_path, "best_gamma.pt"))

        epoch_performances.append(EpochPerformance(
            epoch                   = epoch,
            train_loss_primary      = running["task"],
            train_loss_auxiliary    = running["reg"],
            train_accuracy_primary  = running["acc"],
            train_accuracy_auxiliary= 0.0,
            test_loss_primary       = test_task,
            test_loss_auxiliary     = 0.0,
            test_accuracy_primary   = test_acc,
            test_accuracy_auxiliary = 0.0,
            batch_id                = eff_train_batches * (epoch + 1),
        ))

        with open(os.path.join(save_path, "epoch_performances.pkl"), "wb") as f:
            pickle.dump(epoch_performances, f)

        #log_print(epoch_performances[-1])
        log_print(
            f"EPOCH {epoch:03d} | TaskLoss {running['task']:.4f} Reg {running['reg']:.4f} "
            f"Acc {running['acc']:.4f} | TestTask {test_task:.4f} TestAcc {test_acc:.4f}")
