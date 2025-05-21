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
from wamal.networks.utils import model_fit


def train_auxilearn_network(
        device,
        dataloader_train,
        dataloader_test,
        total_epoch,
        train_batch,                  # kept for signature; overwritten below
        test_batch,                   # idem
        batch_size,
        model,                        # primary network θ
        label_network,                # auxiliary network φ
        optimizer,                    # optimiser for θ
        scheduler,
        gen_optimizer,                # base optimiser for φ
        gen_scheduler,
        num_axuiliary_classes,
        num_primary_classes,
        save_path,
        use_learned_weights,
        model_lr,
        val_range,
        use_auxiliary_set,
        aux_split,
        skip_mal=False,
        normalize_batch_weights=False,
        batch_frac=None
):
    if use_auxiliary_set:
        src_collate = getattr(dataloader_train, "collate_fn", None)
        full_ds  = dataloader_train.dataset
        aux_len  = int(aux_split * len(full_ds))
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
            aux_ds,  batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0),
            **({"collate_fn": src_collate} if src_collate is not None else {})
        )
    else:
        dataloader_aux = dataloader_train

    train_batch = len(dataloader_train)
    aux_batch   = len(dataloader_aux)
    test_batch  = len(dataloader_test)

    aux_optimizer = MetaOptimizer(
        gen_optimizer,
        hpo_lr=gen_optimizer.param_groups[0]["lr"],
    )

    def batch_losses(x, y):
        y   = y.long().to(device)
        x   = x.to(device)

        pri_logits, aux_logits       = model(x)
        gen_labels, aux_weights_raw  = label_network(x, y)

        pri_loss = model_fit(
            pri_logits, y, device,
            pri=True, num_output=num_primary_classes).mean()

        aux_raw  = model_fit(
            aux_logits, gen_labels, device,
            pri=False, num_output=num_axuiliary_classes)

        if use_learned_weights:
            w = torch.pow(2.0, 2 * val_range * aux_weights_raw - val_range)
            if normalize_batch_weights:
                w = w / w.mean()
            aux_loss = (aux_raw * w).mean()
        else:
            aux_loss = aux_raw.mean()

        joint_loss = pri_loss if skip_mal else pri_loss + aux_loss
        return joint_loss, pri_loss, aux_loss

    os.makedirs(save_path, exist_ok=True)
    avg_cost   = np.zeros([total_epoch, 9], dtype=np.float32)
    best_acc   = 0.0
    epoch_performances = []
    global_iter = 0

    for epoch in range(total_epoch):

        # effective number of batches this epoch
        eff_train_batches = train_batch
        eff_aux_batches   = aux_batch

        if batch_frac is not None:
            eff_train_batches = max(1, ceil(train_batch * batch_frac))
            eff_aux_batches   = max(1, ceil(aux_batch   * batch_frac))

        # LR drop every 50 epochs (matches original code)
        if (epoch + 1) % 50 == 0:
            model_lr *= 0.5

        model.train()
        train_iter = iter(dataloader_train)
        cost_epoch = np.zeros(4, dtype=np.float32)

        for _ in range(eff_train_batches):
            try:
                x_tr, y_tr = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader_train)
                x_tr, y_tr = next(train_iter)

            optimizer.zero_grad()
            joint_loss, pri_loss, _ = batch_losses(x_tr, y_tr)
            joint_loss.backward()
            optimizer.step()

            global_iter += 1
            pred = model(x_tr.to(device))[0].argmax(dim=1)
            acc  = (pred == y_tr.to(device)).float().mean().item()

            cost_epoch[0] += pri_loss.item() / eff_train_batches
            cost_epoch[1] += acc           / eff_train_batches

        if not skip_mal:
            aux_iter = iter(dataloader_aux)
            for _ in range(eff_aux_batches):
                try:
                    x_aux, y_aux = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(dataloader_aux)
                    x_aux, y_aux = next(aux_iter)

                # lower-level loss on a *fresh* train batch
                try:
                    x_tr2, y_tr2 = next(train_iter)
                except StopIteration:
                    train_iter = iter(dataloader_train)
                    x_tr2, y_tr2 = next(train_iter)

                train_loss_meta, _, _ = batch_losses(x_tr2, y_tr2)
                val_loss_meta,   _, _ = batch_losses(x_aux, y_aux)

                aux_optimizer.step(
                    val_loss   = val_loss_meta,
                    train_loss = train_loss_meta,
                    aux_params = list(label_network.parameters()),
                    parameters = list(model.parameters()),
                )

        avg_cost[epoch, 0] = cost_epoch[0]
        avg_cost[epoch, 1] = cost_epoch[1]

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_acc  = 0.0
            test_iter = iter(dataloader_test)
            for _ in range(test_batch):
                x_te, y_te = next(test_iter)
                y_te = y_te.long().to(device)
                x_te = x_te.to(device)

                logits, _ = model(x_te)
                loss_te   = model_fit(
                    logits, y_te, device,
                    pri=True, num_output=num_primary_classes).mean()

                pred_te  = logits.argmax(dim=1)
                acc_te   = (pred_te == y_te).float().mean().item()

                test_loss += loss_te.item() / test_batch
                test_acc  += acc_te       / test_batch

            avg_cost[epoch, 7:] = [test_loss, test_acc]

        scheduler.step()
        gen_scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            model.save(os.path.join(save_path, "best_primary_model"))
            label_network.save(os.path.join(save_path, "best_label_model"))

        epoch_performances.append(
            EpochPerformance(
                epoch                 = epoch,
                train_loss_primary    = avg_cost[epoch, 0],
                train_loss_auxiliary  = 0,
                train_accuracy_primary= avg_cost[epoch, 1],
                train_accuracy_auxiliary = 0,
                test_loss_primary     = avg_cost[epoch, 7],
                test_loss_auxiliary   = 0,
                test_accuracy_primary = avg_cost[epoch, 8],
                test_accuracy_auxiliary = 0,
                batch_id              = eff_train_batches * (epoch + 1),
            )
        )

        with open(os.path.join(save_path, "epoch_performances.pkl"), "wb") as f:
            pickle.dump(epoch_performances, f)

        log_print(epoch_performances[-1])
        log_print(
            f"EPOCH {epoch:03d}  |  TrainLoss {avg_cost[epoch,0]:.4f} "
            f"TrainAcc {avg_cost[epoch,1]:.4f}  |  TestLoss {test_loss:.4f} "
            f"TestAcc {test_acc:.4f}"
        )