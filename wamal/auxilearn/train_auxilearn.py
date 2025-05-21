import torch
import numpy as np
import torch.nn.functional as F
from auxilearn.optim import MetaOptimizer
from collections import deque
import pickle
import os

from train.model.performance import EpochPerformance
from utils.log import log_print
from wamal.networks.utils import model_fit


def train_auxilearn_network(
        device,
        dataloader_train,
        dataloader_test,
        total_epoch,
        train_batch,
        test_batch,
        batch_size,
        model,                    # ≙ primary_model
        label_network,            # ≙ auxiliary_model
        optimizer,                # ≙ primary_optimizer  (inner level)
        scheduler,
        gen_optimizer,            # base optimizer for aux-params
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
        aux_params_update_every=1,   # new: meta-step frequency
):
    aux_optimizer = MetaOptimizer(gen_optimizer, hpo_lr=gen_optimizer.param_groups[0]['lr'])
    if use_auxiliary_set:
        full_ds   = dataloader_train.dataset
        aux_len   = int(aux_split * len(full_ds))
        train_len = len(full_ds) - aux_len
        train_ds, aux_ds = torch.utils.data.random_split(
            full_ds, [train_len, aux_len],
            generator=torch.Generator().manual_seed(42))

        dataloader_train = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0))
        dataloader_aux = torch.utils.data.DataLoader(
            aux_ds,   batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=getattr(dataloader_train, "num_workers", 0))
    else:
        dataloader_aux = dataloader_train


    def batch_losses(data, targets):
        targets = targets.long().to(device)
        data    = data.to(device)

        pri_logits, aux_logits      = model(data)
        gen_labels, aux_weights_raw = label_network(data, targets)

        pri_loss = model_fit(pri_logits, targets,
                             device, pri=True,
                             num_output=num_primary_classes).mean()

        aux_raw  = model_fit(aux_logits, gen_labels,
                             device, pri=False,
                             num_output=num_axuiliary_classes)

        if use_learned_weights:
            weight_factors = torch.pow(2.0, (2 * val_range * aux_weights_raw) - val_range)
            aux_loss = (aux_raw * weight_factors).mean()
        else:
            aux_loss = aux_raw.mean()

        joint_loss = pri_loss if skip_mal else (pri_loss + aux_loss)
        return joint_loss, pri_loss, aux_loss


    os.makedirs(save_path, exist_ok=True)
    best_test_acc = 0.0
    epoch_performances = []
    k                = 0

    avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)

    for epoch in range(total_epoch):
        model.train()
        batch_iter_train = iter(dataloader_train)
        batch_iter_aux   = iter(dataloader_aux)

        cost_epoch = np.zeros(4, dtype=np.float32)
        for i in range(train_batch):
            x_tr, y_tr = next(batch_iter_train)
            optimizer.zero_grad()
            loss_joint, loss_pri, loss_aux = batch_losses(x_tr, y_tr)
            loss_joint.backward()
            optimizer.step()
            k += 1

            pred = model(x_tr.to(device))[0].data.max(1)[1]
            acc  = pred.eq(y_tr.to(device)).sum().item() / batch_size
            cost_epoch[0] += loss_pri.item() / train_batch
            cost_epoch[1] += acc           / train_batch

            if not skip_mal and k % aux_params_update_every == 0:
                try:
                    x_aux, y_aux = next(batch_iter_aux)
                except StopIteration:            # shorter aux loader
                    batch_iter_aux = iter(dataloader_aux)
                    x_aux, y_aux   = next(batch_iter_aux)

                train_loss_meta, _, _ = batch_losses(x_tr,  y_tr)   # lower-level
                val_loss_meta,   _, _ = batch_losses(x_aux, y_aux)  # upper-level

                aux_optimizer.step(
                    val_loss   = val_loss_meta,
                    train_loss = train_loss_meta,
                    aux_params = list(label_network.parameters()),  # materialise once
                    parameters = list(model.parameters()),          # materialise once
                )

        avg_cost[epoch][0] = cost_epoch[0]
        avg_cost[epoch][1] = cost_epoch[1]

        model.eval()
        with torch.no_grad():
            test_cost = np.zeros(2, dtype=np.float32)
            test_iter = iter(dataloader_test)
            for _ in range(test_batch):
                x_te, y_te = next(test_iter)
                y_te   = y_te.long().to(device)
                x_te   = x_te.to(device)
                logits, _   = model(x_te)
                loss_te     = model_fit(logits, y_te, device,
                                        pri=True,
                                        num_output=num_primary_classes).mean()
                pred_te     = logits.data.max(1)[1]
                acc_te      = pred_te.eq(y_te).sum().item() / batch_size
                test_cost[0] += loss_te.item() / test_batch
                test_cost[1] += acc_te       / test_batch

            avg_cost[epoch][7:] = test_cost


        scheduler.step()
        gen_scheduler.step()

        # best checkpoint
        if test_cost[1] > best_test_acc:
            best_test_acc = test_cost[1]
            model.save(os.path.join(save_path, "best_primary_model"))
            label_network.save(os.path.join(save_path, "best_label_model"))

        # epoch struct (replicating your EpochPerformance)
        epoch_performances.append(
            EpochPerformance(
                epoch                 = epoch,
                train_loss_primary    = avg_cost[epoch][0],
                train_loss_auxiliary  = 0,
                train_accuracy_primary= avg_cost[epoch][1],
                train_accuracy_auxiliary=0,
                test_loss_primary     = avg_cost[epoch][7],
                test_loss_auxiliary   = 0,
                test_accuracy_primary = avg_cost[epoch][8],
                test_accuracy_auxiliary=0,
            )
        )

        # persist epoch performances
        with open(os.path.join(save_path, 'epoch_performances.pkl'), 'wb') as f:
            pickle.dump(epoch_performances, f)

        # optional console log
        log_print(epoch_performances[-1])
        log_print(f'EPOCH {epoch:03d} done -- best test acc so far: {best_test_acc:.4f}')

