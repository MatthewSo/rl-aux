import pickle
from collections import OrderedDict
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from train.model.performance import EpochPerformance
from utils.log import log_print
from wamal.networks.utils import model_fit, model_entropy


def inner_sgd_update(model, loss, lr):
    fast = OrderedDict(model.named_parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return OrderedDict((n, w - lr * g) for (n, w), g in zip(fast.items(), grads))


def train_wamal_network(device, dataloader_train, dataloader_test,
                         total_epoch, train_batch, test_batch, batch_size,
                         model, label_network, optimizer, scheduler,
                         gen_optimizer, gen_scheduler,
                         num_axuiliary_classes, num_primary_classes,
                         save_path, use_learned_weights, model_lr,normalize_batch_weights, batch_frac,
                         val_range, use_auxiliary_set, aux_split, skip_mal=False,
):

    epoch_performances = []
    avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
    best_training_performance = 0
    k=0

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
    test_batch = len(dataloader_test)
    aux_batch = len(dataloader_aux)

    for index in range(total_epoch):
        cost = np.zeros(4, dtype=np.float32)
        eff_train_batches = train_batch
        eff_aux_batches = aux_batch
        if batch_frac is not None:
            eff_train_batches = max(1, int(np.ceil(train_batch * batch_frac)))
            eff_aux_batches = max(1, int(np.ceil(aux_batch * batch_frac)))

        # drop the learning rate with the same strategy in the multi-task network
        # note: not necessary to be consistent with the multi-task network's parameter,
        # it can also be learned directly from the network
        if (index + 1) % 50 == 0:
            model_lr = model_lr * 0.5

        # evaluate training data (training-step, update on theta_1)
        model.train()
        train_iter = iter(dataloader_train)
        for i in range(eff_train_batches):
            train_data, train_label = next(train_iter)
            train_label = train_label.type(torch.LongTensor)
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_pred1, train_pred2 = model(train_data)
            train_pred3, aux_weight = label_network(train_data, train_label)  # generate auxiliary labels

            # reset optimizers with zero gradient
            optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # choose level 2/3 hierarchy, 20-class (gt) / 100-class classification (generated by labelgeneartor)
            train_loss1 = model_fit(train_pred1, train_label, device, pri=True, num_output=num_primary_classes)
            train_loss2 = model_fit(train_pred2, train_pred3, device, pri=False, num_output=num_axuiliary_classes)

            # compute cosine similarity between gradients from primary and auxiliary loss
            grads1 = torch.autograd.grad(torch.mean(train_loss1), model.parameters(), retain_graph=True, allow_unused=True)
            grads2 = torch.autograd.grad(torch.mean(train_loss2), model.parameters(), retain_graph=True, allow_unused=True)
            cos_mean = 0
            for l in range(len(grads1) - 8):  # only compute on shared representation (ignore task-specific fc-layers)
                grads1_ = grads1[l].view(grads1[l].shape[0], -1)
                grads2_ = grads2[l].view(grads2[l].shape[0], -1)
                cos_mean += torch.mean(F.cosine_similarity(grads1_, grads2_, dim=-1)) / (len(grads1) - 8)
            # cosine similarity evaluation ends here

            if use_learned_weights:
                weight_factors = torch.pow(2.0, ( 2* val_range * aux_weight) - val_range)
                if normalize_batch_weights:
                    weight_factors = weight_factors / weight_factors.mean()
                aux_loss = torch.mean(train_loss2 * weight_factors)

            else:
                aux_loss = torch.mean(train_loss2)

            if skip_mal:
                train_loss = torch.mean(train_loss1)
            else:
                train_loss = torch.mean(train_loss1) + aux_loss

            train_loss.backward()

            optimizer.step()

            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label).sum().item() / batch_size

            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1
            cost[2] = cos_mean
            k = k + 1
            avg_cost[index][0:3] += cost[0:3] / eff_train_batches

        # evaluating training data (meta-training step, update on theta_2)
        aux_iter = iter(dataloader_aux)
        for i in range(eff_aux_batches):
            if skip_mal:
                continue
            train_data, train_label = next(aux_iter)
            train_label = train_label.type(torch.LongTensor)
            train_data, train_label = train_data.to(device), train_label.to(device)

            train_pred1, train_pred2 = model(train_data)
            train_pred3, aux_weight = label_network(train_data, train_label)

            # reset optimizer with zero gradient
            optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # choose level 2/3 hierarchy, 20-class/100-class classification
            train_loss1 = model_fit(train_pred1, train_label, device, pri=True, num_output=num_primary_classes)
            train_loss2 = model_fit(train_pred2, train_pred3, device, pri=False, num_output=num_axuiliary_classes)
            train_loss3 = model_entropy(train_pred3)

            # multi-task loss
            # element-wise multiplication between auxiliary loss and auxiliary weight

            if use_learned_weights:
                weight_factors = torch.pow(2.0, (val_range * 2) * aux_weight - val_range)
                if normalize_batch_weights:
                    weight_factors = weight_factors / weight_factors.mean()
                aux_loss = torch.mean(train_loss2 * weight_factors)

            else:
                aux_loss = torch.mean(train_loss2)

            train_loss = torch.mean(train_loss1) + aux_loss

            # current accuracy on primary task
            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label).sum().item() / batch_size
            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1

            # # current theta_1
            # fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())
            #
            # # create_graph flag for computing second-derivative
            # grads = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
            # data = [p.data for p in list(model.parameters())]
            #
            # # compute theta_1^+ by applying sgd on multi-task loss
            # fast_weights = OrderedDict((name, param - vgg_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
            fast_weights = inner_sgd_update(model, train_loss, model_lr)


            # compute primary loss with the updated thetat_1^+
            train_pred1, train_pred2 = model.forward(train_data, fast_weights)
            train_loss1 = model_fit(train_pred1, train_label, device, pri=True, num_output=num_primary_classes)

            # update theta_2 with primary loss + entropy loss
            (torch.mean(train_loss1) + 0.2 * torch.mean(train_loss3)).backward()
            gen_optimizer.step()

            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label).sum().item() / batch_size

            # accuracy on primary task after one update
            cost[2] = torch.mean(train_loss1).item()
            cost[3] = train_acc1
            avg_cost[index][3:7] += cost[0:4] / eff_train_batches

        # evaluate on test data
        model.eval()
        with torch.no_grad():
            cifar100_test_dataset = iter(dataloader_test)
            for i in range(test_batch):
                test_data, test_label = next(cifar100_test_dataset)
                test_label = test_label.type(torch.LongTensor)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_pred1, test_pred2 = model(test_data)

                test_loss1 = model_fit(test_pred1, test_label, device, pri=True, num_output=num_primary_classes)

                test_predict_label1 = test_pred1.data.max(1)[1]
                test_acc1 = test_predict_label1.eq(test_label).sum().item() / batch_size

                cost[0] = torch.mean(test_loss1).item()
                cost[1] = test_acc1

                avg_cost[index][7:] += cost[0:2] / test_batch

        scheduler.step()
        gen_scheduler.step()

        epoch_performance = EpochPerformance(
            epoch=index,
            train_loss_primary=avg_cost[index][0],
            train_loss_auxiliary=0,
            train_accuracy_primary=avg_cost[index][1],
            train_accuracy_auxiliary=0,
            test_loss_primary=avg_cost[index][7],
            test_loss_auxiliary=0,
            test_accuracy_primary=avg_cost[index][8],
            test_accuracy_auxiliary=0,
            batch_id=eff_train_batches * (index + 1),
        )
        epoch_performances.append(epoch_performance)

        log_print(epoch_performance)

        test_accuracy_primary = avg_cost[index][8]
        if test_accuracy_primary > best_training_performance:
            best_training_performance = test_accuracy_primary
            log_print(f"Best training performance so far: {best_training_performance}")
            model.save(save_path + '/best_primary_model')
            label_network.save(save_path + '/best_label_model')

        # save epoch performances as pickle. has no .save
        with open(save_path + '/epoch_performances.pkl', 'wb') as f:
            pickle.dump(epoch_performances, f)

        log_print('EPOCH: {:04d} Iter {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} COSSIM {:.4f} || '
                  'META [LOSS|ACC.]: PRE {:.4f} {:.4f} AFTER {:.4f} {:.4f} || TEST: {:.4f} {:.4f}'
                  .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                          avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8]))