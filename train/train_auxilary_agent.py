import numpy as np
import torch

from train.model.performance import EpochPerformance
from utils.vars import softmax


def train_auxilary_agent(primary_model, aux_task_model, device, env, test_loader, batch_size, total_epochs):
    epoch_performances = []
    epoch_performance = None
    num_test_batches = len(test_loader)

    for index in range(total_epochs):
        primary_model.train()
        env.train_label_network_with_rl(aux_task_model)
        env.train_main_network(aux_task_model)

        primary_model=env.cannonical_model
        primary_model.eval()

        with torch.no_grad():
            test_dataset = iter(test_loader)
            for i in range(num_test_batches):
                test_data, test_label =  next(test_dataset)
                test_label = test_label.type(torch.LongTensor)
                test_data, test_label = test_data.to(device), test_label.to(device)

                test_primary_pred, test_aux_pred = primary_model(test_data)
                test_primary_pred, test_aux_pred = softmax(test_primary_pred),softmax(test_aux_pred)

                test_loss1  = primary_model.model_fit(test_primary_pred, test_label, pri=True,num_output=20)

                test_predict_label1 = test_primary_pred.data.max(1)[1]

                test_acc1 = test_predict_label1.eq(test_label).sum().item() / batch_size

                epoch_performance = EpochPerformance(
                    epoch=index,
                    train_loss_primary=0,
                    train_loss_auxiliary=0,
                    train_accuracy_primary=0,
                    train_accuracy_auxiliary=0,
                    test_loss_primary=test_loss1.mean().item(),
                    test_loss_auxiliary=0,
                    test_accuracy_primary=test_acc1,
                    test_accuracy_auxiliary=0,
                )
                epoch_performances.append(epoch_performance)

        print(epoch_performance)