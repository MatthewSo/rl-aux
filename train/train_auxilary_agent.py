import torch
import pickle
from train.model.performance import EpochPerformance
from utils.log import log_print
from utils.vars import softmax

def train_auxilary_agent(primary_model, aux_task_model, device, env, test_loader, batch_size, total_epochs, save_path, primary_dimension, model_train_ratio, skip_rl):
    epoch_performances = []
    epoch_performance = None
    num_test_batches = len(test_loader)

    best_training_performance = 0

    for index in range(total_epochs):
        primary_model.train()
        log_print("Starting Epoch: ", index)
        if not skip_rl:
            log_print(f"Skipping RL: {skip_rl}")
            env.train_label_network_with_rl(aux_task_model, ratio=model_train_ratio)

        log_print("Finished Training Auxiliary Task Model")
        env.train_main_network(aux_task_model)
        log_print("Finished Training Primary Task Model")

        # Save the model
        env.save(aux_task_model)

        primary_model=env.cannonical_model
        primary_model.eval()

        with torch.no_grad():
            test_loader_iterator = iter(test_loader)
            test_epoch_loss = 0
            test_epoch_acc = 0
            for i in range(num_test_batches):
                test_data, test_label =  next(test_loader_iterator)
                test_label = test_label.type(torch.LongTensor)
                test_data, test_label = test_data.to(device), test_label.to(device)

                test_primary_pred, test_aux_pred = primary_model(test_data)
                test_primary_pred, test_aux_pred = test_primary_pred, test_aux_pred

                test_loss1  = primary_model.model_fit(test_primary_pred, test_label, device=device, pri=True,num_output=primary_dimension)

                test_predict_label1 = test_primary_pred.data.max(1)[1]

                test_acc1 = test_predict_label1.eq(test_label).sum().item() / batch_size
                test_epoch_loss += test_loss1.mean().item()
                test_epoch_acc += test_acc1

            test_loss1 = test_epoch_loss / num_test_batches
            test_acc1 = test_epoch_acc / num_test_batches

            if test_acc1 > best_training_performance:
                best_training_performance = test_acc1
                log_print(f"Best training performance so far: {best_training_performance}")
                env.save(aux_task_model, save_path + '/best_model')

            epoch_performance = EpochPerformance(
                epoch=index,
                train_loss_primary=0,
                train_loss_auxiliary=0,
                train_accuracy_primary=0,
                train_accuracy_auxiliary=0,
                test_loss_primary=test_loss1,
                test_loss_auxiliary=0,
                test_accuracy_primary=test_acc1,
                test_accuracy_auxiliary=0,
            )
            epoch_performances.append(epoch_performance)

            log_print(epoch_performance)

            # save epoch performances as pickle. has no .save
            with open(save_path + '/epoch_performances.pkl', 'wb') as f:
                pickle.dump(epoch_performances, f)
