import json
import os

from utils.log import log_print


def create_path_name(agent_type, learn_weights, primary_model_type, train_ratio, aux_weight, observation_feature_dimensions, dataset, optimizer, full_dataset, learning_rate, range, aux_set_ratio, normalize_batch, batch_fraction):
    return f"./trained_models/{agent_type}_{primary_model_type}_learn_weights_{learn_weights}_train_ratio_{train_ratio}_aux_weight_{aux_weight}_obs_dim_{observation_feature_dimensions}_{dataset}_optimizer_{optimizer}_fulldataset_{full_dataset}_lr_{learning_rate}_range_{range}_auxiliary_set_ratio_{aux_set_ratio}_normalize_batch_{normalize_batch}_batch_fraction_{batch_fraction}"


def save_all_parameters(
        batch_size,
        aux_dimensions,
        primary_dimensions,
        total_epoch,
        primary_learning_rate,
        rl_learning_rate,
        scheduler_step_size,
        scheduler_gamma,
        aux_weight,
        train_ratio,
        save_path,
        dataset,
        model_name,
        agent_type,
        observation_feature_dimensions,
        aux_task_type,
        learn_weights,
        primary_task_type,
        git_commit_hash,
):
    # save as loadable json
    # create path if it doesnt exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    parameters = {
        "batch_size": batch_size,
        "aux_dimensions": aux_dimensions,
        "primary_dimensions": primary_dimensions,
        "total_epoch": total_epoch,
        "primary_learning_rate": primary_learning_rate,
        "rl_learning_rate": rl_learning_rate,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "aux_weight": aux_weight,
        "train_ratio": train_ratio,
        "save_path": save_path,
        "dataset": dataset,
        "learn_weights": learn_weights,
        "model_name": model_name,
        "agent_type": agent_type,
        "observation_feature_dimensions": observation_feature_dimensions,
        "aux_task_type": aux_task_type,
        "primary_task_type": primary_task_type,
        "git_commit_hash": git_commit_hash,
    }
    # Save the parameters to a JSON file
    with open(f"{save_path}/parameters.json", "w") as f:
        json.dump(parameters, f, indent=4)
    log_print(f"Parameters saved to {save_path}/parameters.json")


def save_parameter_dict(parameter_dict):
    # create path if it doesnt exist
    save_path = parameter_dict["save_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the parameters to a JSON file
    with open(f"{save_path}/parameters.json", "w") as f:
        json.dump(parameter_dict, f, indent=4)
    log_print(f"Parameters saved to {save_path}/parameters.json")