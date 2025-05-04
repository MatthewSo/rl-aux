import subprocess
from tabnanny import verbose

import torch
from stable_baselines3 import PPO
from torch import nn

from datasets.cifar100 import CIFAR100, CoarseLabelCIFAR100
from datasets.transforms import cifar_trans_train, cifar_trans_test
from environment.learn_weight_aux_task import AuxTaskEnv
from environment.weight_training.weight_training_environment import WeightTuningEnv
from networks.ppo.ppo import get_ppo_agent
from networks.primary.vgg import VGG16
from networks.weight_training.ppo import get_weight_training_ppo_agent
from train.train_auxilary_agent import train_auxilary_agent
from utils.analysis.network_details import print_aux_weights
from utils.log import log_print, change_log_location
from utils.path_name import create_path_name, save_all_parameters

LOAD_MODEL_PATH = "/home/cml0/rl-aux/trained_models/PPO_VGG_learn_weights_False_train_ratio_1_aux_weight_1_obs_dim_256_CIFAR100-20v2/best_model_auxiliary"
BATCH_SIZE = 100
AUX_DIMENSION = 100
PRIMARY_DIMENSION = 20
OBSERVATION_FEATURE_DIMENSION = 256
TOTAL_EPOCH = 200
PRIMARY_LEARNING_RATE = 0.01
PPO_LEARNING_RATE = 1e-4
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5
AUX_WEIGHT = 0
LEARN_WEIGHTS = True
TRAIN_RATIO = 1

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

SAVE_PATH = create_path_name(
    agent_type="WEIGHT_TUNING_PPO_RESET",
    primary_model_type="VGG",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
    observation_feature_dimensions=OBSERVATION_FEATURE_DIMENSION,
    dataset="CIFAR100-20",
    learn_weights=True,
)

change_log_location(SAVE_PATH)

save_all_parameters(
    batch_size=BATCH_SIZE,
    aux_dimensions=AUX_DIMENSION,
    primary_dimensions=PRIMARY_DIMENSION,
    total_epoch=TOTAL_EPOCH,
    primary_learning_rate=PRIMARY_LEARNING_RATE,
    rl_learning_rate=PPO_LEARNING_RATE,
    scheduler_step_size=SCHEDULER_STEP_SIZE,
    scheduler_gamma=SCHEDULER_GAMMA,
    aux_weight=AUX_WEIGHT,
    train_ratio=TRAIN_RATIO,
    save_path=SAVE_PATH,
    dataset="CIFAR100-20",
    model_name="VGG",
    agent_type="PPO",
    observation_feature_dimensions=OBSERVATION_FEATURE_DIMENSION,
    aux_task_type="AuxTask",
    primary_task_type="VGG",
    git_commit_hash=git_hash,
    learn_weights=True,
)

log_print("Torch CUDA available:", torch.cuda.is_available())
log_print("Torch CUDA device count:", torch.cuda.device_count())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

log_print("Using device:", device)

# ---------

cifar100_train_set = CIFAR100(root='dataset', train=True, transform=cifar_trans_train, download=True)
cifar100_test_set = CIFAR100(root='dataset', train=False, transform=cifar_trans_test, download=True)

course_cifar_train_set = CoarseLabelCIFAR100(cifar100_train_set)
course_cifar_test_set = CoarseLabelCIFAR100(cifar100_test_set)

cifar100_train_loader = torch.utils.data.DataLoader(
    dataset=course_cifar_train_set,
    batch_size=BATCH_SIZE,
    shuffle=True)

cifar100_test_loader = torch.utils.data.DataLoader(
    dataset=course_cifar_test_set,
    batch_size=BATCH_SIZE,
    shuffle=True)

primary_model = VGG16(
    primary_task_output=PRIMARY_DIMENSION,
    auxiliary_task_output=AUX_DIMENSION
).to(device)
criterion = nn.CrossEntropyLoss()

optimizer_callback = lambda x: torch.optim.SGD(x.parameters(), lr=PRIMARY_LEARNING_RATE)
scheduler_callback = lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
# ---------

task_env = AuxTaskEnv(
    train_dataset=course_cifar_train_set,
    device=device,
    model=primary_model,
    criterion=criterion,
    optimizer_func=optimizer_callback,
    scheduler_func=scheduler_callback,
    batch_size=BATCH_SIZE,
    pri_dim=PRIMARY_DIMENSION,
    aux_dim=AUX_DIMENSION,
    aux_weight=AUX_WEIGHT,
    save_path=SAVE_PATH,
    learn_weights=LEARN_WEIGHTS,
    verbose=True,
)

auxilary_task_agent = get_ppo_agent(env=task_env,
                                    feature_dim=OBSERVATION_FEATURE_DIMENSION,
                                    auxiliary_dim=AUX_DIMENSION,
                                    learning_rate=PPO_LEARNING_RATE,
                                    device=device,
                                    ent_coef=0.01,
                                    n_steps=79,
                                    n_epochs=10,
                                    batch_size=BATCH_SIZE,
                                    weight_bins=21,
                                    )

auxilary_task_agent.set_parameters(LOAD_MODEL_PATH, device=device)

# ----

weight_tuning_env = WeightTuningEnv(
    train_dataset=course_cifar_train_set,
    device=device,
    model=primary_model,
    labeler=auxilary_task_agent,
    criterion=criterion,
    optimizer_func=optimizer_callback,
    scheduler_func=scheduler_callback,
    batch_size=BATCH_SIZE,
    pri_dim=PRIMARY_DIMENSION,
    aux_dim=AUX_DIMENSION,
    save_path=SAVE_PATH,
    verbose=True
)

weight_tuning_ppo_agent = get_weight_training_ppo_agent(weight_tuning_env,
                                                         feature_dim=OBSERVATION_FEATURE_DIMENSION,
                                                         device=device,
                                                         batch_size=BATCH_SIZE,
                                                         learning_rate=PPO_LEARNING_RATE,
                                                         ent_coef=0.01,
                                                         n_steps=BATCH_SIZE,
                                                         n_epochs=10)

train_auxilary_agent(
    primary_model=primary_model,
    rl_model=weight_tuning_ppo_agent,
    env=weight_tuning_env,
    device=device,
    test_loader=cifar100_test_loader,
    batch_size=BATCH_SIZE,
    total_epochs=TOTAL_EPOCH,
    save_path=SAVE_PATH,
    model_train_ratio=TRAIN_RATIO,
    primary_dimension=PRIMARY_DIMENSION,
    skip_rl=False,
    rl_pretraining_epochs=0,
)