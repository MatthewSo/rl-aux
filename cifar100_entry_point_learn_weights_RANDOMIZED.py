import subprocess
from tabnanny import verbose

import torch
from torch import nn

from datasets.cifar100 import CIFAR100, CoarseLabelCIFAR100
from datasets.transforms import trans_train, trans_test
from environment.learn_weight_aux_task import AuxTaskEnv
from networks.ppo.ppo import get_ppo_agent, get_fast_dummy_ppo_agent
from networks.primary.vgg import VGG16
from train.train_auxilary_agent import train_auxilary_agent
from utils.log import log_print, change_log_location
from utils.path_name import create_path_name, save_all_parameters

BATCH_SIZE = 100
AUX_DIMENSION = 100
PRIMARY_DIMENSION = 20
OBSERVATION_FEATURE_DIMENSION = 256
TOTAL_EPOCH = 200
PRIMARY_LEARNING_RATE = 0.01
PPO_LEARNING_RATE = 0.0003
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5
AUX_WEIGHT = 1
TRAIN_RATIO = 1
LEARN_WEIGHTS = False

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

SAVE_PATH = create_path_name(
    agent_type="RANDOMIZED-PPO",
    primary_model_type="VGG",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
    learn_weights=LEARN_WEIGHTS,
    observation_feature_dimensions=OBSERVATION_FEATURE_DIMENSION,
    dataset="CIFAR100-20",
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
    learn_weights=LEARN_WEIGHTS,
    train_ratio=TRAIN_RATIO,
    save_path=SAVE_PATH,
    dataset="CIFAR100-20",
    model_name="VGG",
    agent_type="PPO",
    observation_feature_dimensions=OBSERVATION_FEATURE_DIMENSION,
    aux_task_type="AuxTask",
    primary_task_type="VGG",
    git_commit_hash=git_hash,
)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_print("Torch CUDA available:", torch.cuda.is_available())
log_print("Torch CUDA device count:", torch.cuda.device_count())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

log_print("Using device:", device)

# ---------

cifar100_train_set = CIFAR100(root='dataset', train=True, transform=trans_train, download=True)
cifar100_test_set = CIFAR100(root='dataset', train=False, transform=trans_test, download=True)

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
#optimizer_callback = lambda x: torch.optim.Adam(x.parameters(), lr=PRIMARY_LEARNING_RATE)
#scheduler_callback = lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

optimizer_callback = lambda x: torch.optim.SGD(x.parameters(), lr=PRIMARY_LEARNING_RATE)
scheduler_callback = lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

env = AuxTaskEnv(
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

auxilary_task_agent = get_fast_dummy_ppo_agent(
    env, device
)

log_print("Done Initializing PPO Agent")

# Train the PPO agent
train_auxilary_agent(
    primary_model=primary_model,
    aux_task_model=auxilary_task_agent,
    env=env,
    device=device,
    test_loader=cifar100_test_loader,
    batch_size=BATCH_SIZE,
    total_epochs=TOTAL_EPOCH,
    save_path=SAVE_PATH,
    model_train_ratio=TRAIN_RATIO,
    primary_dimension=PRIMARY_DIMENSION,
    skip_rl=True
)