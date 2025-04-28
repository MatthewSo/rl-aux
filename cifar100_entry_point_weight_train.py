import subprocess
from tabnanny import verbose

import torch
from torch import nn

from datasets.cifar100 import CIFAR100, CoarseLabelCIFAR100
from datasets.transforms import trans_train, trans_test
from environment.aux_task import AuxTaskEnv
from networks.ppo.ppo import get_ppo_agent, load_ppo_labeler
from networks.primary.vgg import VGG16
from train.train_auxilary_agent import train_auxilary_agent
from utils.log import log_print, change_log_location
from utils.path_name import create_path_name, save_all_parameters

LOAD_MODEL_PATH = "/home/cml0/rl-aux/trained_models/PPO_VGG_learn_weights_False_train_ratio_1_aux_weight_1_obs_dim_256_CIFAR100-20"
BATCH_SIZE = 100
AUX_DIMENSION = 100
PRIMARY_DIMENSION = 20
OBSERVATION_FEATURE_DIMENSION = 256
TOTAL_EPOCH = 200
PRIMARY_LEARNING_RATE = 0.01
PPO_LEARNING_RATE = 0.0003
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5
AUX_WEIGHT = 0
TRAIN_RATIO = 1

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

SAVE_PATH = create_path_name(
    agent_type="PPO",
    primary_model_type="VGG",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
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
    ppo_learning_rate=PPO_LEARNING_RATE,
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
)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_print("Torch CUDA available:", torch.cuda.is_available())
log_print("Torch CUDA device count:", torch.cuda.device_count())
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

log_print("Using device:", device)

# ---------

labeler_model = load_ppo_labeler(LOAD_MODEL_PATH, device)
log_print("Labeler model loaded from:", LOAD_MODEL_PATH)

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