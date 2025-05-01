import subprocess
from tabnanny import verbose

import torch
from torch import nn

from datasets.places_365 import Places365
from datasets.transforms import places365_trans_train, places365_trans_test
from environment.learn_weight_aux_task import AuxTaskEnv
from networks.ppo.ppo import get_ppo_agent
from networks.primary.vgg import VGG16
from train.train_auxilary_agent import train_auxilary_agent
from utils.log import log_print, change_log_location
from utils.path_name import create_path_name, save_all_parameters

BATCH_SIZE = 100
AUX_DIMENSION = 1825
PRIMARY_DIMENSION = 365
OBSERVATION_FEATURE_DIMENSION = 256
TOTAL_EPOCH = 200
PRIMARY_LEARNING_RATE = 0.01
PPO_LEARNING_RATE = 0.0003
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5
AUX_WEIGHT = 0
TRAIN_RATIO = 1
LEARN_WEIGHTS = True

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

SAVE_PATH = create_path_name(
    agent_type="PPO",
    primary_model_type="VGG",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
    learn_weights=LEARN_WEIGHTS,
    observation_feature_dimensions=OBSERVATION_FEATURE_DIMENSION,
    dataset="PLACES365",
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
    dataset="PLACES365",
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

places365_train_set = Places365(
    root="/home/places365",
    train=True,
    transform=places365_trans_train
)
places365_test_set = Places365(
    root="/home/places365",
    train=False,
    transform=places365_trans_test)

train_loader = torch.utils.data.DataLoader(
    dataset=places365_train_set,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=places365_test_set,
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
    train_dataset=places365_train_set,
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
    image_shape=(3, 224, 224),
    verbose=True,
)

auxilary_task_agent = get_ppo_agent(env=env,
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

log_print("Done Initializing PPO Agent")

# Train the PPO agent
train_auxilary_agent(
    primary_model=primary_model,
    rl_model=auxilary_task_agent,
    env=env,
    device=device,
    test_loader= test_loader,
    batch_size=BATCH_SIZE,
    total_epochs=TOTAL_EPOCH,
    save_path=SAVE_PATH,
    model_train_ratio=TRAIN_RATIO,
    primary_dimension=PRIMARY_DIMENSION,
    skip_rl=False
)