from tabnanny import verbose

import torch
from torch import nn

from datasets.cifar100 import CIFAR100, CoarseLabelCIFAR100
from datasets.transforms import trans_train, trans_test
from environment.aux_task import AuxTaskEnv
from networks.ppo.ppo import get_ppo_agent
from networks.primary.vgg import VGG16
from train.train_auxilary_agent import train_auxilary_agent

BATCH_SIZE = 64
AUX_DIMENSION = 100
PRIMARY_DIMENSION = 20
OBSERVATION_FEATURE_DIMENSION = 256
TOTAL_EPOCH = 200
PRIMARY_LEARNING_RATE = 0.01
PPO_LEARNING_RATE = 0.0003
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5
AUX_WEIGHT = 0.5

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#select 5th gpu
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device count:", torch.cuda.device_count())
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

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
optimizer_callback = lambda x: torch.optim.Adam(x.parameters(), lr=PRIMARY_LEARNING_RATE)
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
                                    )

print("Done Initializing PPO Agent")

# Train the PPO agent
train_auxilary_agent(
    primary_model=primary_model,
    aux_task_model=auxilary_task_agent,
    env=env,
    device=device,
    test_loader=cifar100_test_loader,
    batch_size=BATCH_SIZE,
    total_epochs=TOTAL_EPOCH,
)