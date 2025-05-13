import pickle
from collections import OrderedDict
from datasets.cifar10 import CIFAR10
from datasets.transforms import cifar_trans_test, cifar_trans_train
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
from train.model.performance import EpochPerformance
from utils.log import change_log_location, log_print
from utils.path_name import create_path_name
from wamal.networks.vgg_16 import SimplifiedVGG16
from wamal.networks.wamal_wrapper import WamalWrapper, LabelWeightWrapper
from wamal.train_network import train_wamal_network

AUX_WEIGHT = 0

PRIMARY_CLASS = 10
AUXILIARY_CLASS = 50

save_path = create_path_name(
    agent_type="MAXL",
    primary_model_type="VGG",
    train_ratio=1,
    aux_weight=AUX_WEIGHT,
    observation_feature_dimensions=0,
    dataset="CIFAR10",
    learn_weights=True,
)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

train_set = CIFAR10(
    root="./data/cifar10",
    train=True,
    transform=cifar_trans_train
)
test_set = CIFAR10(
    root="./data/cifar10",
    train=False,
    transform=cifar_trans_test
)

trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])
trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])

batch_size = 100
dataloader_train = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=True
)

change_log_location(save_path)
epoch_performances=[]


kwargs = {'num_workers': 1, 'pin_memory': True}

psi = [AUXILIARY_CLASS // PRIMARY_CLASS] * PRIMARY_CLASS
label_model = LabelWeightWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS).to(device), num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=(3,32,32) )
label_model = label_model.to(device)
gen_optimizer = optim.SGD(label_model.parameters(), lr=1e-3, weight_decay=5e-4)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 200
train_batch = len(dataloader_train)
test_batch = len(dataloader_test)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
wamal_main_model = WamalWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS).to(device))
optimizer = optim.SGD(wamal_main_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
vgg_lr = 0.01  # define learning rate for second-derivative step (theta_1^+)
k = 0

train_wamal_network(device=device, dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                    total_epoch=total_epoch, train_batch=train_batch, test_batch=test_batch, batch_size=batch_size,
                    model=wamal_main_model, label_network=label_model, optimizer=optimizer, scheduler=scheduler,
                    gen_optimizer=gen_optimizer, gen_scheduler=gen_scheduler,
                    num_axuiliary_classes=AUXILIARY_CLASS, num_primary_classes=PRIMARY_CLASS,
                    save_path=save_path, use_learned_weights=True, model_lr=vgg_lr)