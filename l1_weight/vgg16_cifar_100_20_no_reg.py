import subprocess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets.cifar100 import CIFAR100, CoarseLabelCIFAR100
from datasets.transforms import cifar_trans_train, cifar_trans_test
from l1_weight.l1_learn_auxilearn import train_meta_l1_network
from utils.path_name import create_path_name, save_parameter_dict
from utils.log import change_log_location
from wamal.networks.vgg_16 import SimplifiedVGG16



BATCH_SIZE          = 100
PRIMARY_CLASS       = 20
TOTAL_EPOCH         = 200
PRIMARY_LR          = 0.01
STEP_SIZE           = 50
GAMMA               = 0.5
INIT_GAMMA_RAW      = 0.0
LEARNED_RANGE       = 2.0
AUX_SET_RATIO       = 0.1
DEVICE              = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


save_path = create_path_name(
    agent_type            = "SKIP_REG",
    primary_model_type    = "VGG",
    train_ratio           = 1,
    aux_weight            = None,
    observation_feature_dimensions = 0,
    dataset               = "CIFAR100-20",
    learn_weights         = True,
    optimizer             = "SGD",
    full_dataset          = True,
    learning_rate         = PRIMARY_LR,
    range                 = LEARNED_RANGE,
    aux_set_ratio         = AUX_SET_RATIO,
    normalize_batch       = False,
    batch_fraction        = None,
)
git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
save_parameter_dict({
    "batch_size"      : BATCH_SIZE,
    "primary_classes" : PRIMARY_CLASS,
    "total_epoch"     : TOTAL_EPOCH,
    "primary_lr"      : PRIMARY_LR,
    "step_size"       : STEP_SIZE,
    "gamma_lr_decay"  : GAMMA,
    "init_gamma_raw"  : INIT_GAMMA_RAW,
    "learned_range"   : LEARNED_RANGE,
    "aux_set_ratio"   : AUX_SET_RATIO,
    "git_commit_hash" : git_hash,
    "save_path"       : save_path,
})
change_log_location(save_path)

train_base = CIFAR100(root="./data/cifar100", train=True,
                      transform=cifar_trans_train, download=True)
test_base  = CIFAR100(root="./data/cifar100", train=False,
                      transform=cifar_trans_test, download=True)

train_set = CoarseLabelCIFAR100(train_base)
test_set  = CoarseLabelCIFAR100(test_base)

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
loader_test  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

model = SimplifiedVGG16(device=DEVICE, num_primary_classes=PRIMARY_CLASS).to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=PRIMARY_LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

gamma_params = nn.ParameterList([
    nn.Parameter(torch.tensor(INIT_GAMMA_RAW, device=DEVICE))
    for _ in model.parameters()
])

INIT_GAMMA_RAW = torch.tensor(INIT_GAMMA_RAW, device=DEVICE)
gamma_optimizer = optim.Adam(gamma_params, lr=0.01, weight_decay=5e-4)
gamma_scheduler = optim.lr_scheduler.StepLR(gamma_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

train_meta_l1_network(
    device              = DEVICE,
    dataloader_train    = loader_train,
    dataloader_val      = None,
    dataloader_test     = loader_test,
    total_epoch         = TOTAL_EPOCH,
    batch_size          = BATCH_SIZE,
    gamma_params        = gamma_params,
    model               = model,
    optimizer           = optimizer,
    scheduler           = scheduler,
    gamma_optimizer     = gamma_optimizer,
    gamma_scheduler     = gamma_scheduler,
    num_primary_classes = PRIMARY_CLASS,
    save_path           = save_path,
    init_gamma          = INIT_GAMMA_RAW,
    learned_range       = LEARNED_RANGE,
    aux_split           = AUX_SET_RATIO,
    skip_meta= True,
    skip_regularization= True,
)