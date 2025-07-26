import subprocess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset_loaders.cifar100 import CIFAR100, CoarseLabelCIFAR100, FineLabelCIFAR100
from dataset_loaders.transforms import cifar_trans_train, cifar_trans_test, cifar_256_test_transform, \
    cifar_256_train_transform
from l1_weight.l1_learn_auxilearn import train_meta_l1_network
from utils.path_name import create_path_name, save_parameter_dict
from utils.log import change_log_location
from torchvision.models import resnet50, ResNet50_Weights


BATCH_SIZE          = 16
PRIMARY_CLASS       = 100
TOTAL_EPOCH         = 15
PRIMARY_LR          = 1e-4
STEP_SIZE           = 50
GAMMA               = 0.5
INIT_GAMMA_RAW      = 1.0
LEARNED_RANGE       = 2.0
AUX_SET_RATIO       = 0.1
DEVICE              = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


save_path = create_path_name(
    agent_type            = "META_L1v2_small_gamma",
    primary_model_type    = "RESNET50",
    train_ratio           = 1,
    aux_weight            = None,
    observation_feature_dimensions = 0,
    dataset               = "CIFAR100",
    learn_weights         = True,
    optimizer             = "ADAM",
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
                      transform=cifar_256_train_transform, download=True)
test_base  = CIFAR100(root="./data/cifar100", train=False,
                      transform=cifar_256_test_transform, download=True)

train_set = FineLabelCIFAR100(train_base)
test_set  = FineLabelCIFAR100(test_base)

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True)
loader_test  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                          shuffle=False)

weights = ResNet50_Weights.DEFAULT          # = IMAGENET1K_V2 weights
model   = resnet50(weights=weights)
# Change output to 100 classes
model.fc = nn.Linear(model.fc.in_features, PRIMARY_CLASS)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=PRIMARY_LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

gamma_params = nn.ParameterList([
    nn.Parameter(torch.full_like(p, INIT_GAMMA_RAW))   # shape == p.shape
    for p in model.parameters()
])

INIT_GAMMA_RAW = torch.tensor(INIT_GAMMA_RAW, device=DEVICE)
gamma_optimizer = optim.Adam(gamma_params, lr=0.001, weight_decay=5e-4)
gamma_scheduler = optim.lr_scheduler.StepLR(gamma_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

train_meta_l1_network(
    device              = DEVICE,
    dataloader_train    = loader_train,
    dataloader_val      = loader_test,
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
    skip_meta= False,
    skip_regularization= False,
)
