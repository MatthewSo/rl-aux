import subprocess

from dataset_loaders.cifar10 import CIFAR10
from dataset_loaders.transforms import cifar_trans_test, cifar_trans_train
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
from utils.log import change_log_location
from utils.path_name import create_path_name, save_parameter_dict
from wamal.networks.vgg_16 import SimplifiedVGG16
from wamal.networks.wamal_wrapper import WamalWrapper, LabelWeightWrapper
from wamal.train_network import train_wamal_network

AUX_WEIGHT = 0
BATCH_SIZE = 100
PRIMARY_CLASS = 10
AUXILIARY_CLASS = 50
SKIP_MAL = False
LEARN_WEIGHTS = True
TOTAL_EPOCH = 200
PRIMARY_LR = 0.02
STEP_SIZE = 50
IMAGE_SHAPE = (3, 32, 32)
GAMMA = 0.5
GEN_OPTIMIZER_LR = 1e-3
GEN_OPTIMIZER_WEIGHT_DECAY = 5e-4
TRAIN_RATIO = 1
OPTIMIZER = "SGD"
FULL_DATASET = True
save_path = create_path_name(
    agent_type="WAMAL",
    primary_model_type="VGG",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
    observation_feature_dimensions=0,
    dataset="CIFAR10",
    learn_weights=LEARN_WEIGHTS,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


###  DON'T CHANGE THIS PART ###
git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
save_parameter_dict(
    {
        "batch_size": BATCH_SIZE,
        "aux_dimensions": AUXILIARY_CLASS,
        "primary_dimensions": PRIMARY_CLASS,
        "total_epoch": TOTAL_EPOCH,
        "git_commit_hash": git_hash,
        "primary_learning_rate": PRIMARY_LR,
        "scheduler_step_size": STEP_SIZE,
        "scheduler_gamma": GAMMA,
        "aux_weight": AUX_WEIGHT,
        "save_path": save_path,
        "skip_mal": SKIP_MAL,
        "image_shape": IMAGE_SHAPE,
        "learn_weights": LEARN_WEIGHTS,
        "gen_optimizer_weight_decay": GEN_OPTIMIZER_WEIGHT_DECAY,
        "gen_optimizer_lr": GEN_OPTIMIZER_LR,
        "train_ratio": TRAIN_RATIO,
    }
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

change_log_location(save_path)
epoch_performances=[]

kwargs = {'num_workers': 1, 'pin_memory': True}
simpliefied_vgg = SimplifiedVGG16(device=device, num_primary_classes=PRIMARY_CLASS)

psi = [AUXILIARY_CLASS // PRIMARY_CLASS] * PRIMARY_CLASS
label_model = LabelWeightWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS), num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE )
label_model = label_model.to(device)
gen_optimizer = optim.SGD(label_model.parameters(), lr=GEN_OPTIMIZER_LR, weight_decay=GEN_OPTIMIZER_WEIGHT_DECAY)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
# define parameters
total_epoch = TOTAL_EPOCH
train_batch = len(dataloader_train)
test_batch = len(dataloader_test)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
wamal_main_model = WamalWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS),num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE)
wamal_main_model = wamal_main_model.to(device)
optimizer = optim.SGD(wamal_main_model.parameters(), lr=PRIMARY_LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
vgg_lr = PRIMARY_LR # define learning rate for second-derivative step (theta_1^+)

train_wamal_network(device=device, dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                    total_epoch=total_epoch, train_batch=train_batch, test_batch=test_batch, batch_size=BATCH_SIZE,
                    model=wamal_main_model, label_network=label_model, optimizer=optimizer, scheduler=scheduler,
                    gen_optimizer=gen_optimizer, gen_scheduler=gen_scheduler,
                    num_axuiliary_classes=AUXILIARY_CLASS, num_primary_classes=PRIMARY_CLASS,
                    save_path=save_path, use_learned_weights=LEARN_WEIGHTS, model_lr=vgg_lr, skip_mal=SKIP_MAL)