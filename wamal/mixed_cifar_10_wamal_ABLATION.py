import subprocess

from dataset_loaders.cifar10 import CIFAR10
from dataset_loaders.cifar100 import CIFAR100, CoarseLabelCIFAR100
from dataset_loaders.transforms import cifar_trans_test, cifar_trans_train
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
from utils.log import change_log_location, log_print
from utils.path_name import create_path_name, save_parameter_dict, create_path_name_short
from wamal.networks.vgg_16 import SimplifiedVGG16
from wamal.networks.wamal_wrapper import WamalWrapper, LabelWeightWrapper
from wamal.train_network import train_wamal_network
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

AUX_WEIGHT = 0
BATCH_SIZE = 100
PRIMARY_CLASS = 20
AUXILIARY_CLASS = 100
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
RANGE = 5
USE_AUXILIARY_SET = False
AUXILIARY_SET_RATIO = 0.0
NORMALIZE_BATCH = False
BATCH_FRACTION = None
ENTROPY_LOSS_FACTOR = 0.2

# KEWWORD ARGS for PRIMARY BACKBONE and LABEL BACKBONE
parser = argparse.ArgumentParser(description='WAMAL CIFAR100-20 Training')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
parser.add_argument('--primary_backbone', type=str, default='vgg16', help='Primary backbone model (default: vgg16)')
parser.add_argument('--label_backbone', type=str, default='resnet50', help='Label backbone model (default: resnet50)')
args = parser.parse_args()
GPU = args.gpu
# SAMPLE USAGE = python -m wamal.vgg16_cifar_100_20_wamal_RANGE_ABLATION.py --primary_backbone vgg16 --label_backbone resnet50 --gpu 0
primary_model_type = f"PRIMARY_{args.primary_backbone.upper()}_LABEL_{args.label_backbone.upper()}"
log_print("Primary Backbone:", args.primary_backbone)
log_print("Label Backbone:", args.label_backbone)

save_path = create_path_name_short(
    agent_type="WAMAL_MIXED_ABLATION_CIFAR10",
    primary_model_type=primary_model_type,
)
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
log_print("GPU Device:", device)

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
        "optimizer": OPTIMIZER,
        "full_dataset": FULL_DATASET,
        "use_auxiliary_set": USE_AUXILIARY_SET,
        "auxiliary_set_ratio": AUXILIARY_SET_RATIO,
        "entropy_loss_factor": ENTROPY_LOSS_FACTOR,
        "range": RANGE,
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

if args.label_backbone.lower() == 'resnet50':
    weights = ResNet50_Weights.DEFAULT          # = IMAGENET1K_V2 weights
    resnet_model   = resnet50(weights=weights)
    label_model = LabelWeightWrapper(resnet_model, num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE )
    label_model = label_model.to(device)
elif args.label_backbone.lower() == 'resnet18':
    weights = ResNet18_Weights.DEFAULT  # = IMAGENET1K_V2 weights
    resnet_model = resnet18(weights=weights)
    label_model = LabelWeightWrapper(resnet_model, num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE)
    label_model = label_model.to(device)
elif args.label_backbone.lower() == 'vgg16':
    label_model = LabelWeightWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS), num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE )
    label_model = label_model.to(device)
elif args.label_backbone.lower() == 'none':
    log_print("No label backbone specified, using None.")
    label_model = LabelWeightWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS), num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE )
    label_model = label_model.to(device)
    SKIP_MAL = True
    LEARN_WEIGHTS = False
else:
    raise ValueError(f"Label backbone {args.label_backbone} not recognized. Use 'resnet50' or 'vgg16'.")

log_print("Label Model:", label_model)
gen_optimizer = optim.SGD(label_model.parameters(), lr=GEN_OPTIMIZER_LR, weight_decay=GEN_OPTIMIZER_WEIGHT_DECAY)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
# define parameters
total_epoch = TOTAL_EPOCH
train_batch = len(dataloader_train)
test_batch = len(dataloader_test)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
if args.primary_backbone.lower() == 'resnet50':
    weights = ResNet50_Weights.DEFAULT          # = IMAGENET1K_V2 weights
    resnet_model   = resnet50(weights=weights)
    wamal_main_model = WamalWrapper(resnet_model,num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE)
elif args.primary_backbone.lower() == 'resnet18':
    weights = ResNet18_Weights.DEFAULT  # = IMAGENET1K_V2 weights
    resnet_model = resnet18(weights=weights)
    wamal_main_model = WamalWrapper(resnet_model, num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE)
elif args.primary_backbone.lower() == 'vgg16':
    wamal_main_model = WamalWrapper(SimplifiedVGG16(device=device,num_primary_classes=PRIMARY_CLASS),num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE)

log_print("Primary Model:, ", wamal_main_model)

wamal_main_model = wamal_main_model.to(device)

if OPTIMIZER == "SGD":
    optimizer = optim.SGD(wamal_main_model.parameters(), lr=PRIMARY_LR)
elif OPTIMIZER == "ADAM":
    optimizer = optim.Adam(wamal_main_model.parameters(), lr=PRIMARY_LR, weight_decay=5e-4)
else:
    raise ValueError(f"Optimizer {OPTIMIZER} not recognized. Use 'SGD' or 'ADAM'.")

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
vgg_lr = PRIMARY_LR # define learning rate for second-derivative step (theta_1^+)

train_wamal_network(device=device, dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                    total_epoch=total_epoch, train_batch=train_batch, test_batch=test_batch, batch_size=BATCH_SIZE,
                    model=wamal_main_model, label_network=label_model, optimizer=optimizer, scheduler=scheduler,
                    gen_optimizer=gen_optimizer, gen_scheduler=gen_scheduler,
                    num_axuiliary_classes=AUXILIARY_CLASS, num_primary_classes=PRIMARY_CLASS,
                    save_path=save_path, use_learned_weights=LEARN_WEIGHTS, model_lr=vgg_lr, skip_mal=SKIP_MAL, val_range=RANGE, use_auxiliary_set=USE_AUXILIARY_SET,
                    aux_split=AUXILIARY_SET_RATIO, batch_frac= BATCH_FRACTION, normalize_batch_weights=NORMALIZE_BATCH, entropy_loss_factor=ENTROPY_LOSS_FACTOR)