import subprocess

from dataset_loaders.capped_dataset import PerClassCap
from dataset_loaders.food101 import Food101
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler

from dataset_loaders.places_365 import Places365
from utils.log import change_log_location
from utils.path_name import create_path_name, save_parameter_dict
from wamal.argparse import RUN_ID, GPU
from wamal.networks.vit import get_vit, vit_collate
from wamal.networks.wamal_wrapper import WamalWrapper, LabelWeightWrapper
from wamal.train_network import train_wamal_network
from transformers import ViTForImageClassification, ViTImageProcessor

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

AUX_WEIGHT = 0
BATCH_SIZE = 30
PRIMARY_CLASS = 365
AUXILIARY_CLASS = 1825
SKIP_MAL = False
LEARN_WEIGHTS = False
TOTAL_EPOCH = 75
PRIMARY_LR = 5e-4
STEP_SIZE = 50
IMAGE_SHAPE = (3, 256, 256)
GAMMA = 0.5
GEN_OPTIMIZER_LR = 1e-3
GEN_OPTIMIZER_WEIGHT_DECAY = 5e-4
TRAIN_RATIO = 1
OPTIMIZER = "SGD"
FULL_DATASET = True
RANGE = 5.0
USE_AUXILIARY_SET = False
AUXILIARY_SET_RATIO = 0.0
NORMALIZE_BATCH = False
BATCH_FRACTION = None
ENTROPY_LOSS_FACTOR = 0.2

save_path = create_path_name(
    agent_type="WAMAL-MAXL",
    primary_model_type="VIT",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
    observation_feature_dimensions=0,
    dataset="PLACES365",
    learn_weights=LEARN_WEIGHTS,
    optimizer =OPTIMIZER,
    full_dataset=FULL_DATASET,
    learning_rate=PRIMARY_LR,
    range=RANGE,
    aux_set_ratio=None,
    normalize_batch=False,
    batch_fraction=None,
    entropy_loss_factor=0.2,
    run_id=RUN_ID
)
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

train_set = Places365(
    root="./data/places365",
    train=True,
    download=True,
)
test_set = Places365(
    root="./data/places365",
    train=False,
    download=True,
)

if not FULL_DATASET:
    train_set = PerClassCap(train_set)

dataloader_train = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=vit_collate
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=vit_collate
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
        "range": RANGE,
    }
)

change_log_location(save_path)
epoch_performances=[]

kwargs = {'num_workers': 1, 'pin_memory': True}

psi = [AUXILIARY_CLASS // PRIMARY_CLASS] * PRIMARY_CLASS

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
backbone_model     = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device).eval()

label_model = LabelWeightWrapper(backbone_model, num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE )
label_model = label_model.to(device)
gen_optimizer = optim.SGD(label_model.parameters(), lr=GEN_OPTIMIZER_LR, weight_decay=GEN_OPTIMIZER_WEIGHT_DECAY)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
# define parameters
total_epoch = TOTAL_EPOCH
train_batch = len(dataloader_train)
test_batch = len(dataloader_test)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
backbone_model     = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device).eval()

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
wamal_main_model = WamalWrapper(backbone_model,num_primary=PRIMARY_CLASS, num_auxiliary=AUXILIARY_CLASS, input_shape=IMAGE_SHAPE)
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
