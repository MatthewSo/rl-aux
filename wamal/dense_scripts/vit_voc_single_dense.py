# vit_voc_wamal_dense.py
import subprocess, os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# === Our VOC dense dataset & joint transforms (from prior step) ===
from dataset_loaders.voc_seg import VOCSegmentation            # (B,3,H,W), (B,H,W)
from dataset_loaders.seg_transforms import voc_train_transforms, voc_eval_transforms
from networks.primary.vit_2d_backbone import ViT2DBackbone

# === Repo utilities & training loop ===
from utils.log import change_log_location
from utils.path_name import create_path_name, save_parameter_dict
from wamal.argparse import GPU, RUN_ID

from wamal.train_network_dense import train_network_dense
from wamal.networks.wamal_wrapper_dense import WamalDenseWrapper, LabelWeightDenseWrapper

# Optional: keep parity with your other ViT launchers
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ------------------- CONFIG -------------------
AUX_WEIGHT             = 0                 # kept for parity with other scripts
BATCH_SIZE             = 8
PRIMARY_CLASS          = VOCSegmentation.NUM_CLASSES     # 21 incl. background
K_SUBLABELS_PER_CLASS  = 5
AUXILIARY_CLASS        = PRIMARY_CLASS * K_SUBLABELS_PER_CLASS
SKIP_MAL               = True
LEARN_WEIGHTS          = False

TOTAL_EPOCH            = 80
PRIMARY_LR             = 5e-4
STEP_SIZE              = 50
GAMMA                  = 0.5

GEN_OPTIMIZER_LR       = 1e-3
GEN_OPTIMIZER_WEIGHT_DECAY = 5e-4

TRAIN_RATIO            = 1
OPTIMIZER              = "ADAM"            # "SGD" or "ADAM" (your ViT scripts often used SGD/Adam)
FULL_DATASET           = True
RANGE                  = 5.0
USE_AUXILIARY_SET      = False
AUXILIARY_SET_RATIO    = 0.0
NORMALIZE_BATCH        = False
BATCH_FRACTION         = None
ENTROPY_LOSS_FACTOR    = 0.2
IMAGE_SHAPE            = (3, 224, 224)
IGNORE_INDEX           = VOCSegmentation.IGNORE_INDEX

SAVE_PATH = create_path_name(
    agent_type="WAMAL-DENSE",
    primary_model_type="VIT",
    train_ratio=TRAIN_RATIO,
    aux_weight=AUX_WEIGHT,
    observation_feature_dimensions=0,
    dataset="VOC2012",
    learn_weights=LEARN_WEIGHTS,
    optimizer=OPTIMIZER,
    full_dataset=FULL_DATASET,
    learning_rate=PRIMARY_LR,
    range=RANGE,
    aux_set_ratio=AUXILIARY_SET_RATIO if USE_AUXILIARY_SET else None,
    normalize_batch=NORMALIZE_BATCH,
    batch_fraction=BATCH_FRACTION,
    entropy_loss_factor=ENTROPY_LOSS_FACTOR,
    run_id=RUN_ID
)
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

# ------------------- DATA -------------------
train_set = VOCSegmentation(
    root="./data/voc",
    train=True,
    transforms=voc_train_transforms(resize=256, crop=224),   # fixed 224 so ViT pos-emb stays valid
    download=True,
)
val_set = VOCSegmentation(
    root="./data/voc",
    train=False,
    transforms=voc_eval_transforms(resize=256, crop=224),
    download=True,
)

dataloader_train = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
dataloader_val = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)

# ------------------- LOG + META -------------------
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
        "save_path": SAVE_PATH,
        "skip_mal": SKIP_MAL,
        "image_shape": IMAGE_SHAPE,
        "learn_weights": LEARN_WEIGHTS,
        "gen_optimizer_weight_decay": GEN_OPTIMIZER_WEIGHT_DECAY,
        "gen_optimizer_lr": GEN_OPTIMIZER_LR,
        "train_ratio": TRAIN_RATIO,
        "optimizer": OPTIMIZER,
        "range": RANGE,
        "use_auxiliary_set": USE_AUXILIARY_SET,
        "auxiliary_set_ratio": AUXILIARY_SET_RATIO,
        "normalize_batch": NORMALIZE_BATCH,
        "batch_fraction": BATCH_FRACTION,
        "entropy_loss_factor": ENTROPY_LOSS_FACTOR,
        "ignore_index": IGNORE_INDEX,
        "k_sublabels_per_class": K_SUBLABELS_PER_CLASS,
    }
)
change_log_location(SAVE_PATH)

# ------------------- MODELS -------------------
# ViT -> 2D feature map; same pretrained family as your previous ViT runs (HuggingFace).
vit_backbone_main  = ViT2DBackbone("google/vit-base-patch16-224")
vit_backbone_label = ViT2DBackbone("google/vit-base-patch16-224")

model = WamalDenseWrapper(
    backbone=vit_backbone_main,
    num_primary=PRIMARY_CLASS,
    num_auxiliary=AUXILIARY_CLASS,
    input_shape=IMAGE_SHAPE,
    upsample_to_input=True
).to(device)

label_model = LabelWeightDenseWrapper(
    backbone=vit_backbone_label,
    num_primary=PRIMARY_CLASS,
    num_auxiliary=AUXILIARY_CLASS,
    input_shape=IMAGE_SHAPE,
    upsample_to_input=True
).to(device)

# ------------------- OPTS + SCHEDS -------------------
if OPTIMIZER.upper() == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=PRIMARY_LR, momentum=0.9, weight_decay=5e-4)
elif OPTIMIZER.upper() == "ADAM":
    optimizer = optim.Adam(model.parameters(), lr=PRIMARY_LR, weight_decay=5e-4)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
gen_optimizer = optim.SGD(label_model.parameters(), lr=GEN_OPTIMIZER_LR, weight_decay=GEN_OPTIMIZER_WEIGHT_DECAY)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ------------------- TRAIN -------------------
train_network_dense(
    device=device,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_val,
    total_epoch=TOTAL_EPOCH,
    batch_size=BATCH_SIZE,
    model=model,
    label_network=label_model,
    optimizer=optimizer,
    scheduler=scheduler,
    gen_optimizer=gen_optimizer,
    gen_scheduler=gen_scheduler,
    num_primary_classes=PRIMARY_CLASS,
    num_auxiliary_classes=AUXILIARY_CLASS,
    save_path=SAVE_PATH,
    use_learned_weights=LEARN_WEIGHTS,
    model_lr=PRIMARY_LR,
    val_range=RANGE,
    use_auxiliary_set=USE_AUXILIARY_SET,
    aux_split=AUXILIARY_SET_RATIO,
    skip_mal=SKIP_MAL,
    normalize_batch_weights=NORMALIZE_BATCH,
    batch_frac=BATCH_FRACTION,
    ignore_index=IGNORE_INDEX,
    entropy_loss_factor=ENTROPY_LOSS_FACTOR,
)
