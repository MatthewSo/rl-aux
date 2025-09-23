# wamal/dense_scripts/vit_coco_wamal_dense.py
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# The script already parses --gpu N; pin it here so CUDA context is created once & correctly:
# (Replace '0' with the parsed GPU index string)

# Optional but helpful when debugging:
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import subprocess, os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# joint img/mask transforms you already added for VOC; reuse here
from dataset_loaders.seg_transforms import voc_train_transforms, voc_eval_transforms
from dataset_loaders.coco_panoptic import COCOPanopticSeg
from networks.primary.vit_2d_backbone import ViT2DBackbone

from utils.log import change_log_location
from utils.path_name import create_path_name, save_parameter_dict
from wamal.argparse import GPU, RUN_ID  # same arg helper your other scripts use

from wamal.train_network_dense import train_network_dense
from wamal.networks.wamal_wrapper_dense import WamalDenseWrapper, LabelWeightDenseWrapper

# Keep parity with other ViT launchers
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ------------------- CONFIG -------------------
BATCH_SIZE             = 4
INCLUDE_THINGS         = True           # False => stuff-only semantics
K_SUBLABELS_PER_CLASS  = 5

TOTAL_EPOCH            = 80
PRIMARY_LR             = 5e-4
STEP_SIZE              = 50
GAMMA                  = 0.5

GEN_OPTIMIZER_LR       = 1e-3
GEN_OPTIMIZER_WEIGHT_DECAY = 5e-4

OPTIMIZER              = "SGD"         # or "SGD"
LEARN_WEIGHTS          = True
SKIP_MAL               = False
RANGE                  = 5.0
USE_AUXILIARY_SET      = False
AUXILIARY_SET_RATIO    = 0.0
NORMALIZE_BATCH        = False
BATCH_FRACTION         = None
ENTROPY_LOSS_FACTOR    = 0.2

IMAGE_SHAPE            = (3, 224, 224)
IGNORE_INDEX           = 255
TRAIN_RATIO            = 1
FULL_DATASET           = True

DATA_ROOT              = "./data/coco"

SAVE_PATH = create_path_name(
    agent_type="WAMAL-DENSE",
    primary_model_type="VIT",
    train_ratio=TRAIN_RATIO,
    aux_weight=0,
    observation_feature_dimensions=0,
    dataset=f"COCO-PANOPTIC({'things+stuff' if INCLUDE_THINGS else 'stuff-only'})",
    learn_weights=LEARN_WEIGHTS,
    optimizer=OPTIMIZER,
    full_dataset=FULL_DATASET,
    learning_rate=PRIMARY_LR,
    range=RANGE,
    aux_set_ratio=AUXILIARY_SET_RATIO if USE_AUXILIARY_SET else None,
    normalize_batch=NORMALIZE_BATCH,
    batch_fraction=BATCH_FRACTION,
    entropy_loss_factor=ENTROPY_LOSS_FACTOR,
    run_id=RUN_ID,
)
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

# ------------------- DATA -------------------
# Use 224×224 crops to keep ViT positional grid square (14×14 for patch16).
train_set = COCOPanopticSeg(
    root=DATA_ROOT,
    train=True,
    transforms=voc_train_transforms(resize=256, crop=224),
    include_things=INCLUDE_THINGS,
)
val_set = COCOPanopticSeg(
    root=DATA_ROOT,
    train=False,
    transforms=voc_eval_transforms(resize=256, crop=224),
    include_things=INCLUDE_THINGS,
)

PRIMARY_CLASS = train_set.NUM_CLASSES
AUXILIARY_CLASS = PRIMARY_CLASS * K_SUBLABELS_PER_CLASS

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
        "save_path": SAVE_PATH,
        "skip_mal": SKIP_MAL,
        "image_shape": IMAGE_SHAPE,
        "learn_weights": LEARN_WEIGHTS,
        "gen_optimizer_weight_decay": GEN_OPTIMIZER_WEIGHT_DECAY,
        "gen_optimizer_lr": GEN_OPTIMIZER_LR,
        "optimizer": OPTIMIZER,
        "range": RANGE,
        "use_auxiliary_set": USE_AUXILIARY_SET,
        "auxiliary_set_ratio": AUXILIARY_SET_RATIO,
        "normalize_batch": NORMALIZE_BATCH,
        "batch_fraction": BATCH_FRACTION,
        "entropy_loss_factor": ENTROPY_LOSS_FACTOR,
        "ignore_index": IGNORE_INDEX,
        "k_sublabels_per_class": K_SUBLABELS_PER_CLASS,
        "include_things": INCLUDE_THINGS,
        "data_root": DATA_ROOT,
    }
)
change_log_location(SAVE_PATH)

# ------------------- MODELS -------------------
# Same pretrained ViT family you used previously.
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
