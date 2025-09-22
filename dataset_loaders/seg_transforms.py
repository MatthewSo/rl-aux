# dataset_loaders/seg_transforms.py
from __future__ import annotations
from typing import Callable, Tuple
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD  = (0.229, 0.224, 0.225)

class PairCompose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, img: Image.Image, mask: Image.Image):
        for op in self.ops:
            img, mask = op(img, mask)
        return img, mask

class PairResize:
    def __init__(self, size: int | Tuple[int,int]): self.size = size
    def __call__(self, img, mask):
        img  = TF.resize(img,  self.size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        return img, mask

class PairRandomCrop:
    def __init__(self, size: int | Tuple[int,int]): self.size = size
    def __call__(self, img, mask):
        i, j, h, w = TF.random_crop.get_params(img, output_size=(self.size, self.size) if isinstance(self.size, int) else self.size)
        return TF.crop(img, i, j, h, w), TF.crop(mask, i, j, h, w)

class PairCenterCrop:
    def __init__(self, size: int | Tuple[int,int]): self.size = size
    def __call__(self, img, mask):
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)

class PairRandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask

class ToTensorNormalize:
    """image -> float tensor normalized; mask -> long tensor with class ids"""
    def __init__(self, mean=IMGNET_MEAN, std=IMGNET_STD):
        self.mean, self.std = mean, std
    def __call__(self, img, mask):
        img_t  = TF.to_tensor(img)
        img_t  = TF.normalize(img_t, self.mean, self.std)
        mask_np = np.array(mask, dtype=np.int64)
        mask_t  = torch.from_numpy(mask_np)
        return img_t, mask_t

def voc_train_transforms(resize=256, crop=224, hflip_p=0.5):
    return PairCompose([
        PairResize(resize),
        PairRandomCrop(crop),
        PairRandomHorizontalFlip(hflip_p),
        ToTensorNormalize(),
    ])

def voc_eval_transforms(resize=256, crop=224):
    return PairCompose([
        PairResize(resize),
        PairCenterCrop(crop),
        ToTensorNormalize(),
    ])
