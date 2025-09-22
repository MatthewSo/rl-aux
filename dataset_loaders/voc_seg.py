# dataset_loaders/voc_seg.py
from __future__ import annotations
from typing import Callable, Optional
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation as VOCSegmentationOriginal

class VOCSegmentation(Dataset):
    """
    PASCAL VOC 2012 segmentation wrapper.
    - train=True -> 'train', else 'val'
    - `transforms` is a callable(img, mask) -> (img, mask)
    """
    NUM_CLASSES = 21        # incl. background
    IGNORE_INDEX = 255

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transforms: Optional[Callable] = None,
                 download: bool = False,
                 year: str = "2012"):
        split = "train" if train else "val"
        self.voc = VOCSegmentationOriginal(
            root=root,
            year=year,
            image_set=split,
            download=download,
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, mask = self.voc[idx]  # PIL.Image, PIL.Image (palette)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask
