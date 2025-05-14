from pathlib import Path
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet as OxfordIIITPetOriginal
from torchvision.datasets import StanfordCars as StanfordCarsOriginal

class OxfordIIITPet(Dataset):
    """Oxford-IIIT Pet dataset wrapper."""
    def __init__(self, root, train=True, transform=None, download=False):
        split = "trainval" if train else "test"
        self.pet = OxfordIIITPetOriginal(
            root=root,
            split=split,
            target_types="category",
            transform=transform,
            download=download,
        )

    def __len__(self):
        return len(self.pet)

    def __getitem__(self, idx):
        img, category_id = self.pet[idx]   # returns a single int when target_types="category"
        return img, category_id
