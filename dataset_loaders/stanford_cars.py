from pathlib import Path
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import StanfordCars as StanfordCarsOriginal


class StanfordCars(Dataset):
    """Stanford Cars dataset wrapper."""
    def __init__(self, root, train=True, transform=None, download=False):
        split = "train" if train else "test"
        self.cars = StanfordCarsOriginal(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )

    def __len__(self):
        return len(self.cars)

    def __getitem__(self, idx):
        img, class_id = self.cars[idx]     # already (image, int) in torchvision
        return img, class_id
