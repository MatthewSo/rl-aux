from pathlib import Path
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet as OxfordIIITPetOriginal
from torchvision.datasets import StanfordCars as StanfordCarsOriginal
from datasets import load_dataset

class StanfordCars(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        split = "train" if train else "test"
        dataset = load_dataset("tanganke/stanford_cars")
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, class_id = self.dataset[idx]

