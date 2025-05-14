from pathlib import Path
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet as OxfordIIITPetOriginal
from torchvision.datasets import StanfordCars as StanfordCarsOriginal


class CUB200(Dataset):
    base_folder = "CUB_200_2011"

    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        self.root = Path(root).expanduser()
        self.transform = transform
        if download:
            raise NotImplementedError("Download logic not included.")

        split_file  = self.root / self.base_folder / "train_test_split.txt"
        label_file  = self.root / self.base_folder / "image_class_labels.txt"
        img_file    = self.root / self.base_folder / "images.txt"

        splits  = pd.read_csv(split_file,  sep=" ", names=["img_id", "is_train"])
        labels  = pd.read_csv(label_file,  sep=" ", names=["img_id", "target"])
        images  = pd.read_csv(img_file,   sep=" ", names=["img_id", "rel_path"])

        df = images.merge(labels).merge(splits)
        df = df[df.is_train == int(train)]

        # 0-index the class labels
        self.samples = [
            (self.root / self.base_folder / "images" / p, int(t) - 1)
            for p, t in zip(df.rel_path, df.target)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

