from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class CUB200(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        self.root = Path(root).expanduser()
        self.transform = transform
        self.samples = []

        kaggle_train = self.root / "train"
        kaggle_test  = self.root / "test"
        if kaggle_train.is_dir() and kaggle_test.is_dir():
            split_dir = kaggle_train if train else kaggle_test
            classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            for cls in classes:
                for img_path in (split_dir / cls).glob("*"):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.samples.append((img_path, class_to_idx[cls]))
            if self.samples:
                return  

        base = self.root / "CUB_200_2011"
        if download:
            raise NotImplementedError("Add your own download routine here.")

        split_file  = base / "train_test_split.txt"
        label_file  = base / "image_class_labels.txt"
        img_file    = base / "images.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                "Couldn’t locate train/test annotations *or* a train/-test/ folder "
                f"split in {self.root}. Check that the dataset is complete."
            )

        splits = pd.read_csv(split_file, sep=" ", names=["img_id", "is_train"])
        labels = pd.read_csv(label_file, sep=" ", names=["img_id", "target"])
        images = pd.read_csv(img_file,  sep=" ", names=["img_id", "path"])
        df = images.merge(labels).merge(splits)
        df = df[df.is_train == int(train)]

        self.samples = [
            (base / "images" / p, int(t) - 1)    # 0-index the labels
            for p, t in zip(df.path, df.target)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target
