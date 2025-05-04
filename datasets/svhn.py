from torch.utils.data import Dataset
from torchvision.datasets import SVHN as SVHNOriginal

class SVHN(Dataset):
    def __init__(self, root, train: bool = True,
                 transform=None, download: bool = False):
        super().__init__()
        split = "train" if train else "test"

        self.svhn = SVHNOriginal(
            root=root,
            split=split,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.svhn)

    def __getitem__(self, index):
        image, class_id = self.svhn[index]
        return image, class_id