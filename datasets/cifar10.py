from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10 as CIFAR10Original

class CIFAR10(Dataset):
    """
    CIFAR10 dataset wrapper.
    """
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        self.cifar10 = CIFAR10Original(
            root=root,
            train=train,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, class_id = self.cifar10[index]
        return image, class_id