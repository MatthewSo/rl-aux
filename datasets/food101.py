from torch.utils.data import Dataset
from torchvision.datasets import Food101 as Food101Original

class Food101(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        split = 'train' if train else 'test'
        self.food101 = Food101Original(
            root=root,
            split=split,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.food101)

    def __getitem__(self, index):
        image, class_id = self.food101[index]
        return image, class_id