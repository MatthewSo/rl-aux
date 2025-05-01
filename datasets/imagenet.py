from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as ImageNetOriginal

class ImageNet(Dataset):
    def __init__(self, root, train: bool = True,
                 transform=None,):
        super().__init__()

        split = "train" if train else "val"
        self.imagenet = ImageNetOriginal(
            root=root,
            split=split,
            transform=transform
        )

    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, index):
        image, class_id = self.imagenet[index]
        return image, class_id