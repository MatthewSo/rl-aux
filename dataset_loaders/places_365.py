from torch.utils.data import Dataset
from torchvision.datasets import Places365 as Places365Original

class Places365(Dataset):
    """
    Places365 dataset wrapper.
    """
    def __init__(self, root, train: bool = True,
                 transform=None, download: bool = False):
        super().__init__()

        split = "train-standard" if train else "val"

        # check if root already has dataset
        if download:
            print("Downloading Places365 dataset...")
            if not root.endswith("places365"):
                root = f"{root}/places365"
            if not os.path.exists(root):
                download = False

        self.places = Places365Original(
            root=root,
            split=split,
            small=True,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.places)

    def __getitem__(self, index):
        image, class_id = self.places[index]  
        return image, class_id