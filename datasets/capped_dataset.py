from collections import defaultdict
import random
from torch.utils.data import Dataset

class PerClassCap(Dataset):
    def __init__(self, base_ds, max_per_class: int = 30,
                 seed: int | None = None, keep_order: bool = False):
        super().__init__()
        self.base_ds = base_ds
        self.max_per_class = max_per_class

        rng = random.Random(seed)
        buckets = defaultdict(list)          # class_id -> [indices]

        # one pass to bucket indices by label
        for idx in range(len(base_ds)):
            _, lbl = base_ds[idx]
            buckets[int(lbl)].append(idx)

        # cap each bucket
        self.idxs = []
        for lbl, inds in buckets.items():
            if not keep_order:
                rng.shuffle(inds)
            self.idxs.extend(inds[:max_per_class])

        # (optional) shuffle across classes so dataloader order looks natural
        rng.shuffle(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base_ds[self.idxs[i]]