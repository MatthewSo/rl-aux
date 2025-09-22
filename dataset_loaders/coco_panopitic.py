# dataset_loaders/coco_panoptic.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class COCOPanopticSeg(Dataset):
    """
    COCO Panoptic segmentation (semantic) wrapper.
    Expects the standard COCO 2017 layout:
      root/
        train2017/ *.jpg
        val2017/   *.jpg
        annotations/
          panoptic_train2017/ *.png
          panoptic_val2017/   *.png
          panoptic_train2017.json
          panoptic_val2017.json
    Returns (image_tensor [3,H,W], mask_long [H,W]) with IGNORE_INDEX=255.
    """
    IGNORE_INDEX = 255

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transforms=None,
                 year: str = "2017",
                 include_things: bool = True):
        self.root = Path(root)
        self.transforms = transforms
        self.split = "train" if train else "val"
        self.year = str(year)
        self.include_things = include_things

        # Paths
        self.img_dir = self.root / f"{self.split}{self.year}"
        self.ann_dir = self.root / "annotations"
        self.panoptic_png_dir = self.ann_dir / f"panoptic_{self.split}{self.year}"
        self.panoptic_json = self.ann_dir / f"panoptic_{self.split}{self.year}.json"

        # Basic validation with helpful errors
        missing: List[str] = []
        for p in [self.img_dir, self.panoptic_png_dir, self.panoptic_json]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "COCO Panoptic files not found. Expected:\n  - " +
                "\n  - ".join(missing) +
                "\nPlace COCO 2017 images and panoptic annotations under the paths above."
            )

        data = json.loads(self.panoptic_json.read_text())

        # categories: id, name, isthing
        cats = data["categories"]
        if not include_things:
            cats = [c for c in cats if not c.get("isthing", False)]

        # category_id -> contiguous index [0..C-1]
        # Keep ordering stable by sorting on original id.
        self.catid2idx: Dict[int, int] = {
            c["id"]: i for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))
        }
        self.NUM_CLASSES = len(self.catid2idx)

        # images: id -> file_name
        images_by_id = {im["id"]: im["file_name"] for im in data["images"]}

        # Build samples: (jpg_path, png_path, {segment_id -> class_idx})
        self.samples: list[Tuple[Path, Path, Dict[int, int]]] = []
        for ann in data["annotations"]:
            img_file = images_by_id[ann["image_id"]]
            img_path = self.img_dir / img_file
            seg_png_path = self.panoptic_png_dir / ann["file_name"]
            seg_map: Dict[int, int] = {}
            for s in ann["segments_info"]:
                cls_idx = self.catid2idx.get(s["category_id"], None)
                if cls_idx is not None:
                    seg_map[s["id"]] = cls_idx
                # else: if include_things=False, thing segments will be dropped -> ignored
            self.samples.append((img_path, seg_png_path, seg_map))

    @staticmethod
    def _decode_panoptic_png(png: Image.Image) -> np.ndarray:
        """
        Convert 3‑channel PNG into a segment‑id map.
        COCO panoptic convention: id = R + 256*G + 256*256*B
        """
        arr = np.asarray(png, dtype=np.uint8)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            seg_id = arr[:, :, 0].astype(np.int32) \
                     + arr[:, :, 1].astype(np.int32) * 256 \
                     + arr[:, :, 2].astype(np.int32) * 256 * 256
        else:
            seg_id = arr.astype(np.int32)
        return seg_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, seg_png_path, seg_map = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        seg_png = Image.open(seg_png_path)
        seg_id = self._decode_panoptic_png(seg_png)

        # Build semantic mask with IGNORE for ids not in seg_map
        mask = np.full(seg_id.shape, self.IGNORE_INDEX, dtype=np.uint8)
        for sid, cls_idx in seg_map.items():
            mask[seg_id == sid] = cls_idx
        mask_img = Image.fromarray(mask, mode="L")

        if self.transforms is not None:
            img, mask_img = self.transforms(img, mask_img)

        # transforms should produce tensors; mask LongTensor
        # (Our seg_transforms already do this. If you pass torchvision Compose,
        # ensure mask->long dtype conversion.)
        if not isinstance(mask_img, torch.Tensor):
            mask_t = torch.from_numpy(np.array(mask_img, dtype=np.int64))
        else:
            mask_t = mask_img.long()

        return img, mask_t
