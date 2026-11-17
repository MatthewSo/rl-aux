"""Datasets for new_wamal.

Every training sample is returned as ``(x, y, i)`` where ``i`` is the sample's
position in the *training subset* (0..N-1). That index is what per-sample
quantities (fixed cluster labels, random weights, noise masks, weight dumps)
are keyed on.

The validation split is carved out of the official training set, stratified,
with a split seed that is fixed across runs (``split_seed``), so every method
and every training seed sees the same train/val partition. Model selection
uses validation accuracy; the test set is only ever evaluated, never selected
on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from dataset_loaders.transforms import (
    cifar_trans_train, cifar_trans_test,
    cifar_256_train_transform, cifar_256_test_transform,
    common_train_tf, common_test_tf,
)

# --------------------------------------------------------------------------- #
# label access without decoding images
# --------------------------------------------------------------------------- #

def _labels_of(ds) -> np.ndarray:
    for attr in ("targets", "labels", "_labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr), dtype=np.int64)
    if hasattr(ds, "samples"):
        return np.asarray([t for _, t in ds.samples], dtype=np.int64)
    raise AttributeError(f"cannot read labels of {type(ds)}")


class _CIFAR100Raw(Dataset):
    """CIFAR-100 from the python pickle, exposing coarse and fine labels."""

    def __init__(self, root, train, transform):
        import pickle
        tv = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        path = os.path.join(root, "cifar-100-python", "train" if train else "test")
        with open(path, "rb") as f:
            info = pickle.load(f, encoding="latin1")
        self.data = tv.data
        self.fine = np.asarray(info["fine_labels"], dtype=np.int64)
        self.coarse = np.asarray(info["coarse_labels"], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get(self, i):
        from PIL import Image
        img = Image.fromarray(self.data[i])
        return self.transform(img) if self.transform else img


class _TVImage(Dataset):
    """Adapter so every base dataset exposes ``get(i) -> image``."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def get(self, i):
        return self.ds[i][0]


# --------------------------------------------------------------------------- #
# the dataset actually handed to the DataLoader
# --------------------------------------------------------------------------- #

class Indexed(Dataset):
    def __init__(self, base, indices, labels):
        self.base = base                    # has .get(i)
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels, dtype=np.int64)   # aligned with indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base.get(int(self.indices[i])), int(self.labels[i]), i


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #

# Standard asymmetric CIFAR-10 flips (Patrini et al. 2017):
# truck->automobile, bird->airplane, deer->horse, cat<->dog
_ASYM_C10 = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}


def corrupt(labels: np.ndarray, num_classes: int, kind: str, rate: float, seed: int):
    rng = np.random.RandomState(seed)
    noisy = labels.copy()
    flip = rng.rand(len(labels)) < rate
    if kind == "sym":
        # uniform over the *other* classes
        off = rng.randint(1, num_classes, size=len(labels))
        noisy[flip] = (labels[flip] + off[flip]) % num_classes
    elif kind == "asym":
        if num_classes == 10:
            for i in np.where(flip)[0]:
                noisy[i] = _ASYM_C10.get(int(labels[i]), labels[i])
        else:  # CIFAR-100: next class within the same superclass is not
               # recoverable from fine labels alone; use the common "next class" flip
            noisy[flip] = (labels[flip] + 1) % num_classes
    else:
        raise ValueError(kind)
    return noisy


CIFARN_URL = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/{}"


def cifarn_labels(root, which: str):
    """which: aggre | worst | rand1 (CIFAR-10N) or noisy100 (CIFAR-100N).

    The label files are not downloaded automatically: fetch them yourself from
    CIFARN_URL into <data_root>/cifarn/. They are loaded with weights_only=True
    so a tampered file cannot execute code.
    """
    fname = "CIFAR-100_human.pt" if which == "noisy100" else "CIFAR-10_human.pt"
    path = os.path.join(root, "cifarn", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; download {CIFARN_URL.format(fname)} there first")
    import numpy.core.multiarray as _ma
    with torch.serialization.safe_globals([_ma._reconstruct, np.ndarray, np.dtype,
                                           type(np.dtype(np.int64))]):
        d = torch.load(path, weights_only=True)
    key = {"aggre": "aggre_label", "worst": "worse_label", "rand1": "random_label1",
           "noisy100": "noisy_label"}[which]
    return np.asarray(d[key], dtype=np.int64)


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #

@dataclass
class DataBundle:
    train: Indexed
    val: Indexed
    test: Indexed
    num_primary: int
    image_shape: tuple
    collate: object = None
    fine_labels: np.ndarray | None = None      # aligned with train subset (human arm)
    clean_labels: np.ndarray | None = None     # aligned with train subset (noise runs)
    noise_mask: np.ndarray | None = None       # True where the train label is wrong
    meta: dict = field(default_factory=dict)


def _stratified_split(labels, val_frac, seed):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * val_frac))
        va.extend(idx[:k]); tr.extend(idx[k:])
    return np.sort(np.array(tr)), np.sort(np.array(va))


def _per_class_fraction(idx, labels, frac, seed):
    if frac >= 1.0:
        return idx
    rng = np.random.RandomState(seed + 1)
    keep = []
    for c in np.unique(labels[idx]):
        cidx = idx[labels[idx] == c]
        rng.shuffle(cidx)
        keep.extend(cidx[:max(1, int(round(len(cidx) * frac)))])
    return np.sort(np.array(keep))


def vit_collate_factory():
    from transformers import ViTImageProcessor
    proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def collate(batch):
        imgs, ys, idx = zip(*batch)
        imgs = [im.convert("RGB") for im in imgs]
        px = proc(imgs, return_tensors="pt")["pixel_values"]
        return px, torch.tensor(ys), torch.tensor(idx)
    return collate


def build(cfg) -> DataBundle:
    root = cfg.data_root
    vit = cfg.arch.startswith("vit")
    big = vit or cfg.img224

    # ---- pick transforms ---------------------------------------------------
    if vit:                       # PIL in, HF processor in the collate fn
        tr_tf = te_tf = None
    elif cfg.dataset.startswith(("cifar", "svhn")) and not big:
        tr_tf, te_tf = cifar_trans_train, cifar_trans_test
    elif cfg.dataset.startswith(("cifar", "svhn")):
        tr_tf, te_tf = cifar_256_train_transform, cifar_256_test_transform
    else:
        tr_tf, te_tf = common_train_tf, common_test_tf
    if cfg.dataset.startswith("svhn") and not big:
        # SVHN: no horizontal flip
        tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.ToTensor(),
                           T.Normalize((0.5,) * 3, (0.2,) * 3)])

    fine = None
    ds = cfg.dataset
    # ---- base datasets (train base twice: train- and test-time transforms) -
    if ds == "cifar100_20" or ds.startswith("cifar100"):
        b_tr = _CIFAR100Raw(root, True, tr_tf)
        b_ev = _CIFAR100Raw(root, True, te_tf)
        b_te = _CIFAR100Raw(root, False, te_tf)
        if ds == "cifar100_20":
            y_all, y_test, n_cls = b_tr.coarse, b_te.coarse, 20
            fine = b_tr.fine
        else:                    # cifar100 (fine, 100-way), cifar100n, cifar100_sym..
            y_all, y_test, n_cls = b_tr.fine, b_te.fine, 100
    elif ds.startswith("cifar10"):
        mk = lambda tr, tf: _TVImage(torchvision.datasets.CIFAR10(root, train=tr, transform=tf, download=True))
        b_tr, b_ev, b_te = mk(True, tr_tf), mk(True, te_tf), mk(False, te_tf)
        y_all, y_test, n_cls = _labels_of(b_tr.ds), _labels_of(b_te.ds), 10
    elif ds == "svhn":
        mk = lambda sp, tf: _TVImage(torchvision.datasets.SVHN(os.path.join(root, "svhn"), split=sp, transform=tf, download=True))
        b_tr, b_ev, b_te = mk("train", tr_tf), mk("train", te_tf), mk("test", te_tf)
        y_all, y_test, n_cls = _labels_of(b_tr.ds), _labels_of(b_te.ds), 10
    elif ds == "oxford_pet":
        mk = lambda sp, tf: _TVImage(torchvision.datasets.OxfordIIITPet(os.path.join(root, "oxford_pet"), split=sp, target_types="category", transform=tf, download=True))
        b_tr, b_ev, b_te = mk("trainval", tr_tf), mk("trainval", te_tf), mk("test", te_tf)
        y_all, y_test, n_cls = _labels_of(b_tr.ds), _labels_of(b_te.ds), 37
    elif ds == "food101":
        mk = lambda sp, tf: _TVImage(torchvision.datasets.Food101(os.path.join(root, "food101"), split=sp, transform=tf, download=True))
        b_tr, b_ev, b_te = mk("train", tr_tf), mk("train", te_tf), mk("test", te_tf)
        y_all, y_test, n_cls = _labels_of(b_tr.ds), _labels_of(b_te.ds), 101
    elif ds == "cub200":
        from dataset_loaders.cub200 import CUB200
        mk = lambda tr, tf: _TVImage(CUB200(os.path.join(root, "cub200"), train=tr, transform=tf))
        b_tr, b_ev, b_te = mk(True, tr_tf), mk(True, te_tf), mk(False, te_tf)
        y_all, y_test, n_cls = _labels_of(b_tr.ds), _labels_of(b_te.ds), 200
    else:
        raise ValueError(f"unknown dataset {ds}")

    clean_all = y_all.copy()
    # ---- label noise (training labels only; test stays clean) --------------
    if ds in ("cifar10n_aggre", "cifar10n_worst", "cifar10n_rand1"):
        y_all = cifarn_labels(root, ds.split("_")[1])
    elif ds == "cifar100n":
        y_all = cifarn_labels(root, "noisy100")
    if cfg.noise_type != "none":
        y_all = corrupt(y_all, n_cls, cfg.noise_type, cfg.noise_rate, seed=cfg.split_seed)

    # ---- split -------------------------------------------------------------
    tr_idx, va_idx = _stratified_split(y_all, cfg.val_frac, cfg.split_seed)
    tr_idx = _per_class_fraction(tr_idx, y_all, cfg.train_frac, cfg.split_seed)

    bundle = DataBundle(
        train=Indexed(b_tr, tr_idx, y_all[tr_idx]),
        val=Indexed(b_ev, va_idx, y_all[va_idx]),
        test=Indexed(b_te, np.arange(len(y_test)), y_test),
        num_primary=n_cls,
        image_shape=(3, 224, 224) if big else (3, 32, 32),
        collate=vit_collate_factory() if vit else None,
        fine_labels=None if fine is None else fine[tr_idx],
        clean_labels=clean_all[tr_idx],
        noise_mask=(clean_all[tr_idx] != y_all[tr_idx]),
        meta=dict(n_train=len(tr_idx), n_val=len(va_idx), n_test=len(y_test),
                  noise_rate_actual=float((clean_all[tr_idx] != y_all[tr_idx]).mean())),
    )
    return bundle
