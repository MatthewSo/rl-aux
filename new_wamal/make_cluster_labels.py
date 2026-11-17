"""Fixed auxiliary labels for the `weights_only` ablation.

Embeds every training-split image with an ImageNet ResNet-50 and runs k-means
(k = psi) inside each primary class, so cluster c of class y becomes auxiliary
label y*psi + c -- the same block layout as the learned masked-softmax labels.
The result is aligned with the train subset built by new_wamal.data (same
val split / split_seed), and saved as a small int16 .npy.

    python -m new_wamal.make_cluster_labels --dataset cifar100_20 --device cuda:0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torchvision.models as tvm

from new_wamal import data as D
from new_wamal.train import Config


def kmeans(x, k, iters=50, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if len(x) <= k:
        return torch.arange(len(x)) % k
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
    return torch.cdist(x, c).argmin(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--psi", type=int, default=5)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="new_wamal/cluster_labels")
    a = ap.parse_args()

    # 224px ImageNet-normalised inputs regardless of the training resolution
    cfg = Config(dataset=a.dataset, arch="resnet50", img224=True, data_root=a.data_root)
    b = D.build(cfg)
    ds = D.Indexed(b.val.base, b.train.indices, b.train.labels)   # test-time transform, train split
    dl = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2, collate_fn=b.collate)

    net = tvm.resnet50(weights="DEFAULT"); net.fc = torch.nn.Identity()
    net = net.to(a.device).eval()
    feats = torch.zeros(len(ds), 2048)
    with torch.no_grad():
        for x, _, idx in dl:
            feats[idx] = net(x.to(a.device)).float().cpu()
    feats = torch.nn.functional.normalize(feats, dim=1)

    y = torch.as_tensor(b.train.labels)
    out = np.zeros(len(ds), dtype=np.int16)
    for c in y.unique():
        sel = (y == c).nonzero().squeeze(1)
        out[sel.numpy()] = (int(c) * a.psi + kmeans(feats[sel], a.psi)).numpy()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.dataset}_resnet50_psi{a.psi}_split{cfg.split_seed}.npy")
    np.save(path, out)
    print("wrote", path, out.shape, "clusters per class:", np.bincount(out % a.psi))


if __name__ == "__main__":
    main()
