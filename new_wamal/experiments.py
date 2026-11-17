"""Experiment grids. Each cell is a dict of Config overrides.

Presets reproduce the hyperparameters of the original per-setup scripts in
wamal/ (e.g. vgg16_cifar_100_20_wamal.py uses lr 0.02, batch 100, 200 epochs).
"""
from __future__ import annotations

PRESETS = {
    ("cifar100_20", "vgg16"):   dict(epochs=200, batch_size=100, lr=0.02, gen_lr=1e-3),
    ("svhn", "vgg16"):          dict(epochs=200, batch_size=100, lr=0.02, gen_lr=1e-3),
    ("cifar10", "resnet50"):    dict(epochs=200, batch_size=100, lr=0.01, gen_lr=1e-3, pretrained=True),
    ("oxford_pet", "vit_b16"):  dict(epochs=75, batch_size=20, lr=5e-4, gen_lr=1e-3, pretrained=True),
    ("cub200", "vit_b16"):      dict(epochs=75, batch_size=30, lr=5e-4, gen_lr=1e-3, pretrained=True),
    ("food101", "vit_b16"):     dict(epochs=75, batch_size=30, lr=5e-4, gen_lr=1e-3, pretrained=True),
    ("cub200", "resnet50"):     dict(epochs=75, batch_size=30, lr=1e-2, gen_lr=1e-2, pretrained=True),
    # noisy-label setting: ResNet-18 from scratch, CIFAR resolution
    ("noise", "resnet18"):      dict(epochs=200, batch_size=100, lr=0.01, gen_lr=1e-3, pretrained=False),
}


def preset(dataset, arch):
    noisy = dataset.startswith(("cifar10n", "cifar100n")) or (dataset in ("cifar10", "cifar100") and arch == "resnet18")
    key = ("noise", arch) if noisy else (dataset, arch)
    return dict(PRESETS[key])


def cell(dataset, arch, method, seed, **kw):
    c = dict(dataset=dataset, arch=arch, method=method, seed=seed)
    c.update(preset(dataset, arch))
    c.update(kw)
    return c


def cluster_path(dataset):
    return f"new_wamal/cluster_labels/{dataset}_resnet50_psi5_split0.npy"


def _grid(setups, methods, seeds, **kw):
    out = []
    for (ds, arch) in setups:
        for mth in methods:
            if mth == "human" and ds != "cifar100_20":
                continue
            for s in seeds:
                extra = dict(kw)
                if mth == "weights_only":
                    extra["cluster_labels"] = cluster_path(ds)
                out.append(cell(ds, arch, mth, s, **extra))
    return out


E1_SETUPS = [("cifar100_20", "vgg16"), ("svhn", "vgg16"), ("cifar10", "resnet50"), ("oxford_pet", "vit_b16")]

EXPERIMENTS = {
    # gate: does WAMAL > MAXL survive validation-based selection?
    "E0": lambda: _grid([("cifar100_20", "vgg16")], ["single", "maxl", "wamal"], range(5)),
    # head ablation
    "E1": lambda: _grid(E1_SETUPS, ["single", "maxl", "wamal", "human", "weights_only",
                                    "random_weights", "shuffled_weights"], range(5)),
    # weighting granularity
    "E2": lambda: _grid([("cifar100_20", "vgg16"), ("oxford_pet", "vit_b16")],
                        ["global_weight", "class_weight"], range(3)),
    # noisy labels
    "E3": lambda: (
        _grid([(d, "resnet18") for d in ("cifar10n_aggre", "cifar10n_worst", "cifar100n")],
              ["single", "maxl", "wamal"], range(3))
        + [c for rate in (0.2, 0.4, 0.6) for c in
           _grid([("cifar10", "resnet18"), ("cifar100", "resnet18")], ["single", "maxl", "wamal"], range(3),
                 noise_type="sym", noise_rate=rate)]
        + _grid([("cifar10", "resnet18")], ["single", "maxl", "wamal"], range(3), noise_type="asym", noise_rate=0.4)
    ),
    # cost: small generator, and single-task given WAMAL's wall-clock (fill epochs from E0 timings)
    "E4": lambda: (
        _grid([("cifar10", "resnet50"), ("oxford_pet", "vit_b16")], ["wamal"], range(3), label_arch="resnet18")
    ),
    # data scaling on ViT
    "E5": lambda: [c for frac in (0.1, 0.3) for c in
                   _grid([("oxford_pet", "vit_b16"), ("cub200", "vit_b16")], ["single", "maxl", "wamal"],
                         range(5), train_frac=frac)]
                  + _grid([("cub200", "vit_b16")], ["single", "maxl", "wamal"], range(5)),
    # cheap checks
    "E7": lambda: (
        _grid([("cifar100_20", "vgg16")], ["wamal_nomask"], range(3))
        + _grid([("oxford_pet", "vit_b16")], ["single", "maxl", "wamal"], range(3), optimizer="adamw", lr=5e-6)
    ),
}


def cell_dir(root, exp, c):
    tags = []
    for k in ("noise_type", "noise_rate", "train_frac", "label_arch", "optimizer", "epochs_mult"):
        if k in c and c[k] not in (None, "", "none", 1.0, "sgd"):
            if k == "noise_type" or k == "noise_rate":
                continue
            tags.append(f"{k}={c[k]}")
    if c.get("noise_type", "none") != "none":
        tags.insert(0, f"{c['noise_type']}{c['noise_rate']}")
    name = "__".join([c["dataset"], c["arch"], c["method"]] + tags)
    return f"{root}/{exp}/{name}/seed{c['seed']}"
