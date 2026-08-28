"""Determinism helpers.

The vision code samples few-shot subsets with an unseeded `random.sample`, so
two arms at the same nominal `k` train on different examples and the comparison
between them is not clean. Everything stochastic here routes through an explicit
seed instead.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derived_seed(*parts: object) -> int:
    """A stable 31-bit seed from arbitrary run coordinates.

    Used to decorrelate the data subset from weight init: the few-shot sampler
    keys off (seed, dataset, k) so that all four arms in a cell see byte-identical
    training data, while each cell still gets a different subset.
    """
    text = "|".join(str(p) for p in parts)
    digest = 0
    for ch in text:
        digest = (digest * 31 + ord(ch)) & 0x7FFFFFFF
    return digest


def dataloader_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
