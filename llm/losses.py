"""Focal loss and label entropy, copied from `wamal/networks/utils.py`.

Reimplemented rather than imported: that module does
`from dataset_loaders.cifar10 import CIFAR10` and imports torchvision at module
scope, which would pull the whole vision stack onto a GPU box that only needs
transformers. The arithmetic below is identical -- `llm/tests/test_losses.py`
asserts numerical equality against the original when the vision deps are
importable.
"""

from __future__ import annotations

import torch


def model_fit(x_pred, x_output, device, pri: bool = True, num_output: int = 3):
    """Focal loss over already-softmaxed probabilities. Returns per-sample loss.

    `pri=False` takes `x_output` as a soft assignment vector, which is how the
    generated auxiliary labels arrive.
    """
    if not pri:
        x_output_onehot = x_output
    else:
        x_output_onehot = torch.zeros((len(x_output), num_output), device=device,
                                      dtype=x_pred.dtype)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

    loss = x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-20)
    return torch.sum(-loss, dim=1)


def model_entropy(x_pred1):
    """Negative entropy of the batch-mean label distribution.

    Added to the phi objective to stop the label network collapsing onto one
    sub-label per primary class.
    """
    x_pred1 = torch.mean(x_pred1, dim=0)
    loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
    return torch.sum(loss1)


def scale_weights(raw: torch.Tensor, val_range: float) -> torch.Tensor:
    """Map sigmoid output in [0,1] to a multiplicative weight in [2^-r, 2^r].

    Paper equation 10: 2^(2*r*w - r).
    """
    return torch.pow(2.0, 2.0 * val_range * raw - val_range)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax restricted to the psi sub-labels belonging to the true class.

    Subtracting the per-row max before exponentiating is a departure from the
    vision implementation, which calls `torch.exp` on raw logits. That overflows
    in bf16 far sooner than it did in fp32 on VGG, and the shift is
    mathematically a no-op.
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask.bool(), neg_inf)
    masked = masked - masked.max(dim=1, keepdim=True).values
    exp = torch.exp(masked) * mask
    return exp / (exp.sum(dim=1, keepdim=True) + 1e-20)


def hierarchy_mask(y: torch.Tensor, num_primary: int, psi: int) -> torch.Tensor:
    """[B, num_primary*psi] 0/1 mask selecting sub-labels psi*y ... psi*(y+1)-1."""
    batch = y.shape[0]
    mask = torch.zeros((batch, num_primary * psi), device=y.device)
    index = torch.arange(psi, device=y.device).unsqueeze(0) + (y.unsqueeze(1) * psi)
    mask.scatter_(1, index, 1.0)
    return mask
