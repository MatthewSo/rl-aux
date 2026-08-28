"""Hand-rolled LoRA carrying two adapters over one frozen base weight.

Not `peft`. WAMAL's inner step evaluates the theta-network at perturbed
parameters through `torch.func.functional_call`, which needs stable, predictable
parameter names and a module tree we fully control. `peft` wraps and renames
aggressively, and debugging its interaction with second-order gradients on a
metered GPU is not a good trade.

The two-adapters-in-one-wrapper shape exists because theta and phi must share
frozen base weights while applying different low-rank deltas. The alternatives
are worse: two wrapped module trees over the same tensors is fragile, and
merging `W + s*B@A` per forward materialises a full-size weight for every
targeted layer, which is exactly the cost LoRA exists to avoid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

THETA = "theta"
PHI = "phi"


class LoRAAdapter(nn.Module):
    """One low-rank delta: `x -> (alpha/r) * B @ A @ x`."""

    def __init__(self, in_features: int, out_features: int, r: int, alpha: int,
                 dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        # B starts at zero so the adapter is an exact no-op at init and the
        # frozen backbone's behaviour is the starting point for both networks.
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling


class DualLoRALinear(nn.Module):
    """Frozen `nn.Linear` plus one adapter per network, selected by `active`.

    `active` is a plain Python attribute, not a buffer or parameter, so switching
    it costs nothing and it never appears in a state dict or a `functional_call`
    override.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0, adapters: tuple[str, ...] = (THETA, PHI)):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.adapters = nn.ModuleDict({
            name: LoRAAdapter(base.in_features, base.out_features, r, alpha, dropout)
            for name in adapters
        })
        self.active = adapters[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        adapter = self.adapters[self.active]
        # Adapters are held in fp32 while the backbone runs in bf16. Trainable
        # parameters this small are cheap to keep at full precision, and it
        # avoids bf16's 8-bit mantissa quietly rounding away small SGD updates.
        delta = adapter(x.to(adapter.lora_A.dtype))
        return out + delta.to(out.dtype)


def inject_dual_lora(model: nn.Module, targets: tuple[str, ...] = ("q_proj", "v_proj"),
                     r: int = 16, alpha: int = 32, dropout: float = 0.0,
                     adapters: tuple[str, ...] = (THETA, PHI)) -> list[str]:
    """Replace matching `nn.Linear` submodules in place. Returns wrapped names."""
    to_wrap: list[tuple[str, nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if child_name in targets:
                full = f"{name}.{child_name}" if name else child_name
                to_wrap.append((full, module, child_name, child))

    if not to_wrap:
        available = sorted({
            child_name
            for _, module in model.named_modules()
            for child_name, child in module.named_children()
            if isinstance(child, nn.Linear)
        })
        raise ValueError(
            f"no nn.Linear matched targets {targets}. Linear names present: {available}"
        )

    for full, parent, child_name, child in to_wrap:
        setattr(parent, child_name,
                DualLoRALinear(child, r=r, alpha=alpha, dropout=dropout,
                               adapters=adapters))

    return [full for full, _, _, _ in to_wrap]


def set_active_adapter(model: nn.Module, name: str) -> None:
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            if name not in module.adapters:
                raise KeyError(f"adapter {name!r} not in {list(module.adapters)}")
            module.active = name


def adapter_parameter_names(model: nn.Module, adapter: str) -> list[str]:
    needle = f".adapters.{adapter}."
    return [n for n, _ in model.named_parameters() if needle in n]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
