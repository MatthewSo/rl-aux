"""One module owning the frozen backbone, both LoRA adapters and all four heads.

The vision code keeps the primary network and the label network as separate
objects. That does not survive a shared backbone: `torch.func.functional_call`
resolves override names relative to a single module, and with two siblings over
one base there is no well-defined root. The prefix-stripping trick in
`wamal_wrapper.py` (`k.split('backbone.', 1)[1]`) assumes a tree shape that no
longer holds.

So: one owner, two parameter groups, two optimizers, and a `mode` argument on
`forward` so `functional_call` (which invokes `__call__`) can reach both paths.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from llm.losses import hierarchy_mask, masked_softmax
from llm.networks.backbone import load_backbone, measure_feature_dim, pool
from llm.networks.lora import PHI, THETA, DualLoRALinear, inject_dual_lora, set_active_adapter


def _head(feature_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer feedforward head, matching the vision wrappers."""
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feature_dim, out_dim),
    )


class WamalLM(nn.Module):
    THETA_PREFIXES = (f".adapters.{THETA}.", "primary_head.", "auxiliary_head.")
    PHI_PREFIXES = (f".adapters.{PHI}.", "sublabel_head.", "weight_head.")

    def __init__(self, base, feature_dim: int, num_primary: int, psi: int,
                 pooling: str = "last", model_slug: str = ""):
        super().__init__()
        self.base = base
        self.feature_dim = feature_dim
        self.num_primary = num_primary
        self.psi = psi
        self.num_auxiliary = num_primary * psi
        self.pooling = pooling
        self.model_slug = model_slug

        # theta heads
        self.primary_head = _head(feature_dim, num_primary)
        self.auxiliary_head = _head(feature_dim, self.num_auxiliary)
        # phi heads
        self.sublabel_head = _head(feature_dim, self.num_auxiliary)
        self.weight_head = _head(feature_dim, 1)

        self._validate()

    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        frozen_ok = all(
            not p.requires_grad
            for name, p in self.base.named_parameters()
            if ".adapters." not in name
        )
        if not frozen_ok:
            raise RuntimeError("backbone weights are trainable; they must be frozen")

        theta = set(self.theta_parameter_names())
        phi = set(self.phi_parameter_names())
        if not theta or not phi:
            raise RuntimeError(f"empty parameter group (theta={len(theta)}, phi={len(phi)})")
        overlap = theta & phi
        if overlap:
            raise RuntimeError(f"theta and phi share parameters: {sorted(overlap)[:5]}")

        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        unassigned = trainable - theta - phi
        if unassigned:
            raise RuntimeError(
                f"trainable parameters in neither group: {sorted(unassigned)[:5]}"
            )

    def train(self, mode: bool = True) -> "WamalLM":
        """Keep the frozen backbone in eval mode even while training.

        Backbone dropout would inject noise into an already second-order
        optimization, and it would make the inner forward and the forward at
        `theta+` see different masks -- the inner step's gradient would then be
        measured against a different function than the one it is used to
        evaluate. LoRA's own dropout is left under the caller's control.
        """
        super().train(mode)
        self.base.eval()
        for module in self.base.modules():
            if isinstance(module, DualLoRALinear):
                module.adapters.train(mode)
        return self

    def _names_matching(self, prefixes: tuple[str, ...]) -> list[str]:
        return [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and any(p in name or name.startswith(p) for p in prefixes)
        ]

    def theta_parameter_names(self) -> list[str]:
        return self._names_matching(self.THETA_PREFIXES)

    def phi_parameter_names(self) -> list[str]:
        return self._names_matching(self.PHI_PREFIXES)

    def theta_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        lookup = dict(self.named_parameters())
        return [(n, lookup[n]) for n in self.theta_parameter_names()]

    def phi_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        lookup = dict(self.named_parameters())
        return [(n, lookup[n]) for n in self.phi_parameter_names()]

    def theta_parameters(self) -> list[nn.Parameter]:
        return [p for _, p in self.theta_named_parameters()]

    def phi_parameters(self) -> list[nn.Parameter]:
        return [p for _, p in self.phi_named_parameters()]

    # ------------------------------------------------------------------ #

    def _features(self, input_ids, attention_mask, adapter: str) -> torch.Tensor:
        set_active_adapter(self.base, adapter)
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = pool(hidden, attention_mask, self.pooling)
        # Heads and losses run in fp32 regardless of the backbone dtype; the
        # focal loss takes a log of a softmax and is the numerically weakest link.
        return pooled.float()

    def forward(self, input_ids, attention_mask, y=None, mode: str = "primary"):
        if mode == "primary":
            features = self._features(input_ids, attention_mask, THETA)
            primary = torch.softmax(self.primary_head(features), dim=-1)
            auxiliary = torch.softmax(self.auxiliary_head(features), dim=-1)
            return primary, auxiliary

        if mode == "label":
            if y is None:
                raise ValueError("mode='label' needs the primary labels y for the mask")
            features = self._features(input_ids, attention_mask, PHI)
            logits = self.sublabel_head(features)
            mask = hierarchy_mask(y, self.num_primary, self.psi)
            sub_labels = masked_softmax(logits, mask)
            weights = torch.sigmoid(self.weight_head(features))
            return sub_labels, weights

        raise ValueError(f"unknown mode {mode!r}")

    def forward_primary(self, input_ids, attention_mask,
                        params: OrderedDict | None = None):
        """Primary/auxiliary heads, optionally evaluated at overridden theta.

        `params` is the inner-loop `theta+`. Overrides name parameters relative
        to this module, so no prefix surgery is needed.
        """
        if params is None:
            return self(input_ids, attention_mask, mode="primary")
        from torch.func import functional_call

        return functional_call(
            self, params, (input_ids, attention_mask), {"mode": "primary"}
        )

    def forward_label(self, input_ids, attention_mask, y,
                      params: OrderedDict | None = None):
        if params is None:
            return self(input_ids, attention_mask, y=y, mode="label")
        from torch.func import functional_call

        return functional_call(
            self, params, (input_ids, attention_mask), {"y": y, "mode": "label"}
        )

    # ------------------------------------------------------------------ #

    def trainable_state_dict(self) -> dict:
        keep = set(self.theta_parameter_names()) | set(self.phi_parameter_names())
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}

    def save(self, path: str | Path, **extra) -> None:
        """Persist adapters and heads only.

        The vision wrappers save `self.state_dict()`. Here that would write the
        entire frozen backbone every epoch -- gigabytes of identical bytes, on a
        volume we have to copy off the box before teardown.
        """
        payload = {
            "trainable": self.trainable_state_dict(),
            "model_slug": self.model_slug,
            "feature_dim": self.feature_dim,
            "num_primary": self.num_primary,
            "psi": self.psi,
            "pooling": self.pooling,
            **extra,
        }
        torch.save(payload, str(path))

    def load_trainable(self, path: str | Path, map_location="cpu") -> dict:
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        if payload.get("model_slug") and payload["model_slug"] != self.model_slug:
            raise ValueError(
                f"checkpoint is for {payload['model_slug']!r}, this is {self.model_slug!r}"
            )
        missing, unexpected = self.load_state_dict(payload["trainable"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
        return payload

    def memory_report(self) -> dict[str, int]:
        base = sum(p.numel() for n, p in self.base.named_parameters()
                   if ".adapters." not in n)
        return {
            "frozen_base": base,
            "theta_trainable": sum(p.numel() for p in self.theta_parameters()),
            "phi_trainable": sum(p.numel() for p in self.phi_parameters()),
            "lora_modules": sum(1 for m in self.base.modules()
                                if isinstance(m, DualLoRALinear)),
        }


def build_model(cfg, device: torch.device) -> WamalLM:
    from llm.networks.backbone import resolve_dtype

    dtype = resolve_dtype(cfg.dtype, device)
    base, spec = load_backbone(cfg.model, dtype, device,
                               attn_implementation=cfg.attn_implementation)
    feature_dim = measure_feature_dim(base, device)

    inject_dual_lora(
        base,
        targets=tuple(cfg.lora_targets),
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )

    model = WamalLM(
        base=base,
        feature_dim=feature_dim,
        num_primary=cfg.num_primary,
        psi=cfg.psi,
        pooling=cfg.pooling,
        model_slug=spec.slug,
    )
    # Heads and adapters live in fp32; only the frozen base is low precision.
    model.primary_head.to(device=device, dtype=torch.float32)
    model.auxiliary_head.to(device=device, dtype=torch.float32)
    model.sublabel_head.to(device=device, dtype=torch.float32)
    model.weight_head.to(device=device, dtype=torch.float32)
    for module in model.base.modules():
        if isinstance(module, DualLoRALinear):
            module.adapters.to(device=device, dtype=torch.float32)
    return model
