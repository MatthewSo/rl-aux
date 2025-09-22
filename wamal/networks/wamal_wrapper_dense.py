# wamal/networks/wamal_wrapper_dense.py

from __future__ import annotations
from typing import Callable, Tuple, OrderedDict as OD, Optional
from pathlib import Path
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from torch.func import functional_call


def _get_feat_from_backbone(backbone: nn.Module,
                            x: torch.Tensor,
                            **kwargs) -> torch.Tensor:
    """
    Normalize outputs across common backbones and return a 4D feature map [B, F, H, W].
    """
    out = backbone(x, **kwargs)

    # HuggingFace models (e.g., ViTModel) may return ModelOutput with logits
    if isinstance(out, ModelOutput):
        out = out.logits

    # torchvision segmentation returns dict with "out"
    if isinstance(out, dict) and "out" in out:
        out = out["out"]

    # some models return tuple/list, take first entry
    if isinstance(out, (tuple, list)):
        out = out[0]

    # classification backbone with [B, D] -> convert to [B, D, 1, 1]
    if out.dim() == 2:
        out = out[:, :, None, None]

    if out.dim() != 4:
        raise ValueError(f"Backbone returned tensor with dim={out.dim()}, expected 4D [B,C,H,W].")

    return out


class WamalDenseWrapper(nn.Module):
    """
    Dense (segmentation) multi-head wrapper:
      - Primary head: per-pixel C logits + softmax
      - Auxiliary head: per-pixel (C * K) logits + softmax
    Heads are simple 1x1/3x3 conv stacks on top of the backbone feature map.
    """
    def __init__(self,
                 backbone: nn.Module,
                 num_primary: int,
                 num_auxiliary: int,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 feature_channels: Optional[int] = None,
                 upsample_to_input: bool = True):
        super().__init__()
        self.backbone = backbone
        self.num_primary = num_primary
        self.num_auxiliary = num_auxiliary
        self.input_shape = input_shape
        self.upsample_to_input = upsample_to_input

        # Lazy-build heads on first forward so they inherit the feature device/dtype.
        self._feat_ch = feature_channels
        self.primary_head: Optional[nn.Module] = None
        self.auxiliary_head: Optional[nn.Module] = None

    def _build_heads(self, ch: int, device=None, dtype=None):
        """
        Build primary/aux heads on the same device/dtype as the feature map.
        Call with: self._build_heads(feat.shape[1], device=feat.device, dtype=feat.dtype)
        """
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.primary_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True, **kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, self.num_primary, kernel_size=1, bias=True, **kwargs),
            nn.Softmax(dim=1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True, **kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, self.num_auxiliary, kernel_size=1, bias=True, **kwargs),
            nn.Softmax(dim=1),
        )

    def _heads_built(self) -> bool:
        return (self.primary_head is not None) and (self.auxiliary_head is not None)

    def _maybe_upsample(self, y: torch.Tensor, sizeHW: Tuple[int, int]) -> torch.Tensor:
        if self.upsample_to_input and y.shape[-2:] != sizeHW:
            y = F.interpolate(y, size=sizeHW, mode="bilinear", align_corners=False)
        return y

    def forward(self,
                x: torch.Tensor,
                params: Optional[OD[str, torch.Tensor]] = None,
                buffers: Optional[OD[str, torch.Tensor]] = None,
                **kwargs):
        """
        If params/buffers are provided, do a functional call (used by MAL’s inner update).
        Returns:
          primary_probs : [B, C,   H, W]
          aux_probs     : [B, C*K, H, W]
        """
        if params is None and buffers is None:
            feat = _get_feat_from_backbone(self.backbone, x, **kwargs)
            if not self._heads_built():
                self._build_heads(feat.shape[1], device=feat.device, dtype=feat.dtype)
            else:
                # Safety in case model was moved after build
                self.primary_head.to(feat.device, dtype=feat.dtype)
                self.auxiliary_head.to(feat.device, dtype=feat.dtype)

            pri = self.primary_head(feat)  # [B, C,   h, w]
            aux = self.auxiliary_head(feat)  # [B, C*K, h, w]

            pri = self._maybe_upsample(pri, x.shape[-2:])
            aux = self._maybe_upsample(aux, x.shape[-2:])
            return pri, aux

        # functional_call path (fast-weights)
        params = params or {}
        buffers = buffers or {}
        merged = {**params, **buffers}

        # Backbone overrides
        bb_ov = {k.split('backbone.', 1)[1]: v for k, v in merged.items()
                 if k.startswith('backbone.')}
        feat = functional_call(self.backbone, bb_ov, (x,), kwargs)
        if isinstance(feat, ModelOutput):
            feat = feat.logits
        if isinstance(feat, dict) and "out" in feat:
            feat = feat["out"]
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if not self._heads_built():
            self._build_heads(feat.shape[1], device=feat.device, dtype=feat.dtype)
        else:
            self.primary_head.to(feat.device, dtype=feat.dtype)
            self.auxiliary_head.to(feat.device, dtype=feat.dtype)

        # Primary head overrides
        pri_ov = {k.split('primary_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('primary_head.')}
        pri = functional_call(self.primary_head, pri_ov, (feat,))

        # Auxiliary head overrides
        aux_ov = {k.split('auxiliary_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('auxiliary_head.')}
        aux = functional_call(self.auxiliary_head, aux_ov, (feat,))

        pri = self._maybe_upsample(pri, x.shape[-2:])
        aux = self._maybe_upsample(aux, x.shape[-2:])
        return pri, aux

    def save(self, path: str | Path, **extra):
        ckpt = {
            "state_dict": self.state_dict(),
            "num_primary": self.num_primary,
            "num_auxiliary": self.num_auxiliary,
            "input_shape": self.input_shape,
            "upsample_to_input": self.upsample_to_input,
            "backbone_module": self.backbone.__class__.__module__,
            "backbone_name": self.backbone.__class__.__name__,
            "extra": extra,
        }
        torch.save(ckpt, Path(path))

    @classmethod
    def load(cls,
             path: str | Path,
             backbone_fn: Optional[Callable[[], nn.Module]] = None,
             map_location: Optional[str | torch.device] = None) -> "WamalDenseWrapper":
        ckpt = torch.load(path, map_location=map_location)
        if backbone_fn is not None:
            backbone = backbone_fn()
        else:
            mod = importlib.import_module(ckpt["backbone_module"])
            cls_ = getattr(mod, ckpt["backbone_name"])
            backbone = cls_()

        model = cls(backbone=backbone,
                    num_primary=ckpt["num_primary"],
                    num_auxiliary=ckpt["num_auxiliary"],
                    input_shape=tuple(ckpt["input_shape"]),
                    upsample_to_input=ckpt.get("upsample_to_input", True))
        model.load_state_dict(ckpt["state_dict"])
        return model


class LabelWeightDenseWrapper(nn.Module):
    """
    Dense (segmentation) label generator φ:
      - Produces per-pixel auxiliary probabilities with masked softmax (C*K channels).
      - Produces a per-pixel weight map in [0,1] (1 channel).
    By default, K=5 sub-labels per original class.
    Mirrors the logic of classification LabelWeightWrapper, adapted per pixel.
    """
    def __init__(self,
                 backbone: nn.Module,
                 num_primary: int,
                 num_auxiliary: int,   # should be num_primary * K
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 upsample_to_input: bool = True):
        super().__init__()
        assert num_auxiliary % num_primary == 0, \
            "num_auxiliary must be a multiple of num_primary (K sub-labels per class)."
        self.backbone = backbone
        self.num_primary = num_primary
        self.num_auxiliary = num_auxiliary
        self.K = num_auxiliary // num_primary
        self.input_shape = input_shape
        self.upsample_to_input = upsample_to_input

        self._classifier_head: Optional[nn.Module] = None
        self._weight_head: Optional[nn.Module] = None

        # fixed class-to-subclass mapping mask [C, C*K] with contiguous K slots
        index = torch.zeros(num_primary, num_auxiliary)
        start = 0
        for _ in range(num_primary):
            index[:, start:start+self.K]  # just to clarify span; fill below
            start += self.K
        # fill mask
        start = 0
        for i in range(num_primary):
            index[i, start:start+self.K] = 1.0
            start += self.K
        self.register_buffer("_index_mask", index, persistent=False)

    def _build_heads(self, ch: int, device=None, dtype=None):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype

        self._classifier_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True, **kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, self.num_auxiliary, kernel_size=1, bias=True, **kwargs),
        )
        self._weight_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True, **kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, kernel_size=1, bias=True, **kwargs),
            nn.Sigmoid(),
        )

    def _heads_built(self) -> bool:
        return (self._classifier_head is not None) and (self._weight_head is not None)

    @staticmethod
    def _mask_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        # logits: [B, C*K, H, W], mask: [B, C*K, H, W] with 0/1 entries
        exp = torch.exp(logits) * mask
        denom = exp.sum(dim=dim, keepdim=True).clamp_min_(1e-12)
        return exp / denom

    def _maybe_upsample_feat(self, feat: torch.Tensor, sizeHW: Tuple[int, int]) -> torch.Tensor:
        if self.upsample_to_input and feat.shape[-2:] != sizeHW:
            feat = F.interpolate(feat, size=sizeHW, mode="bilinear", align_corners=False)
        return feat

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,     # ground-truth labels [B, H, W] (long)
                params: Optional[OD[str, torch.Tensor]] = None,
                buffers: Optional[OD[str, torch.Tensor]] = None,
                ignore_index: int = 255,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          gen_labels : [B, C*K, H, W] (soft labels after masked softmax)
          weights    : [B, 1,   H, W] in [0,1]
        """
        if params is None and buffers is None:
            feat = _get_feat_from_backbone(self.backbone, x, **kwargs)
            if not self._heads_built():
                self._build_heads(feat.shape[1], device=feat.device, dtype=feat.dtype)
            else:
                self._classifier_head.to(feat.device, dtype=feat.dtype)
                self._weight_head.to(feat.device, dtype=feat.dtype)

            # predict at feature resolution, then (optionally) upsample to input size
            sizeHW = x.shape[-2:]
            feat = self._maybe_upsample_feat(feat, sizeHW)

            logits  = self._classifier_head(feat)          # [B, C*K, H, W]
            weights = self._weight_head(feat)              # [B, 1,   H, W]

            # build pixel-wise mask based on GT y (handle ignore_index safely)
            y_up = y
            if y_up.shape[-2:] != logits.shape[-2:]:
                y_up = F.interpolate(y.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").long().squeeze(1)

            B, CK, H, W = logits.shape
            C = self.num_primary
            index = self._index_mask.to(y_up.device)       # [C, C*K]

            valid  = (y_up != ignore_index)
            y_safe = y_up.clone()
            y_safe[~valid] = 0        # safe class id for ignored pixels
            y_safe.clamp_(min=0, max=C-1)

            mask = index[y_safe.view(-1)].view(B, H, W, CK).permute(0, 3, 1, 2).contiguous()

            gen_labels = self._mask_softmax(logits, mask, dim=1)
            gen_labels = gen_labels * valid.unsqueeze(1)   # zero out ignored pixels
            weights    = weights * valid.unsqueeze(1)

            return gen_labels, weights

        # -------- functional_call path (for MAL) --------
        params = params or {}
        buffers = buffers or {}
        merged = {**params, **buffers}

        bb_ov = {k.split('backbone.', 1)[1]: v for k, v in merged.items()
                 if k.startswith('backbone.')}
        feat = functional_call(self.backbone, bb_ov, (x,), kwargs)
        if isinstance(feat, ModelOutput):
            feat = feat.logits
        if isinstance(feat, dict) and "out" in feat:
            feat = feat["out"]
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        sizeHW = x.shape[-2:]
        if feat.shape[-2:] != sizeHW and self.upsample_to_input:
            feat = F.interpolate(feat, size=sizeHW, mode="bilinear", align_corners=False)

        # heads
        if not self._heads_built():
            self._build_heads(feat.shape[1], device=feat.device, dtype=feat.dtype)
        else:
            self._classifier_head.to(feat.device, dtype=feat.dtype)
            self._weight_head.to(feat.device, dtype=feat.dtype)

        cls_ov = {k.split('_classifier_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('_classifier_head.')}
        logits = functional_call(self._classifier_head, cls_ov, (feat,))

        w_ov = {k.split('_weight_head.', 1)[1]: v for k, v in merged.items()
                if k.startswith('_weight_head.')}
        weights = functional_call(self._weight_head, w_ov, (feat,))

        y_up = y
        if y_up.shape[-2:] != logits.shape[-2:]:
            y_up = F.interpolate(y.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").long().squeeze(1)

        B, CK, H, W = logits.shape
        C = self.num_primary
        index = self._index_mask.to(y_up.device)

        valid  = (y_up != ignore_index)
        y_safe = y_up.clone()
        y_safe[~valid] = 0
        y_safe.clamp_(min=0, max=C-1)

        mask = index[y_safe.view(-1)].view(B, H, W, CK).permute(0, 3, 1, 2).contiguous()

        gen_labels = self._mask_softmax(logits, mask, dim=1)
        gen_labels = gen_labels * valid.unsqueeze(1)
        weights    = weights * valid.unsqueeze(1)

        return gen_labels, weights

    def save(self, path: str | Path, **extra):
        ckpt = {
            "state_dict": self.state_dict(),
            "num_primary": self.num_primary,
            "num_auxiliary": self.num_auxiliary,
            "input_shape": self.input_shape,
            "upsample_to_input": self.upsample_to_input,
            "backbone_module": self.backbone.__class__.__module__,
            "backbone_name": self.backbone.__class__.__name__,
            "extra": extra,
        }
        torch.save(ckpt, Path(path))

    @classmethod
    def load(cls,
             path: str | Path,
             backbone_fn: Optional[Callable[[], nn.Module]] = None,
             map_location: Optional[str | torch.device] = None) -> "LabelWeightDenseWrapper":
        ckpt = torch.load(path, map_location=map_location)
        if backbone_fn is not None:
            backbone = backbone_fn()
        else:
            mod = importlib.import_module(ckpt["backbone_module"])
            cls_ = getattr(mod, ckpt["backbone_name"])
            backbone = cls_()

        model = cls(backbone=backbone,
                    num_primary=ckpt["num_primary"],
                    num_auxiliary=ckpt["num_auxiliary"],
                    input_shape=tuple(ckpt["input_shape"]),
                    upsample_to_input=ckpt.get("upsample_to_input", True))
        model.load_state_dict(ckpt["state_dict"])
        return model
