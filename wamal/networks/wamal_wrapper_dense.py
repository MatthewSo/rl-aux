# wamal/networks/wamal_wrapper_dense.py

from __future__ import annotations
from typing import Callable, Tuple, OrderedDict as OD, Optional, Dict, Any
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
    Normalize the output across common segmentation/classification backbones.

    Returns a 4D feature map [B, F, H, W].
    """
    out = backbone(x, **kwargs)

    # HuggingFace or torchvision classification with logits - accept [B,*,*]
    if isinstance(out, ModelOutput):
        out = out.logits

    # torchvision segmentation returns dict with 'out'
    if isinstance(out, dict) and "out" in out:
        out = out["out"]

    # If the backbone returned a tuple/list, take first (common for features)
    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.dim() == 2:
        # [B, D] -> reshape to [B, D, 1, 1] so 1×1 conv heads still work
        out = out[:, :, None, None]

    if out.dim() != 4:
        raise ValueError(f"Backbone returned tensor with dim={out.dim()}, "
                         "expected 4D feature map.")

    return out


class WamalDenseWrapper(nn.Module):
    """
    Dense (segmentation) multi-head wrapper:
      - Primary head: per-pixel C logits + softmax
      - Auxiliary head: per-pixel (C * K) logits + softmax
    Heads are simple 1x1 convs on top of the backbone feature map.
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

        # Lazy build heads on first forward if feature_channels not given
        self._feat_ch = feature_channels
        if feature_channels is not None:
            self._build_heads(feature_channels)

    def _build_heads(self, ch: int):
        self.primary_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, self.num_primary, kernel_size=1, bias=True),
            nn.Softmax(dim=1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, self.num_auxiliary, kernel_size=1, bias=True),
            nn.Softmax(dim=1),
        )

    def _heads_built(self) -> bool:
        return hasattr(self, "primary_head") and hasattr(self, "auxiliary_head")

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
        """
        if params is None and buffers is None:
            feat = _get_feat_from_backbone(self.backbone, x, **kwargs)
            if not self._heads_built():
                self._build_heads(feat.shape[1])

            pri = self.primary_head(feat)
            aux = self.auxiliary_head(feat)

            # by default, upsample to input spatial size
            pri = self._maybe_upsample(pri, x.shape[-2:])
            aux = self._maybe_upsample(aux, x.shape[-2:])
            return pri, aux

        # functional_call path (for fast-weights usage)
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
            self._build_heads(feat.shape[1])

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
      - Produces per-pixel auxiliary logits with masked softmax (C*K channels).
      - Optionally produces a per-pixel weight map in [0,1] (1 channel).
    By default, K=5 sub-labels per original class.
    Mirrors the logic of LabelWeightWrapper (classification). :contentReference[oaicite:4]{index=4}
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

        self._feat_ch = None
        self._classifier_head = None
        self._weight_head = None

        # fixed class-to-subclass mapping mask [C, C*K]
        # index[i, start_i : start_i + K] = 1
        index = torch.zeros(num_primary, num_auxiliary)
        start = 0
        for i in range(num_primary):
            index[i, start:start+self.K] = 1.0
            start += self.K
        self.register_buffer("_index_mask", index, persistent=False)

    def _build_heads(self, ch: int):
        self._classifier_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, self.num_auxiliary, kernel_size=1, bias=True),
        )
        self._weight_head = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _heads_built(self) -> bool:
        return (self._classifier_head is not None) and (self._weight_head is not None)

    @staticmethod
    def _mask_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        # logits: [B, C*K, H, W], mask: same shape with 0/1 over channels per pixel
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
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          gen_labels : [B, C*K, H, W] (soft labels after masked softmax)
          weights    : [B, 1,   H, W] in [0,1]
        """
        if params is None and buffers is None:
            feat = _get_feat_from_backbone(self.backbone, x, **kwargs)
            if not self._heads_built():
                self._build_heads(feat.shape[1])

            # predict at feature resolution, then upsample heads to input size
            sizeHW = x.shape[-2:]
            feat = self._maybe_upsample_feat(feat, sizeHW)

            logits = self._classifier_head(feat)          # [B, C*K, H, W]
            weights = self._weight_head(feat)             # [B, 1,   H, W]

            # build pixel-wise mask based on GT y
            y_up = y
            if y_up.shape[-2:] != logits.shape[-2:]:
                y_up = F.interpolate(y.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").long().squeeze(1)

            B, CK, H, W = logits.shape
            index = self._index_mask.to(y_up.device)      # [C, C*K]
            mask = index[y_up.view(-1)].view(B, H, W, CK).permute(0, 3, 1, 2).contiguous()
            gen_labels = self._mask_softmax(logits, mask, dim=1)
            return gen_labels, weights

        # functional_call path for MAL
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
        index = self._index_mask.to(y_up.device)

        valid = (y_up != ignore_index)  # add 'ignore_index' as an argument defaulting to 255
        y_safe = y_up.clone()
        y_safe[~valid] = 0  # put any valid class id to avoid OOB indexing

        index = self._index_mask.to(y_up.device)         # [C, C*K]
        mask   = index[y_safe.view(-1)].view(B, H, W, CK).permute(0, 3, 1, 2).contiguous()

        # masked softmax, then zero-out ignored pixels so their aux loss is exactly 0
        gen_labels = self._mask_softmax(logits, mask, dim=1)
        gen_labels *= valid.unsqueeze(1)                 # zero everywhere invalid
        weights = self._weight_head(feat) * valid.unsqueeze(1)

        mask = index[y_up.view(-1)].view(B, H, W, CK).permute(0, 3, 1, 2).contiguous()
        gen_labels = self._mask_softmax(logits, mask, dim=1)
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
