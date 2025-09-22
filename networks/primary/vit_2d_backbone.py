# wamal/networks/vit_dense_backbone.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from transformers import ViTModel

class ViT2DBackbone(nn.Module):
    """
    Wraps a HuggingFace ViT and returns a 4D feature map [B, C, H/P, W/P]
    reconstructed from patch tokens (CLS removed).
    Default: google/vit-base-patch16-224 -> 224/16 = 14x14 map.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.patch = self.vit.config.patch_size
        self.imgsz = self.vit.config.image_size  # assume fixed 224 like your ViT scripts

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W] normalized to ImageNet mean/std (H=W=224 recommended)
        returns: feature map [B, C, H/P, W/P]
        """
        out = self.vit(pixel_values=pixel_values)             # ModelOutput
        seq = out.last_hidden_state                           # [B, 1+N, C]
        tokens = seq[:, 1:, :]                                # drop CLS -> [B, N, C]
        B, N, C = tokens.shape
        h = w = int(math.sqrt(N))                             # expect 14 when 224/16
        assert h * w == N, "Input size must be divisible by patch size with square grid."
        fmap = tokens.transpose(1, 2).contiguous().view(B, C, h, w)  # [B, C, h, w]
        return fmap
