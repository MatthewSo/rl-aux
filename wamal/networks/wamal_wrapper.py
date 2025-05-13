from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

class WamalWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 num_primary: int,
                 num_auxiliary: int,
                 input_shape: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = self._strip_classifier(self.backbone, input_shape)
        self.num_primary = num_primary
        self.num_auxiliary = num_auxiliary

        self.primary_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, num_primary),
            nn.Softmax(dim=1),
        )

        self.auxiliary_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, num_auxiliary),
            nn.Softmax(dim=1),
        )

    def _strip_classifier(self, model: nn.Module, input_shape):
        # torchvision ResNet / RegNet / EfficientNet (fc)
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            dim = model.fc.in_features
            model.fc = nn.Identity()
            return dim

        # VGG / MobileNet / DenseNet / ViT (classifier)
        if hasattr(model, 'classifier'):
            cls = model.classifier
            # Linear
            if isinstance(cls, nn.Linear):
                dim = cls.in_features
                model.classifier = nn.Identity()
                return dim
            # Sequential – last layer Linear
            if isinstance(cls, nn.Sequential):
                last = list(cls.children())[-1]
                if isinstance(last, nn.Linear):
                    dim = last.in_features
                    model.classifier = nn.Identity()
                    return dim

        # Fallback: run dummy input to measure
        print("Warning: Unable to strip classifier from model. Using dummy input to measure feature dimension.")
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            feat = model(dummy)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.flatten(1)
        return feat.shape[1]

    def forward(self,
                x: torch.Tensor,
                params:  OrderedDict | None = None,
                buffers: OrderedDict | None = None,
                **kwargs):
        if params is None and buffers is None:
            feat = self.backbone(x, **kwargs)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.flatten(1)
            return (
                self.primary_head(feat),
                self.auxiliary_head(feat),
            )

        params  = params or OrderedDict()
        buffers = buffers or OrderedDict()
        merged  = {**params, **buffers}

        # Backbone overrides
        bb_ov = {k.split('backbone.', 1)[1]: v for k, v in merged.items()
                 if k.startswith('backbone.')}
        feat = functional_call(self.backbone, bb_ov, (x,), kwargs)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.flatten(1)

        # Primary head overrides
        pri_ov = {k.split('primary_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('primary_head.')}
        primary_logits = functional_call(self.primary_head, pri_ov, (feat,))

        # Auxiliary head overrides
        aux_ov = {k.split('auxiliary_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('auxiliary_head.')}
        auxiliary_logits = functional_call(self.auxiliary_head, aux_ov, (feat,))

        return primary_logits, auxiliary_logits


class LabelWeightWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 num_primary: int,
                 num_auxiliary: int,
                 input_shape: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = self._strip_classifier(self.backbone, input_shape)
        self.num_primary = num_primary
        self.num_auxiliary = num_auxiliary
        self.psi = np.array([num_auxiliary // num_primary] * num_primary)

        self.classifier_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.num_auxiliary),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )

    # ---------------- utilities --------------------------------------
    def _strip_classifier(self, model: nn.Module, input_shape):
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            dim = model.fc.in_features
            model.fc = nn.Identity()
            return dim
        if hasattr(model, 'classifier'):
            cls = model.classifier
            if isinstance(cls, nn.Linear):
                dim = cls.in_features
                model.classifier = nn.Identity()
                return dim
            if isinstance(cls, nn.Sequential):
                last = list(cls.children())[-1]
                if isinstance(last, nn.Linear):
                    dim = last.in_features
                    model.classifier = nn.Identity()
                    return dim
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            feat = model(dummy)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.flatten(1)
        return feat.shape[1]

    @staticmethod
    def _mask_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1):
        exp = torch.exp(logits) * mask
        return exp / (exp.sum(dim=dim, keepdim=True) + 1e-12)

    def _build_mask(self, y: torch.Tensor):
        index = torch.zeros(self.num_primary, self.num_auxiliary, device=y.device)
        start = 0
        for i, k in enumerate(self.psi):
            index[i, start:start+k] = 1.0
            start += k
        return index[y]

    # ---------------- forward ----------------------------------------
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                params:  OrderedDict | None = None,
                buffers: OrderedDict | None = None,
                **kwargs):
        # ---- stateful -------------------------------------------------
        if params is None and buffers is None:
            feat = self.backbone(x, **kwargs)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.flatten(1)
            logits  = self.classifier_head(feat)
            mask    = self._build_mask(y)
            labels  = self._mask_softmax(logits, mask)
            weights = self.weight_head(feat)
            return labels, weights

        # ---- stateless ------------------------------------------------
        params  = params or OrderedDict()
        buffers = buffers or OrderedDict()
        merged  = {**params, **buffers}

        # backbone overrides
        bb_ov = {k.split('backbone.',1)[1]:v for k,v in merged.items() if k.startswith('backbone.')}
        feat = functional_call(self.backbone, bb_ov, (x,), kwargs)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.flatten(1)

        # classifier head overrides
        cls_ov = {k.split('classifier_head.',1)[1]:v for k,v in merged.items() if k.startswith('classifier_head.')}
        logits = functional_call(self.classifier_head, cls_ov, (feat,))

        # weight head overrides
        w_ov = {k.split('weight_head.',1)[1]:v for k,v in merged.items() if k.startswith('weight_head.')}
        weights = functional_call(self.weight_head, w_ov, (feat,))

        mask   = self._build_mask(y)
        labels = self._mask_softmax(logits, mask)
        return labels, weights