"""Backbones and the label/weight generator used by new_wamal.

The primary network is the original ``wamal.networks.wamal_wrapper.WamalWrapper``
(unchanged, so results stay comparable with the paper). The generator is a
re-implementation of ``LabelWeightWrapper`` that adds the switches the
ablations need:

* ``mask``        -- hierarchical masked softmax (True, as in MAXL/WAMAL) or a
                     plain softmax over all auxiliary classes.
* ``weight_mode`` -- ``sample`` (per-sample head, WAMAL), ``class`` (one learned
                     weight per primary class), ``global`` (one learned scalar),
                     ``none`` (weight 1, i.e. MAXL).

The generator returns the *unscaled* weight u in [0, 1]; the trainer maps it to
2**(2*r*u - r), exactly as the original code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

from wamal.networks.wamal_wrapper import WamalWrapper, strip_classifier


def make_backbone(arch: str, num_primary: int, device, pretrained: bool):
    if arch == "vgg16":
        from wamal.networks.vgg_16 import SimplifiedVGG16
        return SimplifiedVGG16(device=device, num_primary_classes=num_primary)
    if arch in ("resnet18", "resnet50"):
        import torchvision.models as tvm
        fn = getattr(tvm, arch)
        weights = "DEFAULT" if pretrained else None
        return fn(weights=weights)
    if arch == "vit_b16":
        from transformers import ViTForImageClassification
        return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    raise ValueError(arch)


def _features(backbone, x):
    feat = backbone(x)
    if isinstance(feat, ModelOutput):
        feat = feat.logits
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    return feat.flatten(1)


class Generator(nn.Module):
    def __init__(self, backbone, num_primary, num_auxiliary, input_shape,
                 mask=True, weight_mode="sample"):
        super().__init__()
        self.backbone = backbone
        d = strip_classifier(self.backbone, input_shape)
        self.num_primary, self.num_auxiliary = num_primary, num_auxiliary
        self.mask, self.weight_mode = mask, weight_mode
        psi = num_auxiliary // num_primary
        m = torch.zeros(num_primary, num_auxiliary)
        for c in range(num_primary):
            m[c, c * psi:(c + 1) * psi] = 1.0
        self.register_buffer("mask_table", m)

        self.classifier_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, num_auxiliary))
        if weight_mode == "sample":
            self.weight_head = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 1), nn.Sigmoid())
        elif weight_mode == "class":
            self.class_logit = nn.Parameter(torch.zeros(num_primary))
        elif weight_mode == "global":
            self.global_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, need_labels=True):
        feat = _features(self.backbone, x)
        labels = None
        if need_labels:
            logits = self.classifier_head(feat)
            if self.mask:
                # same numerics as the original masked softmax
                e = torch.exp(logits - logits.max(dim=1, keepdim=True).values) * self.mask_table[y]
                labels = e / (e.sum(dim=1, keepdim=True) + 1e-12)
            else:
                labels = torch.softmax(logits, dim=1)
        if self.weight_mode == "sample":
            u = self.weight_head(feat).squeeze(1)
        elif self.weight_mode == "class":
            u = torch.sigmoid(self.class_logit)[y]
        elif self.weight_mode == "global":
            u = torch.sigmoid(self.global_logit).expand(x.shape[0])
        else:
            u = None
        return labels, u


def build_models(cfg, num_primary, image_shape, device):
    num_aux = num_primary * cfg.psi
    primary = WamalWrapper(make_backbone(cfg.arch, num_primary, device, cfg.pretrained),
                           num_primary=num_primary, num_auxiliary=num_aux,
                           input_shape=image_shape).to(device)
    gen = None
    m = METHODS[cfg.method]
    if m["generator"]:
        label_arch = cfg.label_arch or cfg.arch
        gen = Generator(make_backbone(label_arch, num_primary, device, cfg.pretrained),
                        num_primary, num_aux, image_shape,
                        mask=m.get("mask", True), weight_mode=m["weight_mode"]).to(device)
    return primary, gen, num_aux


# --------------------------------------------------------------------------- #
# method table
#   labels:      learned | fixed_cluster | fixed_human | none
#   weight_mode: sample | class | global | none   (generator's weight output)
#   weights:     generator | random | shuffled | one
# --------------------------------------------------------------------------- #
METHODS = {
    "single":           dict(labels="none",          generator=False, weight_mode="none",   weights="one"),
    "maxl":             dict(labels="learned",       generator=True,  weight_mode="none",   weights="one"),
    "wamal":            dict(labels="learned",       generator=True,  weight_mode="sample", weights="generator"),
    "wamal_nomask":     dict(labels="learned",       generator=True,  weight_mode="sample", weights="generator", mask=False),
    "global_weight":    dict(labels="learned",       generator=True,  weight_mode="global", weights="generator"),
    "class_weight":     dict(labels="learned",       generator=True,  weight_mode="class",  weights="generator"),
    "weights_only":     dict(labels="fixed_cluster", generator=True,  weight_mode="sample", weights="generator"),
    "random_weights":   dict(labels="learned",       generator=True,  weight_mode="none",   weights="random"),
    "shuffled_weights": dict(labels="learned",       generator=True,  weight_mode="sample", weights="shuffled"),
    "human":            dict(labels="fixed_human",   generator=False, weight_mode="none",   weights="one"),
}
