import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import interpolate_embeddings
from transformers import ViTForImageClassification, ViTImageProcessor


def get_vit():
    weights = ViT_B_16_Weights.DEFAULT
    state_dict_224 = weights.get_state_dict(progress=True)

    model = vit_b_16(weights=None, image_size=112)

    state_dict_112 = interpolate_embeddings(
        image_size=112,
        patch_size=16,
        model_state=state_dict_224,
        interpolation_mode="bicubic",
        reset_heads=False               #
    )
    model.load_state_dict(state_dict_112, strict=True)
    return model

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def vit_collate(batch):
    imgs, labels = zip(*batch)
    enc = processor(list(imgs), return_tensors="pt")
    return enc["pixel_values"], torch.tensor(labels)