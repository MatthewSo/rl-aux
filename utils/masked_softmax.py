import torch

from utils.log import log_print


def mask_softmax(x, mask, dim=1, epsilon=1e-8):
    # Returns the softmax of x, masked by mask
    if x.isnan().any():
        log_print("input is NAN")
    logits_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(x - logits_max)

    masked_exp_logits = exp_logits * mask

    masked_sum = torch.sum(masked_exp_logits, dim=dim, keepdim=True) + epsilon

    softmax_logits = masked_exp_logits / masked_sum
    if softmax_logits.isnan().any():
        log_print("logits is NAN")
    return softmax_logits

def create_mask_from_labels(labels, num_classes=20, num_features=100):
    # Makes the mask for the softmax layer
    assert num_features % num_classes == 0, "num_features must be a multiple of num_classes"

    ones_per_class = num_features // num_classes

    N = len(labels)
    output_tensor = torch.zeros(N, num_features)

    for i, label in enumerate(labels):
        start_idx = label * ones_per_class
        end_idx = start_idx + ones_per_class
        output_tensor[i, start_idx:end_idx] = 1

    return output_tensor