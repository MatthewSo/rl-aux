import torch

def mask_softmax(x, mask, dim=1, epsilon=1e-8):
    if x.isnan().any():
        print("input is NAN")
    # Subtract the max value from logits (x) for numerical stability
    logits_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(x - logits_max)  # Exponentiate the shifted logits

    # Apply mask to the exponentiated logits
    masked_exp_logits = exp_logits * mask

    # Ensure no division by zero by adding epsilon
    masked_sum = torch.sum(masked_exp_logits, dim=dim, keepdim=True) + epsilon

    # Normalize the masked logits by the sum along the specified dimension
    softmax_logits = masked_exp_logits / masked_sum
    if softmax_logits.isnan().any():
        print("logits is NAN")
    return softmax_logits

def create_mask_from_labels(labels, num_classes=20, num_features=100):
    # Ensure num_features is a multiple of num_classes for correct partitioning
    assert num_features % num_classes == 0, "num_features must be a multiple of num_classes"

    ones_per_class = num_features // num_classes  # This is 5 in your case

    # Initialize an empty tensor of zeros with shape (N, num_features)
    N = len(labels)
    output_tensor = torch.zeros(N, num_features)

    for i, label in enumerate(labels):
        start_idx = label * ones_per_class
        end_idx = start_idx + ones_per_class
        output_tensor[i, start_idx:end_idx] = 1

    return output_tensor