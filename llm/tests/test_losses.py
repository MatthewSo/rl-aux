"""Numerical equality with `wamal/networks/utils.py` and `wamal_wrapper.py`.

`llm/losses.py` is a reimplementation, not an import, because the originals pull
`dataset_loaders.cifar10` and torchvision in at module scope. That makes the
claim "the arithmetic is identical" load-bearing and, until this file existed,
unverified -- the module docstring asserted a test that was never written.

The reference expressions below are transcribed literally from those two modules.
Transcription rather than import: the point is to catch our version drifting, and
importing would drag the vision stack onto a GPU box that has no use for it.

    python -m llm.tests.test_losses
"""

from __future__ import annotations

import traceback

import torch

from llm.losses import (
    hierarchy_mask,
    masked_softmax,
    model_entropy,
    model_fit,
    scale_weights,
)

DEV = torch.device("cpu")


# --- literal transcriptions of the reference ------------------------------- #

def ref_model_fit(x_pred, x_output, device, pri=True, num_output=3):
    """wamal/networks/utils.py, unchanged."""
    if not pri:
        x_output_onehot = x_output
    else:
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
    loss = x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-20)
    return torch.sum(-loss, dim=1)


def ref_model_entropy(x_pred1):
    """wamal/networks/utils.py, unchanged."""
    x_pred1 = torch.mean(x_pred1, dim=0)
    loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
    return torch.sum(loss1)


def ref_mask_softmax(logits, mask, dim=1):
    """LabelWeightWrapper._mask_softmax, unchanged (no max subtraction)."""
    exp = torch.exp(logits) * mask
    return exp / (exp.sum(dim=dim, keepdim=True) + 1e-12)


def ref_build_mask(y, num_primary, psi):
    """LabelWeightWrapper._build_mask, unchanged."""
    num_auxiliary = num_primary * psi
    index = torch.zeros(num_primary, num_auxiliary, device=y.device)
    start = 0
    for i in range(num_primary):
        index[i, start:start + psi] = 1.0
        start += psi
    return index[y]


def ref_weight_factors(aux_weight, val_range):
    """train_network.py, unchanged."""
    return torch.pow(2.0, (val_range * 2) * aux_weight - val_range)


# --------------------------------------------------------------------------- #


def test_focal_loss_matches_the_reference_on_primary_labels():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(16, 10), dim=1)
    labels = torch.randint(0, 10, (16,))

    ours = model_fit(probs, labels, DEV, pri=True, num_output=10)
    ref = ref_model_fit(probs, labels, DEV, pri=True, num_output=10)
    assert torch.equal(ours, ref), (ours - ref).abs().max().item()
    print(f"  16x10 primary: identical, max loss {ours.max():.6f}")


def test_focal_loss_matches_the_reference_on_soft_auxiliary_targets():
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(16, 150), dim=1)
    soft = torch.softmax(torch.randn(16, 150), dim=1)

    ours = model_fit(probs, soft, DEV, pri=False, num_output=150)
    ref = ref_model_fit(probs, soft, DEV, pri=False, num_output=150)
    assert torch.equal(ours, ref), (ours - ref).abs().max().item()
    print(f"  16x150 soft targets: identical, mean loss {ours.mean():.6f}")


def test_focal_loss_is_the_squared_focal_form_not_cross_entropy():
    """Guards the (1-p)^2 factor, which is easy to drop and hard to notice."""
    probs = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    labels = torch.tensor([0, 0])
    ours = model_fit(probs, labels, DEV, pri=True, num_output=2)
    expected = -torch.tensor([(1 - 0.9) ** 2 * torch.log(torch.tensor(0.9)),
                              (1 - 0.5) ** 2 * torch.log(torch.tensor(0.5))])
    assert torch.allclose(ours, expected, atol=1e-6), (ours, expected)
    # Plain cross-entropy would give these instead; they must differ.
    ce = -torch.log(probs[:, 0])
    assert not torch.allclose(ours, ce, atol=1e-3), "focal factor is missing"
    print(f"  focal {ours.tolist()} vs cross-entropy {ce.tolist()}")


def test_entropy_matches_and_rewards_a_uniform_marginal():
    torch.manual_seed(2)
    labels = torch.softmax(torch.randn(32, 150), dim=1)
    ours, ref = model_entropy(labels), ref_model_entropy(labels)
    assert torch.equal(ours, ref), (ours - ref).abs().item()

    # Sign check: the objective adds +factor*entropy, so a collapsed marginal
    # must score *higher* than a uniform one for the term to push toward
    # diversity. Getting this backwards would silently encourage collapse.
    uniform = torch.full((8, 15), 1.0 / 15)
    collapsed = torch.zeros(8, 15)
    collapsed[:, 3] = 1.0
    assert model_entropy(collapsed) > model_entropy(uniform), (
        "entropy term does not penalise collapse"
    )
    print(f"  identical; uniform {model_entropy(uniform):.4f} < "
          f"collapsed {model_entropy(collapsed):.4f}, so minimising spreads mass")


def test_weight_scaling_matches_the_reference_range():
    raw = torch.linspace(0.0, 1.0, 11)
    for val_range in (1.0, 5.0):
        ours = scale_weights(raw, val_range)
        ref = ref_weight_factors(raw, val_range)
        assert torch.allclose(ours, ref, atol=0, rtol=0), (ours - ref).abs().max()
    # r=5 maps [0,1] onto [2^-5, 2^5]; a sigmoid at init sits at 1.0.
    edge = scale_weights(torch.tensor([0.0, 0.5, 1.0]), 5.0)
    assert torch.allclose(edge, torch.tensor([2.0 ** -5, 1.0, 2.0 ** 5]))
    print(f"  identical; r=5 gives {edge.tolist()}")


def test_masked_softmax_matches_the_reference_where_it_is_stable():
    """Ours subtracts the row max; that is a no-op in exact arithmetic."""
    torch.manual_seed(3)
    num_primary, psi = 10, 15
    y = torch.randint(0, num_primary, (24,))
    # Small logits keep the reference's bare torch.exp in range, so the two are
    # comparable at all.
    logits = torch.randn(24, num_primary * psi) * 2.0

    mask = hierarchy_mask(y, num_primary, psi)
    ours = masked_softmax(logits, mask)
    ref = ref_mask_softmax(logits, mask)
    assert torch.allclose(ours, ref, atol=1e-6), (ours - ref).abs().max().item()

    # Rows sum to one, and all mass sits inside the true class's window.
    assert torch.allclose(ours.sum(dim=1), torch.ones(24), atol=1e-5)
    assert (ours * (1 - mask)).abs().max() < 1e-12, "mass leaked outside the mask"
    print(f"  max deviation {(ours - ref).abs().max():.2e}, no mass outside mask")


def test_masked_softmax_survives_logits_that_overflow_the_reference():
    """The reason for the deviation: bf16-scale logits blow torch.exp up."""
    mask = hierarchy_mask(torch.tensor([0]), 2, 3)
    logits = torch.tensor([[100.0, 101.0, 102.0, 0.0, 0.0, 0.0]])

    ours = masked_softmax(logits, mask)
    assert torch.isfinite(ours).all(), ours
    assert abs(ours.sum().item() - 1.0) < 1e-5, ours.sum().item()
    ref = ref_mask_softmax(logits, mask)
    assert not torch.isfinite(ref).all() or ref.sum() == 0, (
        "reference unexpectedly handled this; the deviation may be unnecessary"
    )
    print(f"  ours {ours[0][:3].tolist()} finite; reference produced {ref[0][:3].tolist()}")


def test_hierarchy_mask_matches_the_reference_blocks():
    y = torch.tensor([0, 1, 9, 4])
    ours = hierarchy_mask(y, 10, 15)
    ref = ref_build_mask(y, 10, 15)
    assert torch.equal(ours, ref), (ours != ref).nonzero()[:5]
    # Contiguous psi-wide window starting at psi*y, which is what makes the
    # `human` arm's one-hot over fine labels land inside its own mask.
    for row, cls in enumerate(y.tolist()):
        on = ours[row].nonzero().flatten().tolist()
        assert on == list(range(cls * 15, (cls + 1) * 15)), (cls, on)
    print("  identical; blocks are contiguous [15*y, 15*y+15)")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for test in tests:
        print(f"{test.__name__} ...")
        try:
            test()
        except Exception:  # noqa: BLE001
            failures.append(test.__name__)
            traceback.print_exc()
    print("-" * 72)
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
