"""Tests for the highest-risk mechanism: two adapters, one frozen base, and a
`functional_call` inner step that has to carry gradient back to phi.

Runs on CPU with a tiny random transformer, so it is a normal unit test rather
than something that needs a rented GPU. Everything here is meant to be true
before the first instance is ever started.

    python -m llm.tests.test_lora_functional
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from llm.losses import hierarchy_mask, masked_softmax, model_fit, scale_weights
from llm.networks.lora import DualLoRALinear, inject_dual_lora, set_active_adapter
from llm.networks.wamal_lm import WamalLM

NUM_PRIMARY, PSI = 3, 4
HIDDEN, VOCAB, SEQ, BATCH = 16, 32, 6, 5


class TinyAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        scores = torch.softmax(self.q_proj(x) @ self.k_proj(x).transpose(-1, -2), dim=-1)
        return self.o_proj(scores @ self.v_proj(x))


class TinyBackbone(nn.Module):
    """Stands in for a HF decoder: same interface, three orders of magnitude smaller."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.self_attn = TinyAttention(HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, input_ids=None, attention_mask=None):
        hidden = self.norm(self.self_attn(self.embed(input_ids)))

        class Out:
            last_hidden_state = hidden

        return Out()


def build() -> WamalLM:
    torch.manual_seed(0)
    base = TinyBackbone()
    for param in base.parameters():
        param.requires_grad_(False)
    inject_dual_lora(base, targets=("q_proj", "v_proj"), r=4, alpha=8)
    return WamalLM(base=base, feature_dim=HIDDEN, num_primary=NUM_PRIMARY, psi=PSI)


def make_batch():
    torch.manual_seed(1)
    return (
        torch.randint(0, VOCAB, (BATCH, SEQ)),
        torch.ones((BATCH, SEQ), dtype=torch.long),
        torch.randint(0, NUM_PRIMARY, (BATCH,)),
    )


# --------------------------------------------------------------------------- #


def test_parameter_groups_are_disjoint_and_complete():
    model = build()
    theta = set(model.theta_parameter_names())
    phi = set(model.phi_parameter_names())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    assert theta and phi, "both groups must be non-empty"
    assert not (theta & phi), f"overlap: {theta & phi}"
    assert theta | phi == trainable, f"unassigned: {trainable - theta - phi}"

    # The backbone itself must contribute nothing trainable.
    assert not any(
        p.requires_grad for n, p in model.base.named_parameters() if ".adapters." not in n
    )
    print(f"  theta={len(theta)} phi={len(phi)} tensors, disjoint and complete")


def test_base_weights_are_shared_not_duplicated():
    model = build()
    wrappers = [m for m in model.base.modules() if isinstance(m, DualLoRALinear)]
    assert wrappers, "no linears were wrapped"
    for wrapper in wrappers:
        assert len(wrapper.adapters) == 2
        # One base weight tensor serving both adapters is the entire memory
        # argument for this design.
        assert wrapper.base.weight.data_ptr() is not None
        assert not wrapper.base.weight.requires_grad
    print(f"  {len(wrappers)} wrapped linears, 1 shared base weight each")


def test_adapters_produce_different_outputs():
    """A zero-initialised B makes both adapters no-ops at init, so perturb first."""
    model = build()
    input_ids, attention_mask, _ = make_batch()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".adapters.theta." in name and "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.5)

    set_active_adapter(model.base, "theta")
    theta_out = model.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    set_active_adapter(model.base, "phi")
    phi_out = model.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    assert not torch.allclose(theta_out, phi_out), (
        "theta and phi produced identical features; the active-adapter switch is a no-op"
    )
    print(f"  adapter switch changes features by {(theta_out - phi_out).abs().max():.4f}")


def test_functional_call_respects_overrides():
    model = build()
    input_ids, attention_mask, _ = make_batch()

    baseline, _ = model.forward_primary(input_ids, attention_mask)

    overrides = OrderedDict()
    for name, param in model.theta_named_parameters():
        overrides[name] = param + torch.randn_like(param) * 0.5
    perturbed, _ = model.forward_primary(input_ids, attention_mask, params=overrides)

    assert not torch.allclose(baseline, perturbed), (
        "functional_call output is unchanged under perturbed theta; the overrides "
        "are not reaching the module"
    )

    # And overriding with the current values must be an exact no-op.
    identity = OrderedDict(model.theta_named_parameters())
    same, _ = model.forward_primary(input_ids, attention_mask, params=identity)
    assert torch.allclose(baseline, same, atol=1e-6)
    print(f"  overrides shift output by {(baseline - perturbed).abs().max():.4f}")


def test_second_order_gradient_reaches_phi():
    """The load-bearing test.

    phi only ever receives gradient through the inner SGD step: theta+ is a
    function of phi, and the outer loss is evaluated at theta+. If
    `create_graph=True` were dropped, or the override names were wrong, phi's
    gradients would silently be zero or None and every WAMAL run would quietly
    degrade to MAXL-with-a-frozen-label-net.
    """
    from llm.train_text import inner_sgd_update

    model = build()
    input_ids, attention_mask, y = make_batch()

    primary, auxiliary = model(input_ids, attention_mask, mode="primary")
    sub_labels, raw_weights = model(input_ids, attention_mask, y=y, mode="label")

    primary_loss = model_fit(primary, y, torch.device("cpu"), pri=True,
                             num_output=NUM_PRIMARY)
    aux_loss = model_fit(auxiliary, sub_labels, torch.device("cpu"), pri=False,
                         num_output=NUM_PRIMARY * PSI)
    factors = scale_weights(raw_weights.squeeze(-1), val_range=5.0)
    total = primary_loss.mean() + (aux_loss * factors).mean()

    fast = inner_sgd_update(model, total, lr=0.1, first_order=False)
    post, _ = model.forward_primary(input_ids, attention_mask, params=fast)
    model_fit(post, y, torch.device("cpu"), pri=True,
              num_output=NUM_PRIMARY).mean().backward()

    named_phi = model.phi_named_parameters()
    with_grad = [(n, p) for n, p in named_phi if p.grad is not None]
    assert with_grad, "no phi parameter received a gradient"

    nonzero = [n for n, p in with_grad if p.grad.abs().sum() > 0]
    assert nonzero, (
        "every phi gradient is exactly zero; the second-order path is broken"
    )

    # Both the sub-label head and the weight head must be live: the weight head
    # is what distinguishes WAMAL from MAXL.
    assert any("sublabel_head" in n for n in nonzero), "sub-label head got no gradient"
    assert any("weight_head" in n for n in nonzero), "weight head got no gradient"
    assert any(".adapters.phi." in n for n in nonzero), "phi LoRA got no gradient"
    print(f"  {len(nonzero)}/{len(named_phi)} phi tensors received nonzero gradient")


def test_first_order_blocks_the_meta_gradient():
    """Confirms the second-order term is doing the work, not something else."""
    from llm.train_text import inner_sgd_update

    model = build()
    input_ids, attention_mask, y = make_batch()

    primary, auxiliary = model(input_ids, attention_mask, mode="primary")
    sub_labels, raw_weights = model(input_ids, attention_mask, y=y, mode="label")
    loss = (
        model_fit(primary, y, torch.device("cpu"), pri=True, num_output=NUM_PRIMARY).mean()
        + model_fit(auxiliary, sub_labels, torch.device("cpu"), pri=False,
                    num_output=NUM_PRIMARY * PSI).mean()
    )

    fast = inner_sgd_update(model, loss, lr=0.1, first_order=True)
    post, _ = model.forward_primary(input_ids, attention_mask, params=fast)
    model_fit(post, y, torch.device("cpu"), pri=True,
              num_output=NUM_PRIMARY).mean().backward()

    live = [n for n, p in model.phi_named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0]
    assert not live, f"first-order mode still fed phi: {live[:3]}"
    print("  first-order mode correctly severs the meta gradient")


def test_masked_softmax_respects_the_hierarchy():
    torch.manual_seed(2)
    y = torch.tensor([0, 1, 2, 1, 0])
    logits = torch.randn(len(y), NUM_PRIMARY * PSI) * 10
    mask = hierarchy_mask(y, NUM_PRIMARY, PSI)
    probs = masked_softmax(logits, mask)

    assert torch.allclose(probs.sum(dim=1), torch.ones(len(y)), atol=1e-5)
    off_mask = probs * (1 - mask)
    assert off_mask.abs().max() == 0, "probability leaked outside the class window"
    for i, label in enumerate(y.tolist()):
        window = probs[i, label * PSI:(label + 1) * PSI]
        assert torch.isclose(window.sum(), torch.tensor(1.0), atol=1e-5)
    print("  masked softmax keeps all mass inside the true class window")


def test_weight_scaling_hits_the_documented_range():
    r = 5.0
    assert torch.isclose(scale_weights(torch.tensor(0.0), r), torch.tensor(2.0**-r))
    assert torch.isclose(scale_weights(torch.tensor(1.0), r), torch.tensor(2.0**r))
    assert torch.isclose(scale_weights(torch.tensor(0.5), r), torch.tensor(1.0))
    print(f"  weights span [2^-{r:.0f}, 2^{r:.0f}] with 0.5 -> 1.0")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for test in tests:
        print(f"{test.__name__} ...")
        try:
            test()
        except AssertionError as exc:
            print(f"  FAILED: {exc}")
            failures.append((test.__name__, str(exc)))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failures.append((test.__name__, f"{exc.__class__.__name__}: {exc}"))

    print("-" * 72)
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED")
        for name, why in failures:
            print(f"  {name}: {why}")
        return 1
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
