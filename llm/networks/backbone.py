"""Frozen decoder-only backbone loading."""

from __future__ import annotations

import torch

from llm import env
from llm.models_registry import get


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """bf16 where supported, fp32 otherwise.

    Turing cards (the 2080Tis in the fleet) have no bf16. The fallback is fp32
    rather than fp16 on purpose: `model_fit` takes `log` of a softmax output, and
    fp16's narrow exponent range makes that a realistic source of NaNs.
    """
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}")


def load_backbone(model_slug: str, dtype: torch.dtype, device: torch.device,
                  attn_implementation: str = "eager"):
    """Load the base transformer (no LM head) and freeze it.

    `AutoModel`, not `AutoModelForCausalLM`: we only ever need hidden states, and
    skipping the vocab-sized output projection saves real memory (~38M params on
    OPT-125M alone).

    `attn_implementation="eager"` is not optional. WAMAL's outer step
    differentiates *through* the inner SGD update, so the backward pass itself
    has to be differentiable. The fused attention kernels do not support that:

        RuntimeError: derivative for
        aten::_scaled_dot_product_flash_attention_for_cpu_backward
        is not implemented

    and the CUDA flash/mem-efficient backends fail the same way. Eager attention
    costs throughput and activation memory, which is a real and permanent tax on
    this method's runtime, but it is the only path that supports double backward.
    """
    from transformers import AutoConfig, AutoModel

    spec = get(model_slug)
    cache = str(env.hf_home() / "hub")

    config = AutoConfig.from_pretrained(
        spec.hf_id, cache_dir=cache, trust_remote_code=spec.trust_remote_code
    )
    kwargs = dict(
        config=config,
        cache_dir=cache,
        trust_remote_code=spec.trust_remote_code,
        attn_implementation=attn_implementation,
    )
    try:
        model = AutoModel.from_pretrained(spec.hf_id, dtype=dtype, **kwargs)
    except TypeError:
        # transformers renamed torch_dtype -> dtype around 4.56; support both so
        # the pinned floor in requirements.txt stays honest.
        model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, spec


@torch.no_grad()
def measure_feature_dim(model, device: torch.device) -> int:
    """Infer the pooled feature width with a dummy forward.

    Not `config.hidden_size`: OPT can down-project its output
    (`word_embed_proj_dim != hidden_size`), so the config value is wrong for some
    checkpoints and the failure would show up as a shape error deep in a head.
    """
    input_ids = torch.zeros((1, 8), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, 8), dtype=torch.long, device=device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return int(hidden.shape[-1])


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, how: str) -> torch.Tensor:
    """[B,T,H] -> [B,H]."""
    if how == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if how == "last":
        # Right-padded, so the last real token is at sum(mask)-1. Reading
        # hidden[:, -1] instead would pick up a pad position for every sequence
        # shorter than the batch maximum.
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        index = lengths.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden.gather(1, index).squeeze(1)
    raise ValueError(f"unknown pooling {how!r}")
