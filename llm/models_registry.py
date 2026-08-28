"""Registry of backbone checkpoints and download tiers.

Central so that bootstrap, preflight, sweeps and training all agree on which
checkpoints exist and which ones a given phase needs. Downloading only the tier
you will actually use matters when the box is billed hourly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    hf_id: str
    params: str
    family: str
    note: str = ""
    # Modules to attach LoRA to. Every model in the ladder happens to use the
    # same two projection names, but keep it per-model so an odd architecture
    # can override without touching call sites.
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    trust_remote_code: bool = False
    extras: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    "opt-125m": ModelSpec(
        slug="opt-125m",
        hf_id="facebook/opt-125m",
        params="125M",
        family="opt",
        note="Dev workhorse. ~180B training tokens, 2021 cutoff -> contamination control.",
    ),
    "opt-1.3b": ModelSpec(
        slug="opt-1.3b",
        hf_id="facebook/opt-1.3b",
        params="1.3B",
        family="opt",
        note="Contamination-controlled scale step.",
    ),
    "granite-350m": ModelSpec(
        slug="granite-350m",
        hf_id="ibm-granite/granite-4.0-350m-base",
        params="350M",
        family="granite",
        note="Modern dense baseline, Apache 2.0, ~15T tokens.",
    ),
    "granite-h-350m": ModelSpec(
        slug="granite-h-350m",
        hf_id="ibm-granite/granite-4.0-h-350m-base",
        params="340M",
        family="granite",
        note="Hybrid Mamba2 twin of granite-350m: matched size and data, so the "
        "only variable is architecture.",
    ),
    "granite-1b": ModelSpec(
        slug="granite-1b",
        hf_id="ibm-granite/granite-4.0-1b-base",
        params="1.6B",
        family="granite",
        note="Within-family scale step from granite-350m.",
    ),
    "olmo2-1b": ModelSpec(
        slug="olmo2-1b",
        hf_id="allenai/OLMo-2-0425-1B",
        params="1B",
        family="olmo",
        note="Headline model: fully open weights, data and training code.",
    ),
    "qwen2.5-0.5b": ModelSpec(
        slug="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        params="0.5B",
        family="qwen",
        note="Cross-family replication.",
    ),
}


# facebook/opt-350m is deliberately absent. It is the only OPT checkpoint
# trained with post-layernorm (do_layer_norm_before=False) and a down-projected
# embedding (word_embed_proj_dim=512 vs hidden_size=1024). Meta filed this as a
# bug in metaseq and never retrained it, so including it would silently break
# the within-family scaling ladder that OPT is here to provide.
EXCLUDED: dict[str, str] = {
    "facebook/opt-350m": (
        "post-layernorm outlier in the OPT family; not comparable to 125m/1.3b"
    ),
}


TIERS: dict[str, list[str]] = {
    "smoke": ["opt-125m"],
    "phase0": ["opt-125m"],
    "phase1": ["opt-125m"],
    "phase2": ["granite-350m", "olmo2-1b"],
    "phase3": ["qwen2.5-0.5b", "opt-1.3b", "granite-h-350m", "granite-1b"],
}
TIERS["all"] = sorted({m for models in TIERS.values() for m in models})


def get(slug: str) -> ModelSpec:
    if slug in EXCLUDED:
        raise ValueError(f"{slug} is excluded: {EXCLUDED[slug]}")
    if slug not in MODELS:
        raise KeyError(f"unknown model {slug!r}; known: {sorted(MODELS)}")
    return MODELS[slug]


def tier_models(tier: str) -> list[ModelSpec]:
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; known: {sorted(TIERS)}")
    return [MODELS[s] for s in TIERS[tier]]
