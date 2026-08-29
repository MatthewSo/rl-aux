"""The single source of truth for a run.

Every knob lives here and is serialised to `config.json` next to the results.
Arm behaviour is *derived* from `arm` rather than passed as independent booleans,
which is what makes it structurally impossible to repeat the `vit_coco_single_dense.py`
bug where the "single-task" baseline was silently configured as WAMAL.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

ARMS = ("single", "human", "maxl", "wamal")
DATASETS = ("clinc150", "newsgroups")

# Fewer examples means fewer optimizer steps per epoch, so low-k cells need more
# epochs to see a comparable number of updates. Provisional; phase 0 retunes it.
# Identical across arms within a cell by construction -- that is the point.
#
# k=1 and k=2 continue the existing trend rather than inventing one: each doubling
# of k multiplies the budget by ~0.72 across the original five entries, so halving
# from k=5 gives ~55 and halving again ~80. Note this makes the low-k cells more
# expensive than k=5, not cheaper -- per-epoch cost is roughly 3.0 + 0.33k seconds,
# so the extra epochs outweigh the smaller training set.
EPOCHS_BY_SHOTS: dict[int, int] = {1: 80, 2: 60, 5: 40, 10: 30, 25: 20, 50: 15, 100: 10}

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "clinc150": {"num_primary": 10, "psi": 15, "max_len": 64},
    # Newsgroups posts are full documents rather than utterances, so they need a
    # much longer window.
    "newsgroups": {"num_primary": 4, "psi": 4, "max_len": 256},
}


@dataclass
class ExperimentConfig:
    # --- cell identity -----------------------------------------------------
    dataset: str = "clinc150"
    model: str = "opt-125m"
    arm: str = "wamal"
    shots: int = 100
    seed: int = 0

    # --- data --------------------------------------------------------------
    max_len: int | None = None          # resolved from dataset
    num_primary: int | None = None      # resolved from dataset
    psi: int | None = None              # resolved from dataset
    num_workers: int = 2

    # --- optimization ------------------------------------------------------
    # SGD is the paper-faithful default (WAMAL was reported unstable under Adam);
    # AdamW is the LoRA convention. Phase 0 picks per-optimizer LRs.
    optimizer: str = "sgd"
    primary_lr: float = 1e-2
    gen_lr: float = 1e-3
    momentum: float = 0.9
    gen_weight_decay: float = 5e-4
    batch_size: int = 32
    total_epoch: int | None = None      # resolved from shots
    step_size: int = 50
    gamma: float = 0.5
    grad_clip: float = 1.0

    # --- wamal -------------------------------------------------------------
    val_range: float = 5.0
    entropy_loss_factor: float = 0.2
    normalize_batch_weights: bool = False
    inner_lr: float | None = None       # defaults to primary_lr
    first_order: bool = False           # skip the second-order term (debug/ablation)

    # Fraction of the k-shot training subset withheld from theta. At 0.0 (the
    # default) the meta objective is the primary loss on the same batch theta was
    # just fitted on, which is paper-faithful and what phase 1 ran.
    #
    # Phase 1 showed why that matters: the meta loss reaches ~1e-5 halfway through
    # a k=5 run, so phi trains for only half the epochs, and the arms that learn
    # labels do worst exactly where the meta signal dies soonest. Above 0.0 the
    # withheld split becomes the meta objective's query set -- data theta never
    # trains on, so the loss cannot be driven to zero by memorisation.
    #
    # The withheld examples leave theta's training set for *every* arm, not just
    # the meta ones. Otherwise wamal would be compared against baselines that saw
    # more data, and the deficit under test would be confounded by the split.
    meta_holdout_frac: float = 0.0

    # --- backbone / adapters ----------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] | None = None   # resolved from the registry
    pooling: str = "last"               # "last" | "mean"
    dtype: str = "bf16"
    # Must stay "eager" for any arm with a meta step: the fused SDPA kernels
    # have no double-backward, and differentiating through the inner update
    # needs one. Exposed only so the single-task baseline can be run faster if
    # that ever matters.
    attn_implementation: str = "eager"

    # --- runtime -----------------------------------------------------------
    device: str = "cuda:0"
    results_dir: str | None = None
    dump_every: int = 10
    log_cosine: bool = False            # 2 extra backward passes; diagnostic only
    save_checkpoint: bool = True
    tag: str = ""

    # --- tracking ----------------------------------------------------------
    # Off by default: metrics.jsonl is the source of truth and every analysis
    # reads it, so wandb is a mirror you opt into rather than a dependency.
    wandb: bool = False
    wandb_project: str = "wamal-lm"
    wandb_entity: str | None = None
    wandb_group: str = ""               # sweep sets this to the phase name
    wandb_mode: str = "auto"            # auto | online | offline | disabled

    _resolved: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ #

    def resolve(self) -> "ExperimentConfig":
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {self.pooling!r}")
        if self.optimizer not in ("sgd", "adamw"):
            raise ValueError(f"optimizer must be 'sgd' or 'adamw', got {self.optimizer!r}")
        if self.wandb_mode not in ("auto", "online", "offline", "disabled"):
            raise ValueError(
                f"wandb_mode must be auto/online/offline/disabled, "
                f"got {self.wandb_mode!r}"
            )
        if self.needs_meta_step and self.attn_implementation != "eager":
            raise ValueError(
                f"arm {self.arm!r} differentiates through the inner update, which "
                f"requires attn_implementation='eager'; got "
                f"{self.attn_implementation!r}. Fused attention kernels have no "
                "double-backward and will fail partway through the first epoch."
            )

        if not 0.0 <= self.meta_holdout_frac < 0.5:
            raise ValueError(
                f"meta_holdout_frac must be in [0.0, 0.5), got "
                f"{self.meta_holdout_frac}. Withholding half or more of a "
                "few-shot subset would change the experiment being run."
            )
        if self.meta_holdout_frac > 0:
            # Stratified per fine class, so both splits cover every class. That
            # needs at least one example on each side of the split.
            held = max(1, round(self.shots * self.meta_holdout_frac))
            if self.shots - held < 1:
                raise ValueError(
                    f"meta_holdout_frac={self.meta_holdout_frac} withholds {held} "
                    f"of {self.shots} examples per fine class, leaving "
                    f"{self.shots - held} to train on. Needs k >= 2."
                )

        defaults = DATASET_DEFAULTS[self.dataset]
        if self.num_primary is None:
            self.num_primary = defaults["num_primary"]
        if self.psi is None:
            self.psi = defaults["psi"]
        if self.max_len is None:
            self.max_len = defaults["max_len"]

        if self.total_epoch is None:
            # Unlisted k values fall back to the nearest listed one rather than
            # silently defaulting, so an odd sweep value can't get 0 epochs.
            nearest = min(EPOCHS_BY_SHOTS, key=lambda s: abs(s - self.shots))
            self.total_epoch = EPOCHS_BY_SHOTS[nearest]

        if self.inner_lr is None:
            self.inner_lr = self.primary_lr

        if self.lora_targets is None:
            from llm.models_registry import get
            self.lora_targets = get(self.model).lora_targets
        self.lora_targets = tuple(self.lora_targets)

        if self.results_dir is None:
            from llm import env
            self.results_dir = str(env.results_dir())

        self._resolved = True
        return self

    # --- arm-derived behaviour --------------------------------------------
    # Read-only properties, never constructor arguments: an inconsistent
    # combination cannot be expressed.

    @property
    def num_auxiliary(self) -> int:
        return self.num_primary * self.psi

    @property
    def uses_auxiliary(self) -> bool:
        return self.arm != "single"

    @property
    def uses_human_labels(self) -> bool:
        return self.arm == "human"

    @property
    def uses_generated_labels(self) -> bool:
        return self.arm in ("maxl", "wamal")

    @property
    def uses_learned_weights(self) -> bool:
        return self.arm == "wamal"

    @property
    def needs_meta_step(self) -> bool:
        """The `human` arm has fixed targets and uniform weights, so phi has
        nothing to learn and the expensive bilevel step is skipped."""
        return self.uses_generated_labels

    @property
    def needs_label_network(self) -> bool:
        return self.uses_generated_labels

    @property
    def model_slug(self) -> str:
        return self.model

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in dataclasses.asdict(self).items() if not k.startswith("_")}
        out["lora_targets"] = list(self.lora_targets or ())
        out["derived"] = {
            "num_auxiliary": self.num_auxiliary,
            "uses_auxiliary": self.uses_auxiliary,
            "uses_human_labels": self.uses_human_labels,
            "uses_generated_labels": self.uses_generated_labels,
            "uses_learned_weights": self.uses_learned_weights,
            "needs_meta_step": self.needs_meta_step,
        }
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in data.items() if k in fields and not k.startswith("_")}
        if isinstance(clean.get("lora_targets"), list):
            clean["lora_targets"] = tuple(clean["lora_targets"])
        return cls(**clean).resolve()

    def variant(self) -> str:
        """Method variant, for keeping incomparable runs apart.

        `base` reproduces phase 1 exactly, so existing results keep their paths
        and their place in every table. Anything else is a different method and
        must never be averaged in with `base`: `aggregate` groups on this, and
        `cell_id` puts it in the path so the two cannot even share a directory.
        """
        if self.meta_holdout_frac > 0:
            return f"holdout{self.meta_holdout_frac:g}"
        return "base"

    def cell_id(self) -> str:
        # The variant rides on the arm segment because it is part of the method,
        # not of the data. `base` is spelled as a bare arm so that the 200 phase-1
        # run directories stay exactly where they are.
        variant = self.variant()
        arm = self.arm if variant == "base" else f"{self.arm}__{variant}"
        return f"{self.dataset}/{self.model}/{arm}/k{self.shots}/seed{self.seed}"
