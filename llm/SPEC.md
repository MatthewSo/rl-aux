# WAMAL-LM: Weight-Aware Meta Auxiliary Learning for Small Language Models

Design specification. Status: proposed, not implemented.

---

## 1. Objective

Reproduce the CIFAR-100-20 result from the WAMAL paper in the language domain, on small
decoder-only base LMs, using a parameter-efficient implementation.

The CIFAR-100-20 result is the paper's strongest claim: on a primary task with 20 coarse
classes, auxiliary sub-labels *learned* by the label-weight network (81.5%) beat the *human*
100-class hierarchy (75.7%) and weight-unaware MAXL (76.6%). It works because CIFAR-100
supplies a real, perfectly balanced human hierarchy (20 superclasses x 5 fine classes) to
compare against.

We want the same four-way comparison in text, plus two things the vision work could not do:

- a **quantitative** answer to "what do the learned sub-labels correspond to?", by scoring
  generated sub-labels against ground-truth human fine labels;
- a **few-shot adaptation curve** rather than a single accuracy number, which tests the
  paper's own hypothesis (Section 5.3) that WAMAL's advantage grows as data shrinks.

### 1.1 Claims to test

- **C1.** WAMAL > single-task on few-shot text classification, and the gap widens as shots decrease.
- **C2.** WAMAL > human-hierarchy auxiliary task (the CIFAR-100-20 headline, in text).
- **C3.** WAMAL > MAXL, i.e. the per-sample weight head contributes beyond learned labels.
- **C4.** The effect survives on modern small decoder base LMs, not only on legacy/toy backbones.

### 1.2 Non-goals

- Not a MAML/Reptile experiment. We are not meta-learning an initialization across a task
  distribution. The outer variable is the auxiliary supervision, exactly as in the paper.
- Not chasing SOTA accuracy on CLINC150.
- Not generative/seq2seq. Primary and auxiliary tasks are both classification over a pooled
  representation.

---

## 2. Method recap (what carries over unchanged)

From `wamal/networks/wamal_wrapper.py` and `wamal/train_network.py`.

Two networks share an architecture family but not weights:

- **Primary network** `f_theta`: backbone + primary head (`C` classes) + auxiliary head (`C*psi`).
- **Label-weight network** `g_phi`: backbone + sub-label head (`C*psi`) + weight head (1, sigmoid).

**Masked softmax.** Sub-label logits are masked so primary class `i` may only place mass on
auxiliary slots `[i*psi, (i+1)*psi)`:

```
p_k = exp(z_k) * m_k / sum_j exp(z_j) * m_j
m_k = 1 if psi*y <= k < psi*(y+1) else 0
```

**Weight scaling.** The sigmoid output `w in [0,1]` maps to a multiplier in `[2^-r, 2^r]`:

```
weight_factor = 2 ** (2 * r * w - r)          # r = RANGE = 5
```

**Losses.** `model_fit` in `wamal/networks/utils.py` is a focal loss applied to
already-softmaxed probabilities (`pri=True` one-hot, `pri=False` soft targets).
`model_entropy` is the batch-mean negative entropy over generated labels.

**Bilevel step.** Per epoch: (a) update `theta` on `L_pri + mean(w * L_aux)`; (b) take a
differentiable inner SGD step `theta+ = theta - lr * grad` with `create_graph=True`, re-evaluate
primary loss at `theta+`, and update `phi` on `L_pri(theta+) + entropy_factor * H`.

All of this is preserved. What changes is the backbone, the pooling, and the parameterization.

---

## 3. What changes for language

### 3.1 Pooling (required)

`WamalWrapper.forward` does `feat.flatten(1)` after the backbone. A causal LM returns
`[B, T, H]`; flattening yields a variable-length `[B, T*H]` and breaks the heads.

Replace with last-non-pad-token pooling (the convention used by
`OPTForSequenceClassification`, `LlamaForSequenceClassification`, etc.):

```python
def pool_last(hidden, attention_mask):          # [B, T, H], [B, T] -> [B, H]
    idx = attention_mask.sum(dim=1) - 1         # last real token per row
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
```

Mean pooling over non-pad tokens is the fallback if last-token proves unstable at 125M.
Make it a config flag (`POOLING = "last" | "mean"`), not a hardcoded choice.

### 3.2 Shared frozen backbone with two LoRA adapters (required)

The current code instantiates two complete backbones. At 1B that is ~2B resident parameters
plus a second-order graph over all of them, which does not fit the plan.

Target parameterization:

```
frozen base weights (bf16, no grad, no optimizer state)   x1 in memory
  |
  +-- LoRA_theta  + primary_head + auxiliary_head          <- theta, inner+outer trainable
  +-- LoRA_phi    + sublabel_head + weight_head            <- phi,   outer trainable
```

Two forward passes over the same base weights with different adapters. This is both a memory
necessity and the main methodological contribution of the port: *bilevel meta auxiliary
learning at LLM scale without a second backbone.*

### 3.3 Inner loop restricted to trainable parameters (required)

`inner_sgd_update` currently does:

```python
grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
```

`model.parameters()` would include the frozen base. Restrict to LoRA + head parameters:

```python
def inner_sgd_update(module, loss, lr, param_filter):
    named = OrderedDict((n, p) for n, p in module.named_parameters() if param_filter(n))
    grads = torch.autograd.grad(loss, named.values(), create_graph=True, allow_unused=True)
    return OrderedDict(
        (n, p if g is None else p - lr * g) for (n, p), g in zip(named.items(), grads)
    )
```

Keeping `create_graph=True` (true second-order) is affordable precisely because the parameter
set is small. Do not silently fall back to a first-order approximation; if it becomes
necessary, make it an explicit config flag and report it.

---

## 4. Experiment design

### 4.1 Datasets

**CLINC150** (primary). 10 domains x exactly 15 intents = 150, perfectly balanced.

| Role | Mapping |
|---|---|
| Primary task `y` | 10 domains, `C = 10` |
| Human auxiliary task | 150 intents, `psi = 15`, `C*psi = 150` |
| Learned auxiliary task | 150 generated sub-labels under the same mask |

Use the `full` config (100 train / 20 val / 30 test per intent). Exclude out-of-scope from
training; retain the OOS test split for an optional robustness probe.

The balanced 15-per-domain structure is why this dataset was chosen: the existing
`psi = [num_auxiliary // num_primary] * num_primary` mask requires uniform `psi`. Do not
substitute an unbalanced hierarchy (DBpedia, WOS-46985, ultra-fine typing) without first
implementing variable-`psi` support.

**20 Newsgroups** (secondary). Curate to 4 coarse x 4 fine, `psi = 4`:

| Coarse | Fine (4 each) |
|---|---|
| comp | graphics, os.ms-windows.misc, sys.ibm.pc.hardware, sys.mac.hardware |
| rec | autos, motorcycles, sport.baseball, sport.hockey |
| sci | crypt, electronics, med, space |
| talk | politics.guns, politics.mideast, politics.misc, religion.misc |

Dropped: `comp.windows.x`, `misc.forsale`, `alt.atheism`, `soc.religion.christian` (to keep
`psi` uniform). Document the drop in the paper; it is a curation choice, not a result.

**Shot levels.** Sample `k` examples per *fine* class, `k in {5, 10, 25, 50, 100}`, so the
hierarchy stays balanced at every level. Primary-class counts are `15k` (CLINC) / `4k` (20NG).
`k = 100` is the full CLINC150 train split.

Sampling must be seeded and reproducible. Note that the existing `PerClassCap` defaults to
`seed=None`, which draws a different subset per run; the new sampler must not.

### 4.2 Models

| Phase | Model | Params | Role |
|---|---|---|---|
| 0-1 | `facebook/opt-125m` | 125M | Calibration + dev + 10-seed sweeps; contamination-clean |
| 2 | `ibm-granite/granite-4.0-350m-base` | 350M | Main result, modern |
| 2 | `allenai/OLMo-2-0425-1B` | 1B | Headline; fully open provenance |
| 3 | `Qwen/Qwen2.5-0.5B` | 0.5B | Cross-family replication |
| 3 | `facebook/opt-1.3b` | 1.3B | Contamination-controlled scale step |
| 3 | `ibm-granite/granite-4.0-h-350m-base` | 340M | Attention-vs-hybrid control (matched data/size) |

**Do not use `facebook/opt-350m`.** It is the one checkpoint in the OPT family trained with
post-layernorm (`do_layer_norm_before=False`) and a down-projected embedding
(`word_embed_proj_dim=512` vs `hidden_size=1024`). Meta logged this as a bug in `metaseq` and
never retrained it. Including it would break the within-family scaling ladder that is OPT's
main reason for being here.

**Licensing.** OPT is under the OPT-175B License (non-commercial research only), not Apache
2.0. Acceptable for the paper; relevant if artifacts are released.

### 4.3 Sequence length

CLINC150 utterances are single sentences. Set `MAX_LEN = 64` (covers ~99% without truncation);
20NG posts need `MAX_LEN = 256`. Short sequences are what make the full run matrix affordable.

### 4.4 Optimizer

The paper's default is SGD, and Appendix D reports Adam is unstable with WAMAL and highly
LR-sensitive. LoRA on decoder LMs conventionally wants AdamW. These conflict, so do not guess:
Phase 0 is an explicit LR calibration (see 7.1). Default to SGD for the primary network to stay
faithful to the paper, and treat AdamW as a documented variant.

### 4.5 Arms

Four arms, reproducing the paper's Figure 3a.

| # | Arm | `SKIP_MAL` | `LEARN_WEIGHTS` | Sub-labels | Tests |
|---|---|---|---|---|---|
| 1 | Single-task | True | False | none | C1 baseline |
| 2 | Human auxiliary | False | False | fixed, real fine labels | C2 baseline |
| 3 | MAXL | False | False | learned | C3 baseline |
| 4 | WAMAL | False | True | learned | C1-C4 |

Arm 2 needs a training path that consumes ground-truth fine labels instead of calling the
label network for sub-labels. Port the pattern from `legacy/model_vgg_human.py`.

**Fairness constraints.** All four arms within a `(model, dataset, k, seed)` cell must share:
epochs, batch size, primary LR, LoRA rank, scheduler, max length, and the *identical* few-shot
subset. This is enforced by construction (see 5.3) rather than by convention, because per-arm
config drift is exactly what damaged several comparisons in the vision code.

---

## 5. Module layout

```
llm/
  SPEC.md                     this file
  __init__.py
  args.py                     CLI: --model --dataset --arm --shots --seed --gpu --run_id
  config.py                   ExperimentConfig dataclass; single source of truth
  seeding.py                  seed_everything(seed) -> torch/numpy/random/cuda

  datasets/
    __init__.py
    clinc150.py               CLINC150Hierarchical
    newsgroups.py             NewsgroupsHierarchical
    fewshot.py                SeededPerClassSample
    collate.py                TextCollator

  networks/
    __init__.py
    backbone.py               load_base_backbone(), pool_last(), pool_mean()
    lora.py                   LoRALinear, inject_lora(), lora_parameters()
    wamal_lm.py               WamalLM, LabelWeightLM

  train_text.py               train_wamal_text(...)
  paths.py                    run-directory naming, self-contained
  tracking.py                 optional wandb mirror; never load-bearing
  runner.py                   builds everything from ExperimentConfig and dispatches

  bootstrap/                  see section 11 (vast.ai deployment)
    bootstrap.sh              one-shot: env + models + data + preflight
    setup_env.sh              pip install into the image's python
    setup_secrets.sh          prompt for and persist WANDB_API_KEY / HF_TOKEN
    download_models.py        prefetch HF weights into the persistent cache
    download_data.py          prefetch CLINC150 + 20NG into the persistent cache
    preflight.py              GPU / memory / end-to-end smoke test
    sync_results.sh           tar results for retrieval before teardown

  datasets/
    clinc150_domains.json     the 10x15 hierarchy asset (verified at load)

  scripts/
    run_experiment.py         single entry point, arm selected by flag
    sweep.py                  resumable run matrix

  analysis/
    aggregate.py              seeds -> mean/std tables
    curves.py                 per-epoch curves + training-health diagnostics
    weights.py                weight distributions, top/bottom-k examples
    sublabels.py              NMI / purity vs human fine labels

  tests/
    test_lora_functional.py   the risk-10.1 unit test
    test_train_loop.py        end-to-end, all four arms, on a tiny synthetic backbone
    test_sweep.py             cell enumeration, resumption, cost estimate, multi-GPU queue
    test_tracking.py          wandb degrades to a no-op and never breaks a run
```

### 5.2 Tracking

`metrics.jsonl` is the source of truth. `aggregate`, `curves`, `weights` and `sublabels` all read
the local files and none of them import wandb, so a sweep whose tracking failed is still a
complete, analysable experiment. `tracking.py` only mirrors, and `--wandb` is off by default.

That ordering is deliberate rather than incidental. Tracking runs on an instance billed by the
hour where the only unrecoverable loss is the training itself, so no tracking failure may
propagate into a run: every call is wrapped, and the first failure disables the tracker for the
remainder of the run rather than retrying once per epoch against a dead socket.

Three details matter for sweeps specifically. Runs are **grouped by phase and split by arm**, so
400 cells arrive as one comparable experiment instead of 400 orphans. Run ids are **derived from
the cell id**, so a re-run after a crash resumes its wandb run -- random ids plus a resumable
sweep would produce a project full of half-runs. And with no credentials the tracker records
**offline** rather than blocking on a handshake that will fail, because a five-minute
`init_timeout` multiplied across a sweep is real money.

Plus, at the repo root, the operator-side half of the deployment story:

```
RUNBOOK.md                    the lifecycle: provision -> bootstrap -> launch -> sync -> teardown
deploy/                       runs on the laptop, never on the GPU box
  env.example                 template for tokens and connection vars; deploy/env is gitignored
  _common.sh                  shared ssh/rsync plumbing (bash 3.2 compatible, for macOS)
  remote.sh                   run a command on the box, follow logs, clone/update the repo
  sync.sh                     rsync pull/push, plus a remote progress summary
```

The split is deliberate: `deploy/` needs your SSH config and never needs a GPU; `llm/bootstrap/`
needs a GPU and never needs your SSH config. That means a box reached some other way -- the
vast.ai web terminal, a Jupyter cell -- can still be bootstrapped with no local scaffolding.

Rationale for one `run_experiment.py` rather than the repo's one-script-per-cell convention:
the current `wamal/` directory has 39 near-duplicate scripts and the per-arm config drift
between them (batch size 20 vs 30, 75 vs 200 epochs, aux-set on vs off) is the direct cause of
several unfair comparisons. A single entry point with an `--arm` flag makes those mismatches
impossible.

---

## 6. Interfaces

### 6.1 Datasets

```python
class CLINC150Hierarchical(torch.utils.data.Dataset):
    """Returns (text: str, coarse: int, fine: int).

    coarse in [0, 10)   domain, the primary label
    fine   in [0, 150)  intent, remapped so that fine // 15 == coarse
    """
    NUM_PRIMARY = 10
    NUM_FINE = 150
    PSI = 15

    def __init__(self, root: str, split: str = "train", config: str = "full"): ...
```

The `fine // PSI == coarse` invariant is load-bearing: it makes the human fine labels directly
usable as targets under the existing mask without remapping at train time. Assert it in
`__init__`.

**Hierarchy provenance.** HuggingFace `clinc_oos` ships intent labels but *not* the domain
grouping, and CLINC never released the grouping in machine-readable form (it appears only in
Table 5 of Larson et al. 2019). The mapping therefore lives in
`llm/datasets/clinc150_domains.json` and is validated at load time:

1. exactly 10 domains,
2. exactly 15 intents per domain,
3. the 150 intent names are set-equal to the label names in the loaded split.

Check 3 is the important one — it catches any typo or omission in the asset and fails loudly
rather than silently training on a corrupted hierarchy. It cannot catch a correctly-spelled
intent filed under the wrong domain, so the JSON should be eyeballed once before the first
real run.

Use HF config `plus` (100 train / 20 val / 30 test per intent, i.e. the "full" dataset with a
larger OOS pool); HF does not expose a config named `full`. Drop the `oos` label from train and
validation; retain the OOS test rows only for the optional robustness probe.

`NewsgroupsHierarchical` mirrors this with `NUM_PRIMARY = 4`, `NUM_FINE = 16`, `PSI = 4`.

```python
class SeededPerClassSample(torch.utils.data.Dataset):
    """k examples per FINE class, deterministic given seed."""
    def __init__(self, base, k: int, seed: int, label_index: int = 2): ...
```

```python
class TextCollator:
    """Tokenize a batch of (text, coarse, fine) -> model inputs."""
    def __init__(self, tokenizer, max_len: int): ...
    def __call__(self, batch) -> dict:
        # {"input_ids", "attention_mask", "labels", "fine_labels"}
```

Note `train_wamal_network` already forwards a `collate_fn` when re-splitting for auxiliary
sets (`src_collate`), so the collator is compatible with that path.

### 6.2 LoRA

Hand-rolled rather than `peft`, so that parameter names are fully controlled and
`torch.func.functional_call` overrides work with the existing prefix-stripping pattern.

```python
class DualLoRALinear(nn.Module):
    """Wraps one frozen nn.Linear and holds BOTH adapters.

        y = base(x) + adapters[active](x)

    `active` is a plain attribute (not a parameter), switched by the owning
    module before each forward.
    """
    def __init__(self, base: nn.Linear, r=16, alpha=32, adapters=("theta", "phi")): ...

def inject_dual_lora(model, targets=("q_proj", "v_proj"), r=16, alpha=32) -> list[str]: ...
def set_active_adapter(model, name: str) -> None: ...
```

Start with `q_proj`/`v_proj` only at `r = 16`, `alpha = 32`. Those two names cover every model
in the ladder: OPT uses `q_proj`/`v_proj`, and Llama, Qwen, OLMo and Granite all do too.

**Why dual adapters in one wrapper rather than two wrapped copies.** The two networks must
share one set of frozen base weights while applying different low-rank deltas. Wrapping the
base twice would require two module trees over the same tensors; merging `W + sBA` per forward
would materialise a full-size weight per targeted layer and defeat the point of LoRA. Holding
both adapters in one wrapper and flipping an attribute is the cheapest correct option.

### 6.3 Network

**Design change from the original sketch:** one module owning everything, rather than two
modules sharing a base.

`torch.func.functional_call(module, overrides, args)` resolves override names relative to
`module`. With two sibling objects sharing a base there is no single well-defined name root,
and the existing prefix-stripping pattern in `wamal_wrapper.py` (`k.split('backbone.', 1)[1]`)
does not generalise. A single owner makes the inner step unambiguous.

```python
class WamalLM(nn.Module):
    """Owns the frozen base, both LoRA adapters, and all four heads."""

    def __init__(self, base, num_primary, psi, lora_r=16, lora_alpha=32,
                 pooling="last"): ...

    # mode dispatch exists so functional_call (which invokes __call__) can reach both paths
    def forward(self, input_ids, attention_mask, y=None, mode="primary"):
        """mode="primary" -> (primary_probs [B,C], aux_probs [B,C*psi]), both softmaxed
           mode="label"   -> (sub_labels [B,C*psi] masked-softmaxed, weights [B,1] in [0,1])"""

    def theta_named_parameters(self): ...   # adapters.theta + primary_head + auxiliary_head
    def phi_named_parameters(self): ...     # adapters.phi + sublabel_head + weight_head

    def save(self, path, **extra): ...      # LoRA + heads + base model id only
```

Two optimizers over the disjoint groups `theta_named_parameters()` and
`phi_named_parameters()`. Invariants asserted at construction:

```python
assert all(not p.requires_grad for p in base.parameters())    # base frozen
assert set(theta_names).isdisjoint(phi_names)                 # no shared trainables
```

Checkpointing excludes the base: the existing wrappers save `self.state_dict()`, which here
would write the full backbone every epoch.

The pooled feature dimension is measured with a dummy forward at construction rather than read
from `config.hidden_size`, because OPT can down-project its output
(`word_embed_proj_dim != hidden_size`).

### 6.4 Training

```python
def train_wamal_text(
    cfg: ExperimentConfig,
    device,
    model: WamalLM,
    loaders: dict,                 # {"train","val","test"}
    theta_optimizer, theta_scheduler,
    phi_optimizer, phi_scheduler,
    save_path: str,
) -> dict: ...                     # returns the summary record
```

Behaviour per arm is derived from `cfg.arm`, never passed as loose booleans, so no caller can
construct an inconsistent combination:

| Arm | aux target | weights | meta step |
|---|---|---|---|
| `single` | none | n/a | no |
| `human` | one-hot over true fine labels | uniform | no |
| `maxl` | generated | uniform | yes |
| `wamal` | generated | learned | yes |

The `human` arm has no meta step: with fixed targets and uniform weights there is nothing for
phi to learn.

`model_fit` and `model_entropy` are reimplemented in `llm/losses.py` rather than imported from
`wamal/networks/utils.py`. That module imports `dataset_loaders.cifar10`, `train.model.performance`
and torchvision at module scope, which would drag vision dependencies onto a fresh GPU box. The
maths is copied verbatim, and a unit test asserts numerical equality with the original.

**Metrics.** One JSON object per epoch written to `metrics.jsonl`, carrying train/val/test loss
and accuracy for both the primary and auxiliary heads, plus the auxiliary weight distribution
(`train_weight_mean`, `_std`, `_min`, `_max`) for the arms that learn weights. The std is the load
-bearing one: it is the spread across samples within an epoch, and if it reaches zero then every
sample has the same weight and WAMAL is MAXL with a rescaled auxiliary loss. The mean cannot show
that, since a per-sample-constant weight is free to drift across epochs.

The file is replaced, not appended, when a cell re-runs, with the prior attempt preserved as
`metrics.jsonl.prev`. There is no within-run resume — an interrupted cell restarts from epoch 0 —
so appending left two runs concatenated under one epoch axis that jumped backwards partway
through, which every epoch-indexed analysis read as one implausible curve. This replaces `epoch_performances.pkl`:
a pickle of a dataclass is fragile across environments, and nothing in the repo reads it back
into a table anyway. Auxiliary fields are populated rather than hardcoded to 0.0 as the dense
trainer does.

**Model selection.** Track best *validation* primary accuracy; record test accuracy at that
epoch. Do not select on test, which is what `train_wamal_network` currently does.

Additionally persist, on a configurable interval (default: first, last, every 10th epoch):

- `weights_epoch{N}.npz` — raw sigmoid weights per training example (for 8.3)
- `sublabels_epoch{N}.npz` — argmax generated sub-label per training example (for 8.4)

On completion write a `DONE` marker file. `sweep.py` uses it to skip finished cells, which is
what makes an interrupted spot instance resumable.

---

## 7. Run matrix

### 7.1 Phase 0 — calibration (OPT-125M, CLINC150)

Before any comparison runs. Sweep primary LR over
`{1e-3, 3e-3, 1e-2, 3e-2}` for SGD and `{1e-5, 5e-5, 1e-4, 5e-4}` for AdamW, on arms 1 and 4
only, at `k = 100`, 2 seeds. Pick per-optimizer defaults and freeze them for all later phases.

This exists because the paper found WAMAL "extremely sensitive" to LR under Adam
(Appendix D, Table 8), and LoRA changes the effective scale again. Skipping this risks
attributing an optimizer artifact to the method.

Also validate here: the shared-backbone assertion holds, memory is ~1x base not 2x, and the
second-order inner step runs without OOM.

### 7.2 Phase 1 — signal check (OPT-125M)

| Axis | Values | Count |
|---|---|---|
| Arms | 1-4 | 4 |
| Shots `k` | 5, 10, 25, 50, 100 | 5 |
| Seeds | 0-9 | 10 |
| Dataset | CLINC150 | 1 |

**200 runs.** At `MAX_LEN = 64` on <=15k examples with a 125M frozen backbone, each is minutes;
the whole phase fits on two 2080Tis.

**Go/no-go:** if arm 4 does not beat arm 2 (human hierarchy) at `k = 5` and `k = 10`, stop and
revisit the framing before spending 4090 time. That is claim C2, the headline.

### 7.2b Phase 1 low-k — headroom probe (`phase1_lowk`)

The first phase-1 results put arm 4 roughly 1pp *below* arm 1 at every `k`, on a primary task
already at 90.6% with `k = 5` and 97.7% at `k = 100` — about 7pp of total dynamic range, against a
0.9pp benefit from the true human hierarchy. One reading is that there is too little headroom for
learned auxiliary labels to demonstrate anything.

This probe tests that directly by extending the `k` grid downward: `k in {1, 2}`, same dataset,
model, arms and 10 seeds, so the two grids read as one curve. **80 runs.**

Note the outcome is informative either way, and the negative direction more so. If arm 4's deficit
shrinks or reverses where accuracy is genuinely low, headroom was the constraint and there is a
regime where the method works. If the deficit persists at `k = 1`, headroom is ruled out and the
explanation lies in the mechanism — the meta objective saturating (see 10.x) or the label network
lacking the data to find ψ sub-classes per domain.

These cells are *more* expensive than `k = 5`, not less: measured per-epoch cost is about
`3.0 + 0.33k` seconds, so the larger epoch budgets that low `k` requires outweigh the smaller
training set.

### 7.3 Phase 2 — main result

Same grid at 5 seeds, on Granite-4.0-350M-Base and OLMo-2-1B-Base, both datasets.

`4 arms x 5 shots x 5 seeds x 2 models x 2 datasets = 400 runs.` LoRA + short sequences on the
4090s.

### 7.4 Phase 3 — robustness (opportunistic)

Qwen2.5-0.5B (cross-family), OPT-1.3B (contamination-controlled scale), Granite dense vs hybrid
at 350M (architecture). Reduced grid: arms 1, 3 and 4, `k in {5, 25, 100}`, 3 seeds.

---

## 8. Outputs and analysis

### 8.1 Run directory

`llm/paths.py` owns naming; `utils/path_name.create_path_name` is **not** reused. That function
takes 16 required positional parameters with no defaults, 46 existing call sites already crash
on it, and its fields (`observation_feature_dimensions`, `train_ratio`, `aux_weight`) are
vision/RL leftovers that mean nothing here. Adding a 47th fragile call site would couple this
work to a repo-wide fix that should happen independently.

Layout, rooted at `$RESULTS_DIR` (default `./results`, overridable so vast.ai runs can write to
a mounted volume):

```
results/<dataset>/<model_slug>/<arm>/k<shots>/seed<seed>/
    config.json        the full resolved ExperimentConfig
    metrics.jsonl      one record per epoch; replaced on a re-run
    metrics.jsonl.prev the previous attempt, if this cell was interrupted
    summary.json       best-val epoch, test acc at that epoch, wall time, peak memory
    print.log
    weights_epoch*.npz
    sublabels_epoch*.npz
    best_model.pt      LoRA + heads only
    DONE               completion marker; presence means sweep.py skips this cell
```

Flat, greppable, and one directory per cell, so `aggregate.py` can walk it without parsing a
120-character encoded filename.

### 8.2 Headline figure

Accuracy vs examples-per-fine-class (`k` on a log x-axis), one line per arm, mean with
standard-deviation band over seeds, one panel per model.

This is the figure. It tests C1-C3 simultaneously and supplies the error bars the paper's
NeurIPS checklist currently claims without providing.

### 8.3 Weight analysis

- Histogram of raw sigmoid weights per (model, dataset, k), mirroring paper Figure 6.
- Highest and lowest weighted training utterances, mirroring Figure 7.

The text version of Figure 7 doubles as the **data-selection** framing: the weight head is a
learned per-example utility scorer, which connects the method to current work on data curation
for LLM training. Cheap to produce; likely high narrative value.

### 8.4 Do learned sub-labels recover human ones? (new)

Because CLINC150 supplies ground-truth fine labels, compute agreement between the argmax
generated sub-label and the true intent, *within each primary domain*:

- normalized mutual information
- cluster purity
- the 15x15 contingency matrix per domain, as a heatmap for one or two domains

Three possible outcomes, all publishable:

| Result | Interpretation |
|---|---|
| High agreement | The label network rediscovers the human hierarchy; WAMAL's win then comes from the *weights*, and arms 2 and 4 should differ mainly by the weight head |
| Low agreement, WAMAL still wins | The learned partition is better than the human one — the paper's strongest possible claim |
| Low agreement, WAMAL loses | The mask is doing the work, not the labels; report honestly |

CIFAR-100-20 could have supported this analysis and the paper only answers it qualitatively.
This is the single most novel analysis available from this setup.

### 8.5 Aggregation

`analysis/aggregate.py` walks `./trained_models/`, loads each `epoch_performances.pkl` and
`parameters.json`, and emits a tidy dataframe keyed by
`(model, dataset, arm, k, seed) -> best/final test accuracy`.

No such aggregator exists in the repo today. Without it the multi-seed design is unusable, so
this is a phase-0 deliverable, not an afterthought.

---

## 9. Implementation order

1. `seeding.py`, `config.py`, `args.py` — plumbing, including the seed control the repo lacks.
2. `datasets/clinc150.py` + `fewshot.py` + `collate.py`; assert the `fine // psi == coarse`
   invariant.
3. `networks/backbone.py` + `lora.py`; verify shared-base memory on OPT-125M.
4. `networks/wamal_lm.py`; verify `functional_call` overrides resolve against LoRA + head names.
5. `train_text.py`, arms 1 and 4 only.
6. `analysis/aggregate.py`.
7. Phase 0 calibration.
8. Arms 2 and 3; `newsgroups.py`.
9. Phase 1.

Steps 3-4 are the risk concentration (see 10.1). Prototype them on OPT-125M before writing the
training loop.

---

## 10. Risks

### 10.1 Shared base + two adapters + `functional_call` (highest)

The inner step must evaluate the theta-network at `theta+` via `functional_call` with overrides
naming only LoRA and head parameters, while the frozen base resolves normally and the
phi-network's adapter is untouched. The existing wrappers do prefix-stripping
(`k.split('backbone.', 1)[1]`) that assumes a simple tree; a shared base with two sibling
adapter sets is a different shape.

*Mitigation:* hand-rolled LoRA with explicit naming rather than `peft`, and a standalone unit
test that checks a `functional_call` forward at perturbed LoRA weights differs from the
unperturbed forward and that gradients flow to `phi`. Build this before the training loop.

*Fallback:* two separate LoRA-wrapped backbone copies (2x memory) at 125M-350M only, deferring
the 1B runs until the shared path works.

### 10.1a Fused attention has no double-backward (found during implementation)

The outer step differentiates through the inner SGD update, so the backward pass
must itself be differentiable. The fused attention kernels are not:

```
RuntimeError: derivative for
aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented
```

The CUDA flash and mem-efficient backends fail the same way, so this is not a
CPU-only artifact — it would have taken down the first real sweep partway through
epoch 1.

*Resolved:* backbones load with `attn_implementation="eager"`, and
`ExperimentConfig.resolve()` refuses to construct a meta-step arm with anything
else. The cost is real and permanent: eager attention is slower and holds more
activation memory than SDPA, which raises the price of every `maxl` and `wamal`
run relative to `single`. Budget for it, and note that `single`/`human` could run
with SDPA — though letting the arms differ there would trade a fairness guarantee
for throughput, which is the wrong trade.

### 10.2 Optimizer/LR interaction

The paper's SGD defaults were tuned for full fine-tuning of VGG/ViT; LoRA changes effective
scale. Mitigated by phase 0. Do not carry `PRIMARY_LR = 5e-4` over on the assumption that it
transfers.

### 10.3 Contamination

Granite (15T tokens, 2025) has very likely seen CLINC150. OPT-125M/1.3B (~180B tokens, 2021
cutoff) are the control. If the arms separate on OPT but not on Granite, contamination is a
live explanation and must be discussed rather than buried.

### 10.4 Focal loss on softmax outputs

`model_fit` applies focal loss to already-softmaxed probabilities with a `log(p + 1e-20)`
guard. This is numerically shakier than logits + `cross_entropy`, and bf16 on an LLM backbone
is a harsher setting than fp32 on VGG. Watch for NaNs at low `k`; if they appear, keep the
formulation (for faithfulness) but compute the loss in fp32.

### 10.5 Very low `k`

At `k = 5`, CLINC150 has 750 training examples total. Ten epochs of a 1B model over 750 short
utterances may overfit before the bilevel dynamics matter. Treat `TOTAL_EPOCH` as
`k`-dependent, tuned once in phase 0, and held identical across arms within a cell.

### 10.6 Ephemeral, metered compute

Runs happen on rented vast.ai instances billed by the hour, not on a persistent lab cluster.
That changes several assumptions the vision code makes:

- nothing is pre-installed or pre-downloaded; every run starts from a bare image;
- instances can be interrupted, so long unresumable sweeps lose money;
- anything not copied off the box before teardown is gone;
- debugging time is billed at GPU rates, so failures must surface in seconds, not minutes.

Addressed in section 11.

---

## 11. Deployment (vast.ai)

### 11.1 Assumptions

Target image: any CUDA + PyTorch base (e.g. `pytorch/pytorch:2.*-cuda12.*-cudnn*-runtime`).
We install on top with `pip` rather than creating a conda env — conda solves are slow and we
are paying for the wait. The repo's `environment.yaml` and `auxilearn_env.yaml` are **not**
used here; they pin Python 3.11/3.8 with SB3, gym, and TensorFlow-datasets, none of which this
experiment needs.

Two directories matter:

| Env var | Default | Purpose |
|---|---|---|
| `HF_HOME` | `/workspace/.cache/huggingface` | model + dataset cache; put on the persistent volume |
| `RESULTS_DIR` | `/workspace/results` | run outputs; must survive until synced |

Defaulting both under `/workspace` matters: on most vast.ai templates that is the mounted
volume, while `/root` and the container filesystem are ephemeral. Caching models there means a
re-rented instance with the same volume skips the download entirely.

### 11.2 Bootstrap

One command from a bare box:

```bash
git clone <repo> && cd rl-aux && bash llm/bootstrap/bootstrap.sh --tier phase1
```

which runs, in order:

1. **`setup_env.sh`** — `pip install -r llm/requirements.txt`. Installs torch only if absent,
   so we inherit the image's CUDA-matched build rather than fighting it.
2. **`download_models.py`** — `snapshot_download` for the tier's models, then a check that a
   weight file actually landed. Tiers (`smoke`, `phase1`, `phase2`, `phase3`, `all`) exist so a
   phase-1 box does not pull 1B weights it will never load.

   The verification is not defensive padding. Filtering the download to safetensors leaves
   `.bin`-only repos — `facebook/opt-125m` among them — with a cached config and tokenizer, no
   weights, and a zero exit status. The failure would otherwise surface much later and much
   more expensively.
3. **`download_data.py`** — CLINC150 via `datasets`, 20 Newsgroups via `sklearn`, both into the
   cache. Also validates the domain hierarchy asset (section 6.1) while the box is cheap.
4. **`preflight.py`** — the gate before any real spend.

Every step is idempotent, so re-running after an interruption is safe and mostly a no-op.

### 11.3 Preflight

`preflight.py` must fail loudly in under a minute rather than 20 minutes into a sweep. It
checks, in order:

1. CUDA is visible; prints GPU name, total memory, driver, torch version;
2. every model in the tier loads from cache with no network access
   (`HF_HUB_OFFLINE=1`), confirming step 2 actually worked;
3. the datasets load and the hierarchy assertions pass;
4. a real end-to-end run: `k=5`, 2 epochs, arm `wamal`, smallest model — exercising the
   dual-adapter forward, the second-order inner step, and the writer;
5. asserts no NaN in any loss, reports peak memory, and asserts the shared-base invariant
   (`~1x` base resident, not `2x`).

Step 4 is the important one. It is the cheapest possible instance of the thing most likely to
break (risk 10.1), and it runs in seconds on OPT-125M.

### 11.4 Sweeps and interruption

`sweep.py` enumerates cells from a phase name, skips any with a `DONE` marker, and runs the rest
sequentially, or across GPUs with `--gpus 0,1,2,3`. Because completion is a filesystem marker
under `RESULTS_DIR`, an interrupted instance resumes by re-running the identical command.

`--pack N` runs N cells concurrently on each GPU, so `--gpus 0,1,2,3 --pack 2` gives eight
concurrent slots. This is worth doing because these cells barely load the device: at OPT-125M with
64-token sequences, `nvidia-smi` shows ~1.8GB of 46GB and single-digit utilisation, meaning the GPU
is idle most of the time while the host issues small kernels. Packing interleaves cells into those
gaps. The limits are host-side, not VRAM: each cell is a process with its own dataloader workers,
so `gpus x pack x (1 + num_workers)` must stay inside the container's core and pid budget. `sweep.py`
prints that count and warns when it exceeds the visible cores; `--num-workers` tunes it down.

Multi-GPU mode is one worker thread per slot pulling from a shared queue rather than a static
partition of the cell list: cells differ in cost by an order of magnitude (a k=100 `wamal` cell
against a k=5 `single` one), so a static split would leave most GPUs idle waiting on the unlucky
worker. Per-cell stdout goes to `sweep.log` inside each run directory, since interleaving several
training logs on one stream is unreadable.

Every cell runs in its own subprocess. A CUDA OOM or a wedged allocator would otherwise poison
every cell after it, and losing a 200-cell sweep to cell 12 is the expensive failure mode.

`--dry-run` prints the cell list and count without executing. It also estimates GPU-hours from
the wall times recorded in existing `summary.json` files, so each phase prices the next one from
measurement rather than guesswork; with no prior runs it says so instead of inventing a number.

### 11.5 Getting results off the box

Two paths, depending on whether the box is reachable over SSH:

- `deploy/sync.sh pull` (from the laptop) rsyncs `RESULTS_DIR` into `./vast_artifacts`.
  Incremental and resumable, so a dropped connection costs the current file rather than the whole
  transfer, and re-running after more cells finish moves only the new ones. This is the normal
  path. `deploy/sync.sh status` reports remote progress without transferring anything.
- `sync_results.sh` (on the box) tars `RESULTS_DIR` into a timestamped archive for when only a
  web terminal or `vastai copy` is available.

Both exclude checkpoints by default (`--with-checkpoints` to include them) since `metrics.jsonl`
plus the `.npz` dumps are all the analysis needs and are ~1000x smaller.

**Pull before destroying any instance.** Nothing outside the mounted volume survives teardown.
`RUNBOOK.md` is the operational procedure; this section is the rationale.

### 11.6 Cost discipline

- Phase 0 and 1 target the cheapest available GPU; OPT-125M with `MAX_LEN=64` is not
  compute-bound and a 3090/4090-class card is ample.
- Prefer one long-lived instance running `sweep.py` over many short ones: model download and
  environment setup are per-instance overhead, not per-run.
- `--dry-run` first, always.
- `summary.json` records wall time per run, so after phase 1 the phase-2 bill is an
  extrapolation rather than a guess.

---

## 12. Open decisions

| # | Decision | Default if unresolved |
|---|---|---|
| D1 | Pooling: last token vs mean | last |
| D2 | Optimizer: SGD (paper-faithful) vs AdamW (LoRA-conventional) | SGD primary, AdamW reported as variant |
| D3 | LoRA targets: `q,v` vs `q,k,v,o` vs all-linear | `q,v`, `r=16` |
| D4 | Separate LoRA for the label net vs a shared adapter with separate heads only | separate LoRA |
| D5 | Report best-epoch or final-epoch test accuracy | best-**validation** epoch; see below |
| D6 | Whether to build variable-`psi` support now (unlocks DBpedia/WOS/ultra-fine typing) | defer to a follow-up |
| D7 | Whether the CLINC150 domain grouping in the JSON asset matches Larson et al. Table 5 | ships best-effort + set-equality check; eyeball once |

D5 is worth flagging: selecting the best epoch on the *test* split is what
`train_wamal_network` currently does (`if test_accuracy_primary > best_training_performance`).
That is test-set selection. For this work, select on a validation split and report test at that
epoch. CLINC150 ships a 20-per-intent validation split, so there is no reason to inherit the
vision code's behavior here.

D7 is the one factual dependency that cannot be machine-verified end to end. The load-time
check proves the 150 intent *names* are exactly right; it cannot prove each is filed under the
correct domain. Reviewing `llm/datasets/clinc150_domains.json` once, before the first real run,
closes this out permanently.
