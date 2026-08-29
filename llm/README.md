# WAMAL-LM

Weight-Aware Meta Auxiliary Learning for few-shot text classification on small
decoder-only base LLMs.

The vision work learns auxiliary sub-labels *and* per-sample auxiliary loss
weights through bilevel optimization. This ports that to language, in the regime
where auxiliary supervision should matter most: few-shot, with a frozen backbone
and LoRA adapters.

Four documents, four jobs:

- `SPEC.md` — the design: experiment matrix, claims, open decisions.
- [`FINDINGS.md`](FINDINGS.md) — **what the runs actually showed.** Start here if
  you want results rather than mechanics: phase 1 is complete at 10 seeds, and two
  of the three hypotheses for WAMAL's deficit are now refuted by measurement.
- [`../RUNBOOK.md`](../RUNBOOK.md) — the operational lifecycle on a rented GPU,
  from provisioning to teardown, with a troubleshooting section.
- this file — the code tour and the commands.

---

## Quickstart on a fresh vast.ai box

```bash
git clone <repo> && cd rl-aux
bash llm/bootstrap/bootstrap.sh --tier phase1
```

That installs dependencies (repairing a torch/driver mismatch if the image has
one), downloads the tier's model weights and both datasets into the persistent
cache, and runs a real two-epoch WAMAL job as a preflight check. It is
idempotent — safe to re-run after an interruption.

Then:

```bash
# always price it first: prints the cell list, and GPU-hours if prior runs exist
python -m llm.scripts.sweep --phase phase1 --dry-run

python -m llm.scripts.sweep --phase phase1 --gpu 0

# these cells are host-bound, not GPU-bound (~1.8GB of 46GB on an L40S), so pack
# several onto each card: 4 GPUs x 4 = 16 concurrent saturated an L40S box
python -m llm.scripts.sweep --phase phase2 --gpus 0,1,2,3 --pack 4 --keep-going
```

Phases are `smoke`, `phase0` (LR calibration), `phase1` (signal check, 200 cells),
`phase1_holdout` (saturation control, 40), `phase1_intent` (150-way primary with an
invented hierarchy, 45), `phase1_ladder` (label-quality ladder, 40), `phase1_lowk`
(extends phase 1's k grid down to 1 and 2 shots, 80), `phase2` (main result, 400)
and `phase3` (robustness).

**See [`FINDINGS.md`](FINDINGS.md) for results.** `phase1` is complete at 10 seeds
and `phase1_holdout` is 26/40; the short version is that WAMAL never beat single,
and the two mechanisms we suspected are both ruled out.

`phase1_holdout` was the saturation control, and it settled the question. The meta
objective is the primary loss on the batch theta was just fitted on, so it decays
toward zero early — signal survives only 15% of a k=5 run, rising to 65% at k=100,
the same ordering as wamal's deficit (−1.04pp at k=5 to −0.01pp at k=100). Moving
the outer loss onto withheld data (`--meta-holdout-frac 0.2`) fixed that
completely, to 100% of epochs at both k=5 and k=10. **The deficit did not move**
(−0.92pp at k=5), and recovery got slightly worse rather than better. Saturation
was real and was not the cause. The withheld examples leave theta's training set
for *every* arm, so the baselines stay matched.

`phase1_intent` is the response to the finding that actually holds up: an oracle
hierarchy on the 10-way domain task is worth only +0.3 to +1.0pp, which is too low
a ceiling for any label-generating method to clear by much. It makes all 150
intents the primary task with 5 invented sub-classes each — far harder, and the
auxiliary hierarchy is invented rather than recovered, which is MAXL's original
setting. No `human` arm: nothing exists below an intent, so the config rejects it
rather than letting it train on a constant target. That also removes the ground
truth, so read `recovery --inter-seed` instead of a recovery score.

`phase1_ladder` asks how good learned labels would have to be, by corrupting the
human labels toward chance-within-block (`--label-corruption`) and reading the gain
against label quality. Cheap, since the human arm has no meta step. A steep curve
near zero means quality is the binding constraint; a flat one means even good
labels don't transfer — and §4 of `FINDINGS.md` already points at flat, since phi
recovers most structure at k=100 where the oracle is worth least.

**Do not read `aux_recovery` from `summary.json` for the maxl/wamal arms.** It is
`argmax == fine`, but phi's clusters are only identified up to a permutation
within each coarse block, so it scores 1/psi on a *perfect* recovery — 0.0600,
which is exactly what phase 1 reported. Use `python -m llm.analysis.recovery`,
which does Hungarian matching and mutual information against a within-block
shuffled null. Corrected, phi turns out to recover real structure: 1.14x the null
at k=5 rising to 6.72x at k=100. The naive number is still valid for the human arm,
whose targets are the true fine labels, and is reported as `null` on the
invented-hierarchy tasks where `fine` is a placeholder.

`phase1_lowk` asks a different question — phase 1 runs on a task already at 90.5%
at k=5, leaving ~7pp of range — but `phase1_intent` is now the better way in, since
it adds headroom without starving the meta signal further. Phase 1's
deficit shrinks monotonically as k grows, which is backwards from a headroom story,
and at k=1 the meta signal is starved harder still, so on its own the probe cannot
separate the two explanations. Its cells also cost *more* than k=5, not less:
per-epoch cost is about `3.0 + 0.33k` seconds, so the larger epoch budget outweighs
the smaller train set.

**Before destroying the instance:**

```bash
deploy/sync.sh pull                  # from your laptop; rsync, resumable
# or, on the box, if you only have a web terminal:
bash llm/bootstrap/sync_results.sh
```

Nothing outside the mounted volume survives teardown.

### Driving it from your laptop

`deploy/` wraps the SSH and rsync details so you don't retype them:

```bash
cp deploy/env.example deploy/env && $EDITOR deploy/env   # host, port, key

deploy/remote.sh --clone                                 # code onto the box
deploy/remote.sh 'bash llm/bootstrap/bootstrap.sh --tier phase1'
deploy/remote.sh --gpu                                   # nvidia-smi
deploy/remote.sh --logs                                  # follow the newest run
deploy/sync.sh status                                    # cells done, disk, GPU use
deploy/sync.sh pull                                      # results -> ./vast_artifacts
```

See [`../RUNBOOK.md`](../RUNBOOK.md) for the full sequence including GPU
selection and recovery from a dead instance.

### Where things go

| Env var | Default | Contents |
|---|---|---|
| `HF_HOME` | `/workspace/.cache/huggingface` | model weights, dataset cache |
| `RESULTS_DIR` | `/workspace/results` | run outputs |
| `WAMAL_DATA_DIR` | `$HF_HOME/wamal_data` | raw dataset files |
| `WAMAL_NUM_THREADS` | `4` | per-process BLAS/OpenMP cap |

The paths default under `/workspace` because on most vast.ai templates that is
the mounted volume while the container filesystem is ephemeral. Reattaching the
same volume to a new instance skips the model download entirely. Off a vast.ai
box they fall back to `~/.cache` and `./results`.

The thread cap is not a performance knob so much as a survival one. Rented boxes
are containers with a small pid budget on a large host, and uncapped OpenBLAS
spawns one thread per *host* core, times every dataloader worker — the container
hits its pid ceiling and you get a `BrokenPipeError` on worker fork that looks
like a data bug. `env.configure()` sets the caps, and the entry point calls it
before importing torch or numpy, because the BLAS libraries read their thread
counts once at load time.

---

## Running one cell

```bash
python -m llm.scripts.run_experiment --arm wamal --shots 10 --seed 0
```

Four arms, one code path:

| `--arm` | Auxiliary target | Weights | Meta step | Question it answers |
|---|---|---|---|---|
| `single` | none | — | no | does the auxiliary task help at all? |
| `human` | true fine labels | uniform | no | can a *learned* hierarchy beat the real one? |
| `maxl` | learned | uniform | yes | does learning labels help? |
| `wamal` | learned | learned | yes | does per-sample weighting add anything on top? |

Arm behaviour is derived from `--arm` inside `ExperimentConfig`, never passed as
separate flags. In this repo `wamal/dense_scripts/vit_coco_single_dense.py` is
configured identically to its WAMAL twin, so the published COCO "single-task"
baseline was actually running WAMAL. One entry point with derived behaviour makes
that failure mode unrepresentable rather than merely unlikely.

Every run writes to `$RESULTS_DIR/<dataset>/<model>/<arm>/k<shots>/seed<seed>/`:

```
config.json          the fully resolved config
metrics.jsonl        one record per epoch, replaced (not appended) on a re-run
metrics.jsonl.prev   the previous attempt, if this cell was interrupted and rerun
summary.json         best-val epoch, test acc there, wall time, peak memory
print.log            appended across attempts
dump_epoch*.npz      per-sample weights and generated sub-labels
best_model.pt        LoRA + heads only, never the frozen backbone
DONE                 completion marker
```

An interrupted cell restarts from epoch 0, so `metrics.jsonl` is replaced rather
than appended to: two runs concatenated in one file, with the epoch counter
jumping backwards partway through, reads as a single implausible curve to every
epoch-indexed analysis. The old file is kept as `metrics.jsonl.prev`.

`DONE` is what makes sweeps resumable: `sweep.py` skips any cell that has one, so
an interrupted spot instance resumes by re-running the identical command.

---

## Tracking (optional)

Opt-in, on any run or sweep:

```bash
python -m llm.scripts.run_experiment --arm wamal --shots 10 --wandb
python -m llm.scripts.sweep --phase phase2 --gpus 0,1,2,3 --wandb
```

Sweeps group by phase and split by arm, so 400 cells arrive as one comparable
experiment. Run ids are derived from the cell id, so re-running a cell after a
crash resumes that run instead of leaving a duplicate behind — which matters
because the whole sweep is built to be resumable.

It is a mirror, never a dependency. `metrics.jsonl` is written first and every
analysis reads it, so a missing, unauthenticated, or mid-run-failing wandb costs
visibility and nothing else. With no key, `--wandb` records offline into the run
directory and `wandb sync` uploads it later. `--wandb-mode online` forces the
opposite: fail loudly rather than fall back.

`bash llm/bootstrap/setup_secrets.sh` prompts for the key and persists it; the
bootstrap runs it for you.

---

## Analysis

```bash
python -m llm.analysis.curves --check         # per-epoch health, text only
python -m llm.analysis.curves                 # training curves, arms overlaid per cell
python -m llm.analysis.aggregate --figure     # tables + accuracy-vs-k figure
python -m llm.analysis.weights --examples 10  # weight distributions, extreme texts
python -m llm.analysis.sublabels              # NMI/purity vs human fine labels

# Chance-corrected sub-label recovery. Use this, not summary.json's aux_recovery.
python -m llm.analysis.recovery --arm wamal --psi 15

# Where the auxiliary hierarchy is invented and has no ground truth, ask instead
# whether independent seeds arrive at the same partition of the same test examples.
python -m llm.analysis.recovery --inter-seed --dataset clinc150_intent --psi 5
```

Run `curves --check` first. `aggregate` answers "which arm won" from one row per
run; `curves` answers "did that run train sanely" from every epoch of it, which
is the question you need settled before believing any headline number. It reads
`metrics.jsonl`, works on runs still in progress, and flags overfitting (val
peaking early — expected at k=5, where 40 epochs pass over ~750 examples), a
diverging learning rate, an undertrained epoch budget, a meta step that isn't
lowering the primary loss, and auxiliary weights that no longer vary between
samples.

That last one is the quiet failure: if every sample gets the same weight, WAMAL
has silently become MAXL with a rescaled auxiliary loss and will still report a
plausible accuracy. Nothing in `summary.json` would show it. The check is on
`train_weight_std`, the spread *across samples within an epoch* — not on the
mean, since a per-sample-constant weight that drifts over training moves the mean
while still being pure MAXL.

`aggregate` pairs arms by `(dataset, model, shots, seed)` rather than comparing
group means. The few-shot sampler is a pure function of those coordinates, so all
four arms in a cell train on byte-identical data and the paired comparison is the
statistically correct one.

`sublabels` is the analysis the vision paper could not run: CLINC150 has real
fine labels, so "did the label network rediscover the human hierarchy?" is a
measurement rather than a qualitative claim about blurry images.

`recovery` is the one to trust for that question, because a generated clustering
is identified only up to a permutation within each coarse block. It reports
Hungarian-matched accuracy and mutual information against a shuffled-within-block
null, so a perfect recovery under a permuted labeling scores high instead of
scoring at chance — which is what `aux_recovery` does and why phase 1 was briefly
read as "phi learned nothing." With `--inter-seed` it compares two runs' partitions
of the test split to each other instead of to a ground truth, which is the only
recovery evidence available on the invented-hierarchy tasks.

---

## Tests

```bash
python -m llm.tests.test_lora_functional   # adapters, functional_call, second-order path
python -m llm.tests.test_train_loop        # all four arms end to end
python -m llm.tests.test_sweep             # enumeration, resumption, cost, GPU packing
python -m llm.tests.test_curves            # diagnostics, incl. WAMAL-vs-MAXL collapse
python -m llm.tests.test_tracking          # wandb can never break a run
python -m llm.tests.test_losses            # numerical equality with the vision code
python -m llm.tests.test_recovery          # calibrates the chance-corrected metrics
python -m llm.tests.test_meta_holdout      # holdout split and label-corruption mechanics
python -m llm.tests.test_intent_task       # invented-hierarchy guards
```

CPU-only, no network, a few seconds. Run them before renting anything.

The one that matters is `test_second_order_gradient_reaches_phi`. The label
network receives gradient *only* through the differentiable inner step: if
`create_graph` were dropped or an override name were wrong, phi's gradients would
silently be zero and every WAMAL run would quietly degrade into MAXL with a
frozen label network — producing plausible numbers and a false conclusion.

---

## Design notes

**Shared frozen backbone, two LoRA adapters.** One copy of the base model in
memory, wrapped so each targeted `nn.Linear` carries both a `theta` and a `phi`
adapter and switches between them by attribute. Two full wrapped copies would
double backbone memory; merging `W + sBA` per forward would materialise a
full-size weight per layer and defeat the point of LoRA.

**One module owns everything.** `torch.func.functional_call` resolves override
names relative to a single module, so the primary and label networks cannot be
separate objects sharing a base — there would be no well-defined name root. Hence
`WamalLM` with a `mode` argument and two disjoint parameter groups.

**Eager attention is mandatory.** The outer step differentiates through the inner
update, so the backward pass must itself be differentiable, and the fused SDPA
kernels have no double-backward on CPU or CUDA. `resolve()` refuses to build a
meta-step arm with anything else. This costs throughput on every `maxl`/`wamal`
run and is a permanent tax on the method.

**Selection on validation, not test.** `train_wamal_network` picks the best epoch
by test accuracy. CLINC150 ships a validation split, so there is no reason to
inherit that.

**Seeded subsampling.** `dataset_loaders/capped_dataset.py` samples with an
unseeded `random.sample`, so vision arms at the same nominal cap saw different
examples. Here the subset is a pure function of `(seed, dataset, k)`.

---

## Known caveat

`llm/datasets/clinc150_domains.json` transcribes the 10x15 domain grouping from
Larson et al. 2019 Table 5, because HuggingFace ships the intents without the
domains and CLINC never released the grouping in machine-readable form.

Load-time validation confirms the 150 intent *names* are set-equal to the
dataset's own labels, and that check passes. It cannot confirm that each intent
sits under the *right* domain. Read the JSON once before the first real run
(SPEC D7); it is 150 lines and the groupings are self-evident.
