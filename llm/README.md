# WAMAL-LM

Weight-Aware Meta Auxiliary Learning for few-shot text classification on small
decoder-only base LLMs.

The vision work learns auxiliary sub-labels *and* per-sample auxiliary loss
weights through bilevel optimization. This ports that to language, in the regime
where auxiliary supervision should matter most: few-shot, with a frozen backbone
and LoRA adapters.

Three documents, three jobs:

- `SPEC.md` — the design: experiment matrix, claims, open decisions.
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
# several onto each card: 4 GPUs x 2 = 8 concurrent
python -m llm.scripts.sweep --phase phase2 --gpus 0,1,2,3 --pack 2 --keep-going
```

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
config.json      the fully resolved config
metrics.jsonl    one record per epoch
summary.json     best-val epoch, test acc there, wall time, peak memory
print.log
dump_epoch*.npz  per-sample weights and generated sub-labels
best_model.pt    LoRA + heads only, never the frozen backbone
DONE             completion marker
```

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
```

Run `curves --check` first. `aggregate` answers "which arm won" from one row per
run; `curves` answers "did that run train sanely" from every epoch of it, which
is the question you need settled before believing any headline number. It reads
`metrics.jsonl`, works on runs still in progress, and flags overfitting (val
peaking early — expected at k=5, where 40 epochs pass over ~750 examples), a
diverging learning rate, an undertrained epoch budget, a meta step that isn't
lowering the primary loss, and a mean auxiliary weight that has gone flat.

That last one is the quiet failure: if the weights collapse to a constant, WAMAL
has silently become MAXL and will still report a plausible accuracy. Nothing in
`summary.json` would show it.

`aggregate` pairs arms by `(dataset, model, shots, seed)` rather than comparing
group means. The few-shot sampler is a pure function of those coordinates, so all
four arms in a cell train on byte-identical data and the paired comparison is the
statistically correct one.

`sublabels` is the analysis the vision paper could not run: CLINC150 has real
fine labels, so "did the label network rediscover the human hierarchy?" is a
measurement rather than a qualitative claim about blurry images.

---

## Tests

```bash
python -m llm.tests.test_lora_functional   # adapters, functional_call, second-order path
python -m llm.tests.test_train_loop        # all four arms end to end
python -m llm.tests.test_sweep             # enumeration, resumption, cost, multi-GPU queue
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
