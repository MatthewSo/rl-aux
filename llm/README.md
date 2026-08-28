# WAMAL-LM

Weight-Aware Meta Auxiliary Learning for few-shot text classification on small
decoder-only base LLMs.

The vision work learns auxiliary sub-labels *and* per-sample auxiliary loss
weights through bilevel optimization. This ports that to language, in the regime
where auxiliary supervision should matter most: few-shot, with a frozen backbone
and LoRA adapters.

`SPEC.md` is the design document — experiment matrix, claims, open decisions.
This file is how to run it.

---

## Quickstart on a fresh vast.ai box

```bash
git clone <repo> && cd rl-aux
bash llm/bootstrap/bootstrap.sh --tier phase1
```

That installs dependencies, downloads the tier's model weights and both datasets
into the persistent cache, and runs a real two-epoch WAMAL job as a preflight
check. It is idempotent — safe to re-run after an interruption.

Then:

```bash
# always price it first
python -m llm.scripts.sweep --phase phase1 --dry-run

python -m llm.scripts.sweep --phase phase1 --gpu 0
```

**Before destroying the instance:**

```bash
bash llm/bootstrap/sync_results.sh
# then scp / vastai copy the printed archive path
```

Nothing outside the mounted volume survives teardown.

### Where things go

| Env var | Default | Contents |
|---|---|---|
| `HF_HOME` | `/workspace/.cache/huggingface` | model weights, dataset cache |
| `RESULTS_DIR` | `/workspace/results` | run outputs |
| `WAMAL_DATA_DIR` | `$HF_HOME/wamal_data` | raw dataset files |

Both default under `/workspace` because on most vast.ai templates that is the
mounted volume while the container filesystem is ephemeral. Reattaching the same
volume to a new instance skips the model download entirely. Off a vast.ai box
they fall back to `~/.cache` and `./results`.

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

## Analysis

```bash
python -m llm.analysis.aggregate --figure     # tables + accuracy-vs-k figure
python -m llm.analysis.weights --examples 10  # weight distributions, extreme texts
python -m llm.analysis.sublabels              # NMI/purity vs human fine labels
```

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
