# new_wamal

One configurable trainer for the CVPR experiments, replacing the per-setup
scripts in `wamal/` (which stay untouched for reproducing the paper).

## What changed relative to `wamal/train_network.py`

| | old | new |
|---|---|---|
| reported number | max test accuracy over epochs (checkpoint chosen on **test**) | test accuracy at the best **validation** epoch; also final epoch and last-5 mean |
| validation split | none | 10% of train, stratified, fixed `split_seed` so every method/seed sees the same split |
| accuracy | correct / nominal batch size | correct / total |
| gradient cosine-similarity logging | every step, all arms (2 extra backward passes) | removed |
| single-task | still built and ran the generator | no generator, no meta step (fair timing) |
| generator in the inner step | full graph built, then discarded | `no_grad` (same theta update, less memory) |
| cost logging | none | per-epoch inner/meta/eval seconds, peak GPU memory |
| per-sample data | none | `dump_epochNNN.npz`: index, raw weight u, primary loss, aux loss, aux assignment |
| configs | ~40 copied scripts | `--dataset --arch --method ...` flags; grids in `experiments.py` |

The bi-level update, focal losses, entropy term, weight scaling
`2**(2*r*u - r)`, masked softmax and LR schedule are unchanged, and the
primary network is the original `WamalWrapper`.

## Methods (`--method`)

| method | auxiliary labels | auxiliary weights |
|---|---|---|
| `single` | none | none |
| `maxl` | learned (masked softmax) | 1 |
| `wamal` | learned | per-sample head |
| `wamal_nomask` | learned, plain softmax | per-sample head |
| `global_weight` | learned | one learned scalar |
| `class_weight` | learned | one learned weight per primary class |
| `weights_only` | fixed: k-means clusters of ImageNet ResNet-50 features (`make_cluster_labels.py`) | per-sample head |
| `random_weights` | learned | fixed random per sample, log-uniform in [2^-r, 2^r] |
| `shuffled_weights` | learned | WAMAL's weights, randomly permuted within each batch |
| `human` | CIFAR-100 fine labels (cifar100_20 only) | 1 |

## Datasets (`--dataset`)

`cifar100_20`, `cifar100`, `cifar10`, `svhn`, `oxford_pet`, `cub200`, `food101`,
`cifar10n_aggre`, `cifar10n_worst`, `cifar10n_rand1`, `cifar100n`.
Synthetic noise on any of them: `--noise-type sym|asym --noise-rate 0.4`
(training and validation labels only; test stays clean; the noise mask is
saved as `noise_mask.npy`). `--train-frac 0.3` keeps 30% of each class.

CIFAR-N label files are not downloaded automatically. Put
`CIFAR-10_human.pt` / `CIFAR-100_human.pt` from
https://github.com/UCSC-REAL/cifar-10-100n/tree/main/data in `data/cifarn/`.
CUB-200 is expected under `data/cub200/` (Kaggle layout or `CUB_200_2011/`).

## Running

```bash
python -m new_wamal.train --dataset cifar100_20 --arch vgg16 --method wamal --seed 0 --out results_new/try
python -m new_wamal.sweep --exp E0 --gpus 0,1,3      # restartable; skips cells with DONE
python -m new_wamal.aggregate --exp E0
python -m new_wamal.make_cluster_labels --dataset cifar100_20   # before weights_only
```

`sweep.py` is built for a shared server: at most 3 GPUs, one job per GPU, it
skips GPUs other users are on, runs under `nice -n 10` with 2 CPU threads and
2 dataloader workers, and stops launching when free disk drops below 8 GB.

## Grids (`experiments.py`)

| exp | question | cells |
|---|---|---|
| E0 | does WAMAL > MAXL survive validation-based selection? (gate) | 15 |
| E1 | which head matters: labels vs weights, and does *which* sample gets which weight matter? | 125 |
| E2 | per-sample vs per-class vs global weighting | 12 |
| E3 | noisy labels (CIFAR-N, synthetic sym/asym) | 90 |
| E4 | cost: small ResNet-18 generator | 6 |
| E5 | ViT data scaling (10/30/100%) | 75 |
| E7 | no-mask collapse, AdamW on ViT | 12 |

Outputs: `results_new/<exp>/<dataset>__<arch>__<method>[__tags]/seed<k>/`
with `config.json`, `metrics.jsonl`, `summary.json`, dumps and `DONE`.
No checkpoints are saved (the lab disk is nearly full).
