# RUNBOOK — running WAMAL-LM on a rented GPU

End-to-end lifecycle for the `llm/` experiments on vast.ai (or any remote GPU
box): local setup → provision → clone → environment → data → preflight → launch
→ monitor → sync → teardown.

`llm/SPEC.md` is the experiment design. `llm/README.md` is the code tour. This is
the operational path.

**The short version, once you're set up:**

```bash
deploy/remote.sh --clone                                   # get the code onto the box
deploy/remote.sh 'bash llm/bootstrap/bootstrap.sh --tier phase1'
deploy/remote.sh 'cd /workspace/rl-aux && nohup python -m llm.scripts.sweep --phase phase1 > /workspace/sweep.log 2>&1 &'
deploy/sync.sh status                                      # check progress
deploy/sync.sh pull                                        # bring results home
```

---

## Division of labour

Two directories, and the split matters:

| | Runs where | Does what |
|---|---|---|
| `deploy/` | your laptop | connection, secrets, rsync, remote command helpers |
| `llm/bootstrap/` | the rented box | environment, model/data download, preflight |

`deploy/` never needs the GPU. `llm/bootstrap/` never needs to know your SSH
config. Keeping them apart means you can bootstrap a box you reached some other
way (the vast.ai web terminal, a Jupyter cell) without any of the local
scaffolding.

---

## 0. One-time local setup

### SSH key

vast.ai injects the *public* key you register into every instance you create
afterwards. Register it before provisioning — adding it later does not
retroactively fix a running box.

```bash
# skip if you already have one you want to use
ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/id_ed25519

cat ~/.ssh/id_ed25519.pub
# paste into https://cloud.vast.ai/account/ -> "SSH Keys"
```

### Credentials file

```bash
cp deploy/env.example deploy/env
$EDITOR deploy/env
```

Fill in `VAST_HOST` / `VAST_PORT` after provisioning (step 1). `deploy/env` is
gitignored; do not commit it.

### Optional: an SSH alias

If you'd rather not carry the port around, add a block to `~/.ssh/config` and set
`VAST_SSH_ALIAS=vast` in `deploy/env`:

```
Host vast
  HostName ssh5.vast.ai
  User root
  Port 41234
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 30
  ServerAliveCountMax 6
```

---

## 1. Provision an instance

### Which GPU

These runs are **not** compute-bound in the usual way. The backbone is frozen,
sequences are 64 tokens on CLINC150, and only LoRA adapters and small heads get
gradients. What actually costs you is that the meta step needs eager attention
and a second-order backward, which inflates activation memory well above what a
normal fine-tune of the same model would use.

| Phase | Models | Sensible card | Why |
|---|---|---|---|
| 0–1 | OPT-125M | anything with ≥12GB — 3090, 4090, even a 2080Ti | 125M frozen params; you are paying for wall-clock, not FLOPs. Take the cheapest. |
| 2 | Granite-350M, OLMo-2-1B | 4090 / L40S, ≥24GB | 1B in bf16 plus eager-attention activations |
| 3 | OPT-1.3B, Qwen, Granite hybrid | L40S / A100 40GB | headroom for the largest cells |

Selection notes:

- **Prefer one long-lived instance over several short ones.** Environment setup
  and model download are per-instance overhead, not per-run. A sweep is
  embarrassingly parallel but the setup is not.
- **Multi-GPU is worth it for phase 2+.** `sweep.py --gpus 0,1,2,3` runs one cell
  per GPU from a shared queue. Four cheap GPUs on one box beat one expensive one.
- **Check the disk allocation**, not just the GPU. Phase 2 pulls ~5GB of weights;
  ask for ≥40GB.
- **Verify bf16.** Turing (2080Ti) has no bf16 and silently falls back to fp32,
  which roughly doubles memory. Fine at 125M, not at 1B.
- **Avoid spot/interruptible for long sweeps** unless you have accepted that
  they get preempted — which is survivable here, because sweeps resume (see §6).

### Template

Any CUDA + PyTorch image works; `pytorch/pytorch:2.*-cuda12.*-cudnn*-runtime`
is a safe default. `setup_env.sh` repairs a torch/driver mismatch automatically,
so you don't have to match the image to the host by hand.

Make sure `/workspace` is a mounted volume. Everything persistent lives there.

Once it's running, copy the SSH host and port from the instance card into
`deploy/env`, then check the connection:

```bash
deploy/remote.sh --gpu
```

---

## 2. Get the code onto the box

```bash
deploy/remote.sh --clone
```

Clones to `WORKDIR` (default `/workspace/rl-aux`) or fast-forwards it if already
present. For a private repo, set `GITHUB_TOKEN` in `deploy/env`; it is written to
a repo-local `.git-credentials` rather than embedded in the remote URL, so it
does not leak into `git remote -v` or any log that echoes the URL.

To test uncommitted local changes without a git round-trip:

```bash
deploy/sync.sh push
```

---

## 3. Bootstrap

```bash
deploy/remote.sh 'bash llm/bootstrap/bootstrap.sh --tier phase1'
```

Five idempotent steps — re-running after an interruption is safe and mostly a
no-op:

1. **`setup_env.sh`** — pip installs, and reconciles torch with the host driver.
   If the preinstalled torch can't see the GPU, it detects the driver's CUDA
   ceiling from `nvidia-smi` and reinstalls the matching `cu12x` wheel.
2. **`setup_secrets.sh`** — prompts for `WANDB_API_KEY` and `HF_TOKEN` (see
   below).
3. **`download_models.py`** — the tier's checkpoints into `/workspace`, then
   verifies a weight file actually landed.
4. **`download_data.py`** — CLINC150 and 20 Newsgroups, plus validation of the
   domain hierarchy.
5. **`preflight.py`** — the gate before you spend anything.

### API keys

Step 2 asks for each key, using anything already exported or saved from a
previous run on the same volume, and prompting only for what's missing. Both are
optional — blank skips them, and a box with no terminal skips silently rather
than hanging, so an automated bootstrap still works.

Keys go to `/workspace/.wamal_env` with mode 600. That's outside the repo, so it
can't be committed, and on the mounted volume, so it survives a container
restart. `wandb login` additionally writes `~/.netrc`, which is what lets an
unattended `nohup ... --wandb` in a fresh shell authenticate without sourcing
anything.

To avoid the prompt entirely, either export the keys first, or pass
`--skip-secrets`:

```bash
WANDB_API_KEY=xxx bash llm/bootstrap/bootstrap.sh --tier phase1
bash llm/bootstrap/bootstrap.sh --tier phase1 --skip-secrets
```

If you keep the key in `deploy/env` on your laptop, `deploy/remote.sh` forwards
it automatically and the prompt won't appear.

Tiers: `smoke`, `phase0`, `phase1`, `phase2`, `phase3`, `all`. Pull only what
that box will run.

### Preflight is the important step

It runs a real two-epoch WAMAL job at k=5 on OPT-125M — the dual-adapter forward,
the second-order inner step, the writer — and checks memory is ~1× the base model
rather than 2×. It takes seconds on a GPU. **If it fails, fix that before
launching a sweep**; every failure mode it catches is one that would otherwise
surface partway through a paid run.

---

## 4. Price the run before committing

```bash
deploy/remote.sh 'python -m llm.scripts.sweep --phase phase1 --dry-run'
```

Prints the cell list and count. Once any runs exist it also estimates GPU-hours
from their measured wall times, so after `--phase smoke` the phase-1 number is an
extrapolation rather than a guess, and after phase 1 the phase-2 number is too.

Multiply by the instance's hourly rate. If it's more than you want to spend,
reduce seeds before reducing arms — the arms are the experiment, the seeds are
the error bars.

---

## 5. Launch

Headless, so it survives your laptop closing:

```bash
# single GPU
deploy/remote.sh 'cd /workspace/rl-aux && \
  nohup python -m llm.scripts.sweep --phase phase1 --keep-going \
  > /workspace/sweep.log 2>&1 & echo started'

# multiple GPUs: one cell per GPU, pulled from a shared queue
deploy/remote.sh 'cd /workspace/rl-aux && \
  nohup python -m llm.scripts.sweep --phase phase2 --gpus 0,1,2,3 --keep-going \
  > /workspace/sweep.log 2>&1 & echo started'
```

Add `--wandb` to either command for a live dashboard. Runs are grouped by phase
and split by arm, so a 400-cell sweep arrives as one comparable experiment rather
than 400 loose runs:

```bash
python -m llm.scripts.sweep --phase phase2 --gpus 0,1,2,3 --keep-going --wandb
```

Tracking is a mirror, never a dependency. `metrics.jsonl` is written first and is
what every analysis reads, so if wandb is missing, unauthenticated, or drops
mid-sweep, the run continues and the experiment is still complete. Without a key
it records offline into each run directory; `deploy/sync.sh pull` brings those
home and `wandb sync vast_artifacts/**/wandb/offline-run-*` uploads them later.

`--keep-going` logs a failed cell and continues instead of aborting the sweep.
Use it for overnight runs; omit it when you want the first failure to stop
everything so you can look at it.

In multi-GPU mode each cell's stdout goes to `sweep.log` inside its own run
directory — four interleaved training logs on one terminal is unreadable. The
top-level log gets one status line per completed cell.

Useful flags: `--limit N` (stop after N cells, for a timeboxed rental),
`--force` (rerun completed cells), `--primary-lr` / `--optimizer` (phase 0
calibration).

---

## 6. Monitor

```bash
deploy/sync.sh status                      # cells done, disk, GPU utilisation
deploy/remote.sh 'tail -f /workspace/sweep.log'
deploy/remote.sh --logs                    # follow the newest individual run
deploy/remote.sh --gpu                     # nvidia-smi snapshot
```

You can pull partial results at any time; completed cells are self-contained.

### If the instance dies

Nothing is lost that was already written. Rent a new box, reattach the volume if
you can, and re-run the **identical** sweep command. Completion is a `DONE` file
per cell, so finished work is skipped and the sweep picks up where it stopped.

If the volume is gone, you lose the runs but not the setup: bootstrap and
re-launch.

---

## 7. Bring results home

```bash
deploy/sync.sh pull                        # -> ./vast_artifacts
deploy/sync.sh pull --with-checkpoints     # include best_model.pt
```

rsync, so it's incremental and resumable — a dropped connection costs the current
file, not the transfer, and re-running after more cells finish moves only the new
ones. Checkpoints are excluded by default because `metrics.jsonl`, `summary.json`
and the `.npz` dumps are everything the analysis reads and are three orders of
magnitude smaller.

Then, locally:

```bash
RESULTS_DIR=./vast_artifacts python -m llm.analysis.aggregate --figure
RESULTS_DIR=./vast_artifacts python -m llm.analysis.sublabels
RESULTS_DIR=./vast_artifacts python -m llm.analysis.weights --plots --examples 10
```

---

## 8. Teardown

**Pull first. Verify second. Destroy third.**

```bash
deploy/sync.sh pull
find vast_artifacts -name DONE | wc -l      # matches the expected cell count?
```

Then destroy the instance from the vast.ai console. If you want to keep the
model cache for next time, detach the volume rather than deleting it — it saves
the download, not the compute.

---

## Troubleshooting

Every entry below is a failure that actually happened while building this, not a
hypothetical.

### `RuntimeError: derivative for aten::_scaled_dot_product_flash_attention... is not implemented`

The meta step differentiates *through* the inner SGD update, so the backward pass
must itself be differentiable, and the fused attention kernels are not — on CUDA
as well as CPU.

Already handled: backbones load with `attn_implementation="eager"`, and
`ExperimentConfig.resolve()` refuses to build a `maxl`/`wamal` arm with anything
else. If you see this, something overrode `attn_implementation`. Note the
standing cost: eager attention is slower and holds more activation memory, which
is why the meta-step arms are meaningfully more expensive than `single`.

### Model "downloaded" but training can't find weights

`snapshot_download` filtered to safetensors returns success with only a config
and tokenizer for repos that ship just `.bin` — `facebook/opt-125m` is one.

Already handled: `download_models.py` verifies a weight file landed and retries
unfiltered, and preflight checks it independently. To confirm by hand:

```bash
deploy/remote.sh 'du -sh /workspace/.cache/huggingface/hub/models--*'
```

A model directory in the low hundreds of KB is config-only.

### `NVIDIA driver is too old` / `torch.cuda.is_available()` is False

The image's torch was built against a newer CUDA than the host driver supports.
Vast hosts vary a lot here.

`setup_env.sh` detects and repairs this. To force a specific build:

```bash
deploy/remote.sh 'cd /workspace/rl-aux && FORCE_CUDA=cu121 bash llm/bootstrap/setup_env.sh'
```

### `BrokenPipeError` when dataloader workers start

The container ran out of pids. Uncapped OpenBLAS/OpenMP spawn one thread per
*host* core — on a 128-core machine where you were sold 8, that's dozens of
threads per process times every worker.

`env.configure()` caps the BLAS thread variables at 4. Lower it, and the
dataloader workers, if it recurs:

```bash
deploy/remote.sh 'cd /workspace/rl-aux && WAMAL_NUM_THREADS=2 \
  python -m llm.scripts.run_experiment --arm wamal --shots 5 --num-workers 0'
```

The caps only take effect if set before torch and numpy import, which is why the
entry point calls `env.configure()` before importing anything else.

### CUDA OOM

In rough order of what to try: lower `--batch-size`; drop to a smaller model;
`--dtype fp32` only if bf16 is unsupported (fp32 uses *more* memory, it is not an
OOM fix); `--first-order` to skip the second-order term — but that changes the
method, so it is a diagnostic, not a run you would report.

A single bad cell will not kill the sweep: each runs in its own process, and
`--keep-going` moves on.

### Results directory is empty after a sweep

Check `RESULTS_DIR` agrees between the training and the pull. `deploy/remote.sh`
exports it; a manual `ssh` session does not.

```bash
deploy/remote.sh 'python -c "from llm import env; print(env.describe())"'
```

### Host key verification failed

Vast recycles host keys across instances, so a `known_hosts` entry from a
previous rental fails on a new box at the same address. `deploy/` passes
`StrictHostKeyChecking=no` for exactly this reason. If you hit it via plain
`ssh`, remove the stale entry:

```bash
ssh-keygen -R '[ssh5.vast.ai]:41234'
```

### wandb runs are offline when you wanted them online

`auto` mode goes offline whenever it can't find a key, rather than blocking on a
handshake that will fail — five idle minutes per cell across 400 cells is real
money. Check the key landed:

```bash
deploy/remote.sh 'echo ${WANDB_API_KEY:+key present}; grep -c api.wandb.ai ~/.netrc'
```

Re-run `bash llm/bootstrap/setup_secrets.sh` to fix it, or force the intended
behaviour with `--wandb-mode online` so a missing key fails loudly instead of
quietly recording to disk.

Nothing is lost either way: upload after the fact with

```bash
wandb sync vast_artifacts/**/wandb/offline-run-*
```

### Sweep looks stuck

Multi-GPU mode prints one line per *completed* cell, so a long cell looks like
silence. Check the individual run log:

```bash
deploy/remote.sh --logs
deploy/remote.sh --gpu       # utilisation > 0 means it is working
```

---

## Cost discipline

- `--dry-run` first, every time.
- Phases 0 and 1 belong on the cheapest card that fits. OPT-125M at 64 tokens is
  not compute-bound.
- One long instance beats several short ones; setup is per-instance overhead.
- `summary.json` records wall time per run, so each phase prices the next.
- Phase 1 has an explicit **go/no-go**: if `wamal` doesn't beat `human` at k=5 and
  k=10, stop and revisit the framing before paying for phase 2. That gate exists
  to be used.
