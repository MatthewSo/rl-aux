# WAMAL-LM: what the runs actually showed

Snapshot taken 29 Aug 2026, when work on the rented box was paused. Everything
below is computed from `vast_artifacts/`, which is the synced copy of
`/workspace/results` — no numbers here are from memory.

Reproduce any table with:

```bash
export RESULTS_DIR=./vast_artifacts
python -m llm.analysis.aggregate                       # accuracy and paired deltas
python -m llm.analysis.recovery --arm wamal --psi 15   # chance-corrected recovery
```

---

## 1. Headline

**On CLINC150 domain classification with OPT-125M, WAMAL never beat the
single-task baseline, and the reason is not the one we expected.**

Three candidate explanations were tested. Two are now refuted by direct
measurement, and the third is the one that survives:

| Hypothesis | Test | Verdict |
| --- | --- | --- |
| The meta-gradient saturates, so φ stops learning | `phase1_holdout` | **Refuted.** Saturation was fixed completely and the deficit stayed |
| φ never learns real structure | permutation-invariant recovery | **Refuted.** φ recovers real structure, up to 6.7x chance |
| The task has almost no room for auxiliary labels | oracle (`human`) arm | **Supported.** Perfect labels are worth only +0.3 to +1.0pp |

The uncomfortable part is that the second and third findings do not line up.
φ recovers *most* structure exactly where auxiliary labels matter *least*.

---

## 2. Phase 1 — the main grid (complete)

200 cells, all 4 arms x k ∈ {5,10,25,50,100} x 10 seeds, `clinc150` (10 domains
primary, 150 human intents auxiliary), OPT-125M. 0 failures, 159.2 min wall-clock
on 4x L40S at `--pack 4`.

Test accuracy at best validation epoch, mean over 10 seeds:

| k | single | human | maxl | wamal |
| --- | --- | --- | --- | --- |
| 5 | 0.9049 | 0.9121 | 0.8955 | 0.8946 |
| 10 | 0.9301 | 0.9402 | 0.9233 | 0.9252 |
| 25 | 0.9545 | 0.9605 | 0.9520 | 0.9523 |
| 50 | 0.9651 | 0.9682 | 0.9650 | 0.9663 |
| 100 | 0.9720 | 0.9751 | 0.9724 | 0.9719 |

Paired by seed, so every arm in a row saw a byte-identical few-shot subset:

| k | wamal − single | wins | human − single | wamal − maxl |
| --- | --- | --- | --- | --- |
| 5 | **−1.04pp** | 0/10 | +0.72pp | −0.09pp |
| 10 | **−0.49pp** | 2/10 | +1.00pp | +0.20pp |
| 25 | −0.22pp | 2/10 | +0.60pp | +0.03pp |
| 50 | +0.11pp | 7/10 | +0.31pp | +0.13pp |
| 100 | −0.01pp | 6/10 | +0.31pp | −0.04pp |

Three things to read off this table.

**The oracle ceiling is ~1pp.** `human − single` is the value of a perfect,
hand-authored hierarchy. It peaks at +1.00pp. Any method that generates auxiliary
labels is competing for at most that much, and WAMAL starts 1.04pp *below*
single at k=5 — so it must close 1.8pp on a task where the best possible
auxiliary signal is worth 1.0pp. **This is the binding constraint, and it is a
property of the task, not of the method.**

**WAMAL ≈ MAXL everywhere.** The learned per-sample weighting — the entire
difference between the two arms — moves accuracy by between −0.09 and +0.20pp,
winning 5–7 times out of 10. That is indistinguishable from noise. Whatever else
is happening, the weighting contributes nothing here.

**The deficit shrinks monotonically with data**, from −1.04pp at k=5 to −0.01pp
at k=100. Auxiliary learning hurts most in the few-shot regime it was supposed to
help.

---

## 3. Phase 1 holdout — the saturation hypothesis, refuted

The suspected mechanism was: the inner loss reaches ~0 early, the meta-gradient
vanishes with it, and φ stops receiving signal for most of training.
`phase1_holdout` withholds 20% of the training data (stratified, from *every* arm,
so the comparison stays fair) and computes the outer loss on the withheld split,
which cannot saturate by memorization.

**It worked as designed.** Fraction of epochs where the inner loss still carries
signal, measured as `meta_loss_before > 5% of its own maximum`:

| k | base | with 20% holdout |
| --- | --- | --- |
| 5 | 15% | **100%** |
| 10 | 25% | **100%** |
| 25 | 38% | — |
| 50 | 47% | — |
| 100 | 65% | — |

The meta objective now stays alive for the whole run instead of dying after 15% of
it. And the deficit did not move:

| k | wamal − single (base) | wamal − single (holdout) | pairs |
| --- | --- | --- | --- |
| 5 | −1.04pp | **−0.92pp** | 3 |
| 10 | −0.49pp | **−0.88pp** | 2 |

Nor did φ's labels improve — recovery lift went *down* slightly (§4). Note the
base-vs-holdout comparison is across different training-set sizes, so the levels
are not strictly comparable; what is comparable is that removing the saturation
did not reverse the sign, and did not help at all.

This phase is **26/40 cells complete** (paused mid-run), so k=5 rests on 3 pairs
and k=10 on 2. Weak on its own — but it agrees with phase 1's much larger sample,
and the saturation metric it was built to move is unambiguous.

**Conclusion: the meta-gradient saturating was real, and was not the problem.**

---

## 4. Sub-label recovery — φ does learn, and it does not help

An earlier read of `aux_vs_fine_acc` suggested φ learned nothing (~6%, near the
1/15 = 6.7% chance rate). That metric is permutation-blind: clustering is only
identified up to a relabeling within each coarse block, so a *perfect* recovery
under a permuted labeling still scores at chance. `llm/analysis/recovery.py`
fixes this with Hungarian matching and mutual information against a
shuffled-within-block null.

WAMAL, mean over 10 seeds, final epoch:

| k | matched acc | null | lift | MI / null |
| --- | --- | --- | --- | --- |
| 5 | 0.337 | 0.283 | +0.054 | 1.14x |
| 10 | 0.284 | 0.213 | +0.071 | 1.35x |
| 25 | 0.248 | 0.155 | +0.092 | 2.16x |
| 50 | 0.230 | 0.128 | +0.102 | 3.69x |
| 100 | 0.221 | 0.108 | +0.113 | **6.72x** |
| 5 (holdout) | 0.305 | 0.268 | +0.037 | 1.13x |
| 10 (holdout) | 0.260 | 0.206 | +0.054 | 1.32x |

φ genuinely discovers the human intent hierarchy it was never shown — at k=100,
6.7x more mutual information than chance. So the label network works.

**But recovery and benefit are anti-correlated.** Recovery climbs 1.14x → 6.72x
as k grows, while the oracle's value falls +1.00pp → +0.31pp. At k=100, φ has
found real structure and it buys nothing; at k=5, where there is headroom, φ has
found almost nothing. **Good auxiliary labels do not convert into primary
accuracy on this task**, which is a stronger and more discouraging statement than
"the labels are bad."

---

## 5. The weighting is active but inert

WAMAL's per-sample auxiliary weights at k=5, over 400 epoch-records:

```
min 0.3045   max 0.5511   std 0.0403      (absolute extremes 0.2129 .. 0.7426)
```

Not collapsed — the weights vary across samples by ~10% of their own mean, and
that dispersion is stable at ~0.040 across every k. So the mechanism runs. It
simply does not change the outcome, which is exactly what `wamal − maxl` ≈ 0
says independently.

---

## 6. What this means for the next experiment

The ceiling is the thing to fix. A task where oracle auxiliary labels are worth
1pp cannot demonstrate a large WAMAL win no matter how good φ becomes, and §4
shows that even a good φ does not cash in.

Two candidate causes for the low ceiling, which are testable and not mutually
exclusive:

1. **The primary task is nearly solved without help.** 10-way domain
   classification at 90–97% leaves little for an auxiliary task to add.
   → `phase1_intent` (built, not yet run): all 150 intents as the primary task
   with 5 invented sub-classes each. A local smoke run reaches 19% after 2 epochs
   with the inner loss at 4.66 against `ln(150) = 5.01`, i.e. nowhere near
   saturating. Real headroom, and the auxiliary hierarchy is invented rather than
   recovered, which is MAXL's original setting.

2. **The representation is frozen, so auxiliary supervision has nothing to
   shape.** MAXL and WAMAL were demonstrated on VGG16/ResNet trained from
   scratch, where the auxiliary loss reshapes the shared trunk. Here 125.2M of
   129M parameters are frozen and only 1.89M of LoRA moves. Auxiliary learning
   works *through* representation shaping, and we have largely removed the
   mechanism. The ~1pp oracle ceiling is the symptom you would predict.
   → Not yet tested. Would need much higher LoRA rank, or unfrozen top blocks.

Hypothesis 2 is the more likely explanation and the cheaper one to falsify: if
raising trainable capacity raises the *oracle* ceiling, then the frozen backbone
was the binding constraint all along and every negative result above is a
statement about the setup rather than about WAMAL.

---

## 7. Artifact inventory

Synced to `vast_artifacts/` (54MB; checkpoints deliberately left on the box):

| Path | Contents |
| --- | --- |
| `clinc150/opt-125m/<arm>/k*/seed*/` | 200 completed phase-1 cells |
| `clinc150/opt-125m/<arm>__holdout0.2/k*/seed*/` | 26 completed holdout cells |
| `.../metrics.jsonl` | per-epoch losses, accuracy, meta losses, weight dispersion |
| `.../summary.json` | final/best accuracy, wall time, parameter counts |
| `.../dump_epoch*.npz` | φ's sub-label assignments (406 files) — input to `recovery.py` |
| `.../config.json` | full resolved config, so any cell can be re-run exactly |
| `_analysis/` | aggregate tables |
| `logs/sweep.log` | phase 1, 200 cells, 0 failures, 159.2 min |
| `logs/sweep_phase1_holdout.log` | holdout, 26/40 at pause |
| `logs/deploy_next.log` | chained deploy: git pull, test run, phase launches |

Also on wandb under project `wamal-lm`, grouped by phase.

### Paused state, and how to resume

The box is idle — all sweeps, cell subprocesses and the `deploy_next.sh` watcher
were stopped, and all four GPUs report 0% / 0MiB.

14 cells were in flight when work was paused and are partial. They have no `DONE`
marker, so a resumed sweep re-runs them from scratch; `metrics.jsonl` is
truncated at the start of each run, so no partial file is ever appended to. To
resume where we left off:

```bash
python -m llm.scripts.sweep --phase phase1_holdout --gpus 0,1,2,3 --pack 4 --keep-going --wandb
```

`phase1_ladder` (40 cells, the label-quality ladder) never started.
`phase1_intent` (45 cells) is implemented and smoke-tested but never launched.
