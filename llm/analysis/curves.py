"""Per-epoch training curves from `metrics.jsonl`.

`aggregate.py` answers "which arm won", from one row per run. This answers "did
that run train sanely", from every epoch of it -- the question you need before
believing any headline number.

The diagnostics that matter here, in rough order of how likely they are to bite:

- **Overfitting.** k=5 is 40 epochs over ~750 examples. Train/val divergence is
  the expected behaviour, not an edge case, and the epoch where val peaks is the
  only one whose test number means anything.
- **Learning rate.** Phase 0 exists to calibrate it, which is not possible from
  final accuracy alone.
- **Whether the meta step does anything.** `meta_loss_after` below
  `meta_loss_before` is the direct evidence that differentiating through the
  inner update improves the primary loss. If they sit on top of each other, the
  bilevel machinery is running and costing double-backward time for nothing.
- **Weight collapse.** If `train_weight_std` goes to zero, every sample is
  getting the same auxiliary weight and WAMAL has degenerated into MAXL with a
  rescaled auxiliary loss, while still reporting plausible accuracy. Dispersion
  between samples is the thing to watch, not the mean: a constant weight that
  drifts across epochs moves the mean and is still MAXL. Nothing in the summary
  would reveal either case.

    python -m llm.analysis.curves --list
    python -m llm.analysis.curves                    # every cell, arms overlaid
    python -m llm.analysis.curves --cell clinc150/opt-125m/k5/seed0
    python -m llm.analysis.curves --check            # text-only, no matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm import env

ARM_ORDER = ["single", "human", "maxl", "wamal"]
ARM_COLORS = {
    "single": "#888888",
    "human": "#1f77b4",
    "maxl": "#ff7f0e",
    "wamal": "#d62728",
}


def load_curve(run_path: Path) -> dict | None:
    """One run's per-epoch records plus its config.

    Reads runs that are still in progress: `metrics.jsonl` is flushed per epoch
    precisely so a live or preempted run is still readable, and refusing to plot
    those would defeat the point.
    """
    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    epochs = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            epochs.append(json.loads(line))
        except ValueError:
            # A torn final line means we caught the run mid-write. Everything
            # before it is still good.
            continue
    if not epochs:
        return None

    config = {}
    config_path = run_path / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except ValueError:
            pass

    return {
        "path": run_path,
        "epochs": epochs,
        "arm": config.get("arm", run_path.parent.parent.name),
        "dataset": config.get("dataset", "?"),
        "model": config.get("model", "?"),
        "shots": config.get("shots"),
        "seed": config.get("seed"),
        "complete": (run_path / "DONE").exists(),
    }


def cell_key(run: dict) -> str:
    """Everything except the arm, so arms of one cell overlay on one figure."""
    return f"{run['dataset']}/{run['model']}/k{run['shots']}/seed{run['seed']}"


def discover(results_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for metrics_path in sorted(results_dir.rglob("metrics.jsonl")):
        run = load_curve(metrics_path.parent)
        if run is not None:
            groups.setdefault(cell_key(run), []).append(run)
    for runs in groups.values():
        runs.sort(key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99)
    return groups


# --------------------------------------------------------------------------- #
# text diagnostics
# --------------------------------------------------------------------------- #


def series(epochs: list[dict], key: str) -> list:
    return [e.get(key) for e in epochs if e.get(key) is not None]


def diagnose(run: dict) -> list[str]:
    """Warnings worth acting on. Silence means nothing obviously went wrong."""
    notes = []
    epochs = run["epochs"]
    val_acc = series(epochs, "val_acc")
    train_acc = series(epochs, "train_acc_primary")

    if not val_acc:
        return ["no val_acc recorded"]

    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    n = len(val_acc)

    # Val peaking early and then decaying is the signature of overfitting; the
    # later epochs are just memorising the few-shot subsample.
    if n >= 4 and best_idx < n * 0.5 and val_acc[-1] < val_acc[best_idx] - 0.02:
        notes.append(
            f"overfitting: val peaked {val_acc[best_idx]:.3f} at epoch "
            f"{best_idx + 1}/{n}, ended {val_acc[-1]:.3f}"
        )

    # The opposite problem: still climbing at the end means the epoch budget,
    # not the method, is the binding constraint.
    if n >= 4 and best_idx == n - 1 and val_acc[-1] > val_acc[max(0, n - 4)] + 0.01:
        notes.append(f"undertrained: val still rising at the last epoch ({val_acc[-1]:.3f})")

    if train_acc and train_acc[-1] > 0.99 and val_acc[-1] < 0.8:
        notes.append(
            f"train saturated at {train_acc[-1]:.3f} with val {val_acc[-1]:.3f}"
        )

    losses = series(epochs, "train_loss_primary")
    if losses:
        if any(x != x for x in losses):  # NaN
            notes.append("NaN in train loss")
        elif len(losses) >= 3 and losses[-1] > losses[0]:
            notes.append(f"train loss rose: {losses[0]:.4f} -> {losses[-1]:.4f} (lr too high?)")

    # Is the meta step actually buying anything?
    before, after = series(epochs, "meta_loss_before"), series(epochs, "meta_loss_after")
    if before and after:
        pairs = list(zip(before, after))
        improved = sum(1 for b, a in pairs if a < b)
        gain = sum(b - a for b, a in pairs) / len(pairs)
        if improved < len(pairs) * 0.5:
            notes.append(
                f"meta step is not helping: primary loss improved in only "
                f"{improved}/{len(pairs)} epochs (mean {gain:+.5f})"
            )

    # Weight collapse turns WAMAL into MAXL invisibly. The question is whether
    # weights differ *between samples*, which is dispersion within an epoch, not
    # movement of the mean across epochs -- a per-sample-constant weight that
    # drifts over training would pass the latter check while being pure MAXL.
    # Judged relative to the mean because weight magnitude depends on the
    # scaling range, and only over a long enough run: the first epochs are
    # legitimately flat while the label network is still near its init.
    if run["arm"] == "wamal":
        stds = series(epochs, "train_weight_std")
        means = series(epochs, "train_weight_mean")
        if len(stds) >= 8 and means:
            centre = sum(means) / len(means)
            late = stds[len(stds) // 2:]           # after the warm-up half
            dispersion = sum(late) / len(late)
            if centre and dispersion / abs(centre) < 0.01:
                notes.append(
                    f"per-sample weights are constant: std {dispersion:.2e} at "
                    f"mean {centre:.4f} ({dispersion / abs(centre):.2%}) -- this "
                    f"is MAXL with a rescaled auxiliary loss, not WAMAL"
                )
        elif not stds and len(means) >= 8:
            notes.append(
                "train_weight_std missing: cannot tell WAMAL from MAXL "
                "(run predates the dispersion logging)"
            )
    return notes


def report(groups: dict[str, list[dict]]) -> int:
    flagged = 0
    for key in sorted(groups):
        print(f"\n{key}")
        for run in groups[key]:
            epochs = run["epochs"]
            val_acc = series(epochs, "val_acc")
            best = max(val_acc) if val_acc else float("nan")
            state = "" if run["complete"] else "  [running]"
            print(f"  {run['arm']:<8} {len(epochs):>3} epochs  best val {best:.4f}{state}")
            for note in diagnose(run):
                flagged += 1
                print(f"           ! {note}")
    return flagged


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def plot_cell(key: str, runs: list[dict], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_meta = any(series(r["epochs"], "meta_loss_before") for r in runs)
    has_weights = any(series(r["epochs"], "train_weight_mean") for r in runs)
    panels = 2 + int(has_meta) + int(has_weights)

    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4), squeeze=False)
    axes = axes[0]

    for run in runs:
        epochs = run["epochs"]
        xs = [e["epoch"] for e in epochs]
        color = ARM_COLORS.get(run["arm"], None)

        # Train solid, val dashed: the gap between them is the thing to look at.
        axes[0].plot(xs, [e.get("train_loss_primary") for e in epochs],
                     color=color, label=f"{run['arm']} train")
        axes[0].plot(xs, [e.get("val_loss") for e in epochs],
                     color=color, linestyle="--", alpha=0.7, label=f"{run['arm']} val")

        val_acc = [e.get("val_acc") for e in epochs]
        axes[1].plot(xs, val_acc, color=color, label=run["arm"])
        # Mark the selected epoch: its test number is the one that gets reported.
        clean = [(x, v) for x, v in zip(xs, val_acc) if v is not None]
        if clean:
            bx, bv = max(clean, key=lambda p: p[1])
            axes[1].scatter([bx], [bv], color=color, zorder=5, s=30)

    axes[0].set_title("primary loss (solid train / dashed val)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=7)

    axes[1].set_title("validation accuracy (dot = selected epoch)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val acc")
    axes[1].legend(fontsize=8)

    idx = 2
    if has_meta:
        for run in runs:
            epochs = run["epochs"]
            before = [e.get("meta_loss_before") for e in epochs]
            if not any(b is not None for b in before):
                continue
            xs = [e["epoch"] for e in epochs]
            color = ARM_COLORS.get(run["arm"], None)
            axes[idx].plot(xs, before, color=color, linestyle="--", alpha=0.7,
                           label=f"{run['arm']} before")
            axes[idx].plot(xs, [e.get("meta_loss_after") for e in epochs],
                           color=color, label=f"{run['arm']} after")
        axes[idx].set_title("meta step: primary loss before/after\n(after < before = the step helps)")
        axes[idx].set_xlabel("epoch"); axes[idx].legend(fontsize=7)
        idx += 1

    if has_weights:
        for run in runs:
            epochs = run["epochs"]
            weights = [e.get("train_weight_mean") for e in epochs]
            if not any(w is not None for w in weights):
                continue
            xs = [e["epoch"] for e in epochs]
            color = ARM_COLORS.get(run["arm"], None)
            axes[idx].plot(xs, weights, color=color, label=run["arm"])
            # The +/-1 std band is the panel's actual point: a line with no band
            # is a constant weight, i.e. MAXL. Drawn only where std exists so
            # runs predating the dispersion logging still plot their mean.
            stds = [e.get("train_weight_std") for e in epochs]
            if any(s is not None for s in stds):
                lo = [w - s for w, s in zip(weights, stds) if w is not None and s is not None]
                hi = [w + s for w, s in zip(weights, stds) if w is not None and s is not None]
                axes[idx].fill_between(xs[:len(lo)], lo, hi, color=color, alpha=0.2)
        axes[idx].set_title("auxiliary weight, mean +/- std across samples\n"
                            "(no band = collapsed to MAXL)")
        axes[idx].set_xlabel("epoch"); axes[idx].legend(fontsize=8)

    fig.suptitle(key)
    fig.tight_layout()
    out_path = out_dir / f"curves_{key.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--cell", default=None,
                        help="one cell key, e.g. clinc150/opt-125m/k5/seed0")
    parser.add_argument("--list", action="store_true", help="list cell keys and exit")
    parser.add_argument("--check", action="store_true",
                        help="text diagnostics only; no matplotlib needed")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir) if args.results_dir else env.results_dir()
    if not results_dir.exists():
        print(f"no results at {results_dir}")
        return 1

    groups = discover(results_dir)
    if not groups:
        print(f"no metrics.jsonl found under {results_dir}")
        return 1

    if args.list:
        for key in sorted(groups):
            arms = ",".join(r["arm"] for r in groups[key])
            print(f"{key}  [{arms}]")
        return 0

    if args.cell:
        if args.cell not in groups:
            print(f"unknown cell {args.cell!r}; --list to see the {len(groups)} available")
            return 1
        groups = {args.cell: groups[args.cell]}

    flagged = report(groups)

    if not args.check:
        out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        for key, runs in sorted(groups.items()):
            print(f"wrote {plot_cell(key, runs, out_dir)}")

    print(f"\n{len(groups)} cell(s), {flagged} diagnostic warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
