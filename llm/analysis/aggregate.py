"""Collect finished runs into tables and the headline figure.

    python -m llm.analysis.aggregate --figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARM_ORDER = ["single", "human", "maxl", "wamal"]
ARM_LABELS = {
    "single": "Single-task",
    "human": "Human hierarchy",
    "maxl": "MAXL (learned labels)",
    "wamal": "WAMAL (labels + weights)",
}


def load_runs(results_dir: Path) -> "list[dict]":
    """Every completed run. Incomplete cells are skipped, not partially read."""
    rows = []
    for marker in sorted(results_dir.rglob("DONE")):
        summary_path = marker.parent / "summary.json"
        if not summary_path.exists():
            continue
        row = json.loads(summary_path.read_text())
        config_path = marker.parent / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())
            row["optimizer"] = config.get("optimizer")
            row["primary_lr"] = config.get("primary_lr")
            row["total_epoch"] = config.get("total_epoch")
        row["variant"] = variant_of(config)
        row["path"] = str(marker.parent)
        rows.append(row)
    return rows


def variant_of(config: dict) -> str:
    """Which method a run used, for keeping incomparable runs out of one average.

    Derived from the config rather than read from the summary so that runs
    predating the field still resolve -- absent `meta_holdout_frac` means the
    paper-faithful path, which is what those runs did.

    This has to be a grouping key everywhere. `summarise` would otherwise average
    two different methods into one row, and `paired_deltas` is worse: it pivots on
    `arm`, and `pivot_table` silently takes the mean of duplicate keys, so a
    holdout run would be folded into the base result with no warning at all.
    """
    parts = []
    frac = config.get("meta_holdout_frac") or 0.0
    if frac > 0:
        parts.append(f"holdout{frac:g}")
    corruption = config.get("label_corruption") or 0.0
    if corruption > 0:
        parts.append(f"corrupt{corruption:g}")
    return "_".join(parts) if parts else "base"


def to_frame(rows: list[dict]):
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["arm"] = pd.Categorical(frame["arm"], categories=ARM_ORDER, ordered=True)
    return frame


def summarise(frame):
    """Mean/std over seeds, which is the unit of variation the paper omits."""
    grouped = (
        frame.groupby(["dataset", "model", "variant", "shots", "arm"], observed=True)
        .agg(
            acc_mean=("test_acc_at_best_val", "mean"),
            acc_std=("test_acc_at_best_val", "std"),
            aux_recovery=("final_test_aux_vs_fine_acc", "mean"),
            seeds=("seed", "nunique"),
            wall_s=("wall_time_s", "mean"),
        )
        .reset_index()
    )
    grouped["acc_std"] = grouped["acc_std"].fillna(0.0)
    return grouped


def paired_deltas(frame, baseline: str = "human", treatment: str = "wamal"):
    """Per-seed differences between two arms.

    Paired on `(dataset, model, shots, seed)` because the few-shot sampler makes
    every arm within a cell see identical training data. Comparing group means
    would throw away that pairing and widen the error bars for no reason.
    """
    import pandas as pd

    # `variant` in the index, not just the groupby: arms are only comparable
    # within one method, and leaving it out would let pivot_table average a
    # holdout run together with its base counterpart.
    keys = ["dataset", "model", "variant", "shots", "seed"]
    wide = frame.pivot_table(
        index=keys, columns="arm", values="test_acc_at_best_val", observed=True
    )
    if baseline not in wide or treatment not in wide:
        return pd.DataFrame()

    paired = wide[[baseline, treatment]].dropna().reset_index()
    paired["delta"] = paired[treatment] - paired[baseline]
    out = (
        paired.groupby(["dataset", "model", "variant", "shots"])
        .agg(delta_mean=("delta", "mean"),
             delta_std=("delta", "std"),
             n_pairs=("delta", "size"),
             wins=("delta", lambda d: int((d > 0).sum())))
        .reset_index()
    )
    out["delta_std"] = out["delta_std"].fillna(0.0)
    return out


def headline_figure(summary, out_path: Path) -> None:
    """Accuracy vs shots, log x, one panel per (dataset, model), band = 1 std."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = summary.groupby(["dataset", "model"], observed=True)
    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, ((dataset, model), block) in zip(axes[0], panels):
        for arm in ARM_ORDER:
            arm_block = block[block["arm"] == arm].sort_values("shots")
            if arm_block.empty:
                continue
            ax.plot(arm_block["shots"], arm_block["acc_mean"], marker="o",
                    label=ARM_LABELS[arm])
            ax.fill_between(
                arm_block["shots"],
                arm_block["acc_mean"] - arm_block["acc_std"],
                arm_block["acc_mean"] + arm_block["acc_std"],
                alpha=0.15,
            )
        ax.set_xscale("log")
        ax.set_xlabel("examples per fine class (k)")
        ax.set_ylabel("test accuracy at best val epoch")
        ax.set_title(f"{dataset} / {model}")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[aggregate] wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    from llm import env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out", default=None, help="directory for tables/figures")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)

    results = Path(args.results_dir) if args.results_dir else env.results_dir()
    out = Path(args.out) if args.out else results / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows = load_runs(results)
    if not rows:
        print(f"[aggregate] no completed runs under {results}")
        return 1

    frame = to_frame(rows)
    print(f"[aggregate] {len(frame)} completed runs")

    summary = summarise(frame)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))

    for baseline in ("single", "human", "maxl"):
        deltas = paired_deltas(frame, baseline=baseline, treatment="wamal")
        if deltas.empty:
            continue
        deltas.to_csv(out / f"delta_wamal_vs_{baseline}.csv", index=False)
        print(f"\nwamal - {baseline} (paired by seed):")
        print(deltas.to_string(index=False))

    if args.figure:
        headline_figure(summary, out / "headline.png")

    print(f"\n[aggregate] tables in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
