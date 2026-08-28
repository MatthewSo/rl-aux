"""What did the weight head actually learn?

The paper's qualitative claim is that low-weight images are blurrier. Text lets
that be checked rather than asserted: the extreme-weight examples can be printed
verbatim, and the weight can be correlated against measurable properties
(length, per-class mean) instead of eyeballed.

The distribution is also the sanity check on `val_range`. Weights piled against
0 or 1 mean the head is saturated and r=5 is the wrong range.

    python -m llm.analysis.weights --examples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_last_dump(run: Path):
    dumps = sorted(run.glob("dump_epoch*.npz"))
    if not dumps:
        return None
    data = np.load(dumps[-1])
    if "weights" not in data:
        return None
    weights = data["weights"]
    if np.all(np.isnan(weights)):
        return None
    return data, dumps[-1].name


def describe(weights: np.ndarray) -> dict:
    valid = weights[~np.isnan(weights)]
    if len(valid) == 0:
        return {}
    return {
        "n": int(len(valid)),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "p05": float(np.percentile(valid, 5)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(valid.max()),
        # Fraction pinned to the ends of the sigmoid. High values mean the
        # scaling range is mis-set, not that the model is confident.
        "saturated_low": float((valid < 0.02).mean()),
        "saturated_high": float((valid > 0.98).mean()),
    }


def plot_distribution(weights: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = weights[~np.isnan(weights)]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(valid, bins=50, range=(0, 1))
    ax.set_xlabel("raw sigmoid weight (before $2^{2rw-r}$ scaling)")
    ax.set_ylabel("training examples")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def extreme_examples(config: dict, data, count: int) -> dict:
    """Highest and lowest weighted training texts.

    Rebuilds the exact training subset from the config: the sampler is a pure
    function of (seed, dataset, k), so the indices in the dump line up.
    """
    from llm import env
    from llm.datasets import build_dataset
    from llm.datasets.fewshot import FewShotSubset

    env.configure()
    base = build_dataset(config["dataset"], "train", config["seed"])
    subset = FewShotSubset(base, k=config["shots"], seed=config["seed"],
                           dataset_name=config["dataset"])

    weights = data["weights"]
    valid = np.flatnonzero(~np.isnan(weights))
    order = valid[np.argsort(weights[valid])]

    def render(indices):
        out = []
        for i in indices:
            text, coarse, fine = subset[int(i)]
            out.append({"weight": float(weights[i]), "coarse": int(coarse),
                        "fine": int(fine), "text": text[:200]})
        return out

    return {"lowest": render(order[:count]), "highest": render(order[-count:][::-1])}


def main(argv: list[str] | None = None) -> int:
    from llm import env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--examples", type=int, default=0,
                        help="print N extreme-weight texts per run (0 = off)")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)

    results = Path(args.results_dir) if args.results_dir else env.results_dir()
    out = Path(args.out) if args.out else results / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows, examples = [], {}
    for config_path in sorted(results.rglob("config.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        if config["arm"] != "wamal":
            continue
        loaded = load_last_dump(run)
        if loaded is None:
            continue
        data, dump_name = loaded

        stats = describe(data["weights"])
        if not stats:
            continue
        rows.append({
            "dataset": config["dataset"], "model": config["model"],
            "shots": config["shots"], "seed": config["seed"],
            "epoch_file": dump_name, **stats,
        })

        slug = f"{config['dataset']}_{config['model']}_k{config['shots']}_s{config['seed']}"
        if args.plots:
            plot_distribution(data["weights"], slug, out / f"weights_{slug}.png")
        if args.examples:
            examples[slug] = extreme_examples(config, data, args.examples)

    if not rows:
        print(f"[weights] no wamal dumps with weights under {results}")
        return 1

    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "weights_per_run.csv", index=False)
    print(frame.to_string(index=False))

    saturated = frame["saturated_low"] + frame["saturated_high"]
    if (saturated > 0.5).any():
        print("\n[weights] WARNING: >50% of weights are pinned to the sigmoid "
              "ends in at least one run. val_range is probably mis-set.")

    if examples:
        (out / "weight_examples.json").write_text(json.dumps(examples, indent=2))
        for slug, block in list(examples.items())[:1]:
            print(f"\n--- {slug} lowest-weighted ---")
            for item in block["lowest"]:
                print(f"  {item['weight']:.3f}  {item['text']!r}")
            print(f"--- {slug} highest-weighted ---")
            for item in block["highest"]:
                print(f"  {item['weight']:.3f}  {item['text']!r}")

    print(f"\n[weights] wrote {out}/weights_per_run.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
