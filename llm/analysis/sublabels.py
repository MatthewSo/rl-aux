"""Did the label network rediscover the human hierarchy?

The strongest claim available here that the vision paper could not make: CLINC150
has real fine labels, so the generated sub-labels can be scored against them
directly instead of eyeballed. If WAMAL's sub-labels recover human intents better
than MAXL's, the weighting is doing something legible rather than just
regularising.

Scored within each primary class, because the hierarchy mask already guarantees
agreement across classes and including it would inflate every number.

    python -m llm.analysis.sublabels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def purity(generated: np.ndarray, truth: np.ndarray) -> float:
    """Fraction correct if every generated cluster is named after its plurality
    true label. Chance is 1/psi for balanced data."""
    if len(generated) == 0:
        return float("nan")
    total = 0
    for cluster in np.unique(generated):
        members = truth[generated == cluster]
        if len(members):
            total += np.bincount(members).max()
    return total / len(generated)


def score_dump(path: Path, psi: int) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    data = np.load(path)
    generated, fine, coarse = data["sublabels"], data["fine"], data["coarse"]

    valid = generated >= 0
    generated, fine, coarse = generated[valid], fine[valid], coarse[valid]
    if len(generated) == 0:
        return {"n": 0}

    per_class = []
    for cls in np.unique(coarse):
        sel = coarse == cls
        g = generated[sel] % psi          # offset within the class window
        t = fine[sel] % psi
        if len(np.unique(t)) < 2:
            continue
        per_class.append({
            "nmi": normalized_mutual_info_score(t, g),
            "ari": adjusted_rand_score(t, g),
            "purity": purity(g, t),
            "used": len(np.unique(g)),
        })

    if not per_class:
        return {"n": len(generated)}

    return {
        "n": int(len(generated)),
        "nmi": float(np.mean([p["nmi"] for p in per_class])),
        "ari": float(np.mean([p["ari"] for p in per_class])),
        "purity": float(np.mean([p["purity"] for p in per_class])),
        # Collapse detector. The entropy penalty exists to keep this near psi;
        # if it drifts toward 1 the auxiliary task has degenerated into a copy
        # of the primary one and the whole method is a no-op.
        "sublabels_used": float(np.mean([p["used"] for p in per_class])),
        "sublabels_available": psi,
    }


def main(argv: list[str] | None = None) -> int:
    from llm import env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    results = Path(args.results_dir) if args.results_dir else env.results_dir()
    out = Path(args.out) if args.out else results / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for config_path in sorted(results.rglob("config.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        if config["arm"] not in ("maxl", "wamal"):
            continue
        dumps = sorted(run.glob("dump_epoch*.npz"))
        if not dumps:
            continue
        scores = score_dump(dumps[-1], psi=config["psi"])
        if scores.get("n", 0) == 0:
            continue
        rows.append({
            "dataset": config["dataset"], "model": config["model"],
            "arm": config["arm"], "shots": config["shots"], "seed": config["seed"],
            "epoch_file": dumps[-1].name, **scores,
        })

    if not rows:
        print(f"[sublabels] no maxl/wamal dumps under {results}")
        return 1

    import pandas as pd

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["dataset", "model", "shots", "arm"])
        .agg(nmi=("nmi", "mean"), ari=("ari", "mean"), purity=("purity", "mean"),
             used=("sublabels_used", "mean"), seeds=("seed", "nunique"))
        .reset_index()
    )
    frame.to_csv(out / "sublabels_per_run.csv", index=False)
    grouped.to_csv(out / "sublabels.csv", index=False)
    print(grouped.to_string(index=False))
    print(f"\n[sublabels] wrote {out}/sublabels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
