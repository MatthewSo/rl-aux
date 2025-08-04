from collections import defaultdict

import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Iterable, Optional, Dict


def _apply_labels_title(xlabel: Optional[str], ylabel: Optional[str], title: Optional[str]) -> None:
    """Helper to optionally set xlabel, ylabel, and title (leave blank if None)."""
    plt.title(title if title is not None else "")
    plt.xlabel(xlabel if xlabel is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")

def plot_epoch_metrics_bar(
        runs: Iterable[Tuple[Sequence, str]],
        *,
        j: int = 15,
        train: bool = False,
        aux: bool = False,
        include_loss: bool = False,
        figsize: Tuple[float, float] = (7, 4),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        y_window: Optional[Tuple[float, float]] = None,
        label_order: Optional[Sequence[str]] = None,
) -> None:
    """Summarise metrics over the last *j* epochs as *bar charts*.

    Parameters
    ----------
    label_order : Sequence[str], optional
        Desired left‑to‑right ordering of **run labels** on the x‑axis. If
        provided, bars are arranged to match this sequence; otherwise, the
        original order (from *runs*) is preserved. This makes it easy to feed
        ``OrderedDict.keys()`` directly when you need a specific ordering.
    """

    suffix = "auxiliary" if aux else "primary"

    metrics: Dict[str, list] = {
        "test accuracy": [],
        "train accuracy": [],
    }
    if include_loss:
        metrics.update({"test loss": [], "train loss": []})

    # ------------------------------------------------------------------
    # Compute averages over trailing *j* epochs
    # ------------------------------------------------------------------
    for epoch_data, label in runs:
        if not epoch_data:
            continue  # skip empty runs

        if len(epoch_data) < j:
            print(
                f"[WARN] Run '{label}' only has {len(epoch_data)} epochs; using all of them for averaging."
            )
        window = epoch_data[-j:] if len(epoch_data) >= j else epoch_data
        n = len(window)

        # Average helper
        avg = lambda attr: sum(getattr(e, attr) for e in window) / n

        if include_loss:
            metrics["test loss"].append((label, avg(f"test_loss_{suffix}")))
        metrics["test accuracy"].append((label, avg(f"test_accuracy_{suffix}")))

        if train:
            if include_loss:
                metrics["train loss"].append((label, avg(f"train_loss_{suffix}")))
            metrics["train accuracy"].append((label, avg(f"train_accuracy_{suffix}")))

    # ------------------------------------------------------------------
    # Render bar charts
    # ------------------------------------------------------------------
    for metric, series in metrics.items():
        if not series:
            continue  # skip empty lists (e.g. train=False or include_loss=False)

        # ------------------------------------------------------------------
        # Resolve plotting order
        # ------------------------------------------------------------------
        if label_order is not None:
            order_map = {lbl: idx for idx, lbl in enumerate(label_order)}
            series = sorted(series, key=lambda t: order_map.get(t[0], float("inf")))

        labels, values = zip(*series)
        x_pos = range(len(labels))

        # Colours: honour explicit mapping; otherwise default cycle
        default_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        bar_colors = [
            (colors[lbl] if colors and lbl in colors else next(default_cycle))
            for lbl in labels
        ]

        plt.figure(figsize=figsize)
        plt.bar(x_pos, values, color=bar_colors)
        plt.xticks(x_pos, labels, rotation=45, ha="right")

        # Annotate bars with numeric values (e.g. "0.738")
        for x, val in zip(x_pos, values):
            plt.text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        _apply_labels_title(xlabel, ylabel, title)

        if y_window is not None:
            plt.ylim(*y_window)

        plt.grid(axis="y", linestyle=":", linewidth=0.5)
        plt.tight_layout()
        plt.show()

# Prints epoch performance metrics for training and testing
def plot_epoch_metrics(
        runs: Iterable[Tuple[Sequence, str]],
        *,
        train: bool = False,
        aux: bool = False,
        x_window: Optional[Tuple[float, float]] = None,
        y_window_loss: Optional[Tuple[float, float]] = None,
        y_window_acc: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (7, 4),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
):
    suffix = "auxiliary" if aux else "primary"

    curves = {
        "test loss": [],
        "test accuracy": [],
        "train loss": [],
        "train accuracy": [],
    }

    for epoch_data, label in runs:
        epochs = [e.epoch for e in epoch_data]

        curves["test loss"].append((
            epochs,
            [getattr(e, f"test_loss_{suffix}") for e in epoch_data],
            label,
        ))
        curves["test accuracy"].append((
            epochs,
            [getattr(e, f"test_accuracy_{suffix}") for e in epoch_data],
            label,
        ))

        if train:
            curves["train loss"].append((
                epochs,
                [getattr(e, f"train_loss_{suffix}") for e in epoch_data],
                label,
            ))
            curves["train accuracy"].append((
                epochs,
                [getattr(e, f"train_accuracy_{suffix}") for e in epoch_data],
                label,
            ))

        if len(epoch_data) > 15:
            avg_accuracy = sum(getattr(e, f"test_accuracy_{suffix}") for e in epoch_data[-15:]) / 15
            print(f"Average test accuracy of {label}: {avg_accuracy:.4f}")

    for metric, series in curves.items():
        if not series:
            continue

        plt.figure(figsize=figsize)
        for epochs, values, lbl in series:
            plt.plot(
                epochs,
                values,
                label=lbl,
                color=(colors.get(lbl) if colors and lbl in colors else None),
            )

        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel("")
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("")

        plt.grid(linestyle=":", linewidth=0.5)
        plt.legend()

        if x_window is not None:
            plt.xlim(*x_window)

        if "loss" in metric:
            if y_window_loss is not None:
                plt.ylim(*y_window_loss)
        else:
            if y_window_acc is not None:
                plt.ylim(*y_window_acc)

        plt.tight_layout()
        plt.show()


def best_test_accuracy_first_n_epochs(
        runs: Iterable[Tuple[Sequence, str]],
        n_epochs: int,
        *,
        aux: bool = False,
) -> Tuple[str, float]:
    if n_epochs <= 0:
        raise ValueError("n_epochs must be a positive integer.")
    suffix = "auxiliary" if aux else "primary"


    best_accuries_per_label = defaultdict(lambda: float("-inf"))

    for epoch_data, label in runs:
        # Examine at most the first n_epochs entries for this run
        window = epoch_data[:n_epochs]

        for e in window:
            acc = getattr(e, f"test_accuracy_{suffix}")
            if acc > best_accuries_per_label[label]:
                best_accuries_per_label[label] = acc

    return best_accuries_per_label