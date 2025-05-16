from collections import defaultdict

import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Iterable, Optional, Dict

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


    best_accuries_per_label = defaultdict(0)

    for epoch_data, label in runs:
        # Examine at most the first n_epochs entries for this run
        window = epoch_data[:n_epochs]

        for e in window:
            acc = getattr(e, f"test_accuracy_{suffix}")
            if acc > best_accuries_per_label[label]:
                best_accuries_per_label[label] = acc

    return best_accuries_per_label