import pickle
import torch
from train.model.performance import EpochPerformance
from utils.log import log_print
from utils.vars import softmax
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Iterable, Optional

def plot_epoch_metrics(
        runs: Iterable[Tuple[Sequence, str]],
        *,
        train: bool = False,
        aux: bool   = False,
        x_window: Optional[Tuple[float, float]] = None,
        y_window_loss: Optional[Tuple[float, float]] = None,
        y_window_acc: Optional[Tuple[float, float]]  = None,
        figsize: Tuple[float, float] = (7, 4),
):
    suffix = "auxiliary" if aux else "primary"

    curves = {
        "test loss":     [],
        "test accuracy": [],
        "train loss":    [],
        "train accuracy":[],
    }

    for epoch_data, label in runs:
        epochs = [e.epoch for e in epoch_data]

        curves["test loss"].append((
            epochs,
            [getattr(e, f"test_loss_{suffix}")     for e in epoch_data],
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
                [getattr(e, f"train_loss_{suffix}")     for e in epoch_data],
                label,
            ))
            curves["train accuracy"].append((
                epochs,
                [getattr(e, f"train_accuracy_{suffix}") for e in epoch_data],
                label,
            ))

    for metric, series in curves.items():
        if not series:                   # skip empty (e.g. train=False)
            continue

        plt.figure(figsize=figsize)
        for epochs, values, lbl in series:
            plt.plot(epochs, values, label=lbl)

        plt.title(f"{metric.capitalize()} ({suffix})")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
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