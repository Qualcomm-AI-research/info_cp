# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides auxiliary functions used throughout the code."""


import os
import random
import sys
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from diffsort import DiffSortNet
from numpy.typing import NDArray
from ray.tune import ResultGrid
from torch import Tensor


def binary_entropy(alpha: Union[float, Tensor]) -> Union[float, Tensor]:
    """Compute the binary entropy at alpha."""
    if isinstance(alpha, torch.Tensor):
        return -alpha * torch.log(alpha) - (1 - alpha) * torch.log(1 - alpha)
    return -alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha)


def bootstrap(
    x: NDArray, y: NDArray, n_samples: int, replace: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Bootstrap sampling of tuple (x, y).

    Parameters
    ----------
    x, y: NDArray
        Input and output variables.
    n_samples: int
        Number of samples to draw.
    replace: bool
        Whether to sample with replacement.

    Returns
    -------
    x, y: NDArray
        Resampled input and output variables.
    """
    assert x.shape[0] == y.shape[0]
    assert n_samples <= x.shape[0]
    idx = np.arange(x.shape[0])
    sampled_idx = np.random.choice(idx, size=n_samples, replace=replace)
    return x[sampled_idx], y[sampled_idx]


def get_sorting_function(method: str, steepness: float) -> Callable:
    """
    Return a differentiable sorting function defined by steepness.

    Parameters
    ----------
    method: str
        The differentiable sorting method to be used. Only "diffsort" supported for now.
    steepness: float
        Smoothing factor for diffsort. The higher, the less smooth is the sorting.

    Returns
    -------
    sorting_function: Callable
        The differentiable sorting function.
    """
    if method == "diffsort":

        def sorting_function(scores):
            sorting_net = DiffSortNet(
                "bitonic", scores.shape[1], steepness=steepness, device=scores.device
            )
            sorted_scores, permutation_matrix = sorting_net(scores)
            return sorted_scores, permutation_matrix

    else:
        raise ValueError("Invalid differentiable sorting method.")
    return sorting_function


def seed_everything(seed: int):
    """Set the random seeds for random number generators in Pytorch, numpy and native Python."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


def print_results(res: dict, file_name: Optional[str] = None) -> None:
    """Print results and save them to file_name."""
    if file_name is None:
        file = sys.stdout
    else:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        file = open(file_name, "w", encoding="utf8")
    print("FINAL RESULTS", file=file)

    if res["train_acc"] is not None:
        print(f"Train acc: {res['train_acc']:.4f} \tTest acc: {res['test_acc']:.4f}", file=file)
    else:
        print(f"Test acc: {res['test_acc']:.4f}", file=file)
    for cp_method in ["thr", "thr-l", "aps", "raps"]:
        print(f"Results for {cp_method.upper()}", file=file)
        print(
            f"Test coverage: {res['test_coverage_' + cp_method]:.2f} "
            f"\tTest size: {res['test_size_' + cp_method]:.2f} "
            f"± {res['test_size_std_' + cp_method]:.2f}",
            file=file,
        )
    if file_name:
        file.close()


def print_tune_results(res: ResultGrid, file_name: Optional[str] = None) -> None:
    """Print ray tune results and save them to file_name."""
    if file_name is None:
        file = sys.stdout
    else:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        file = open(file_name, "w", encoding="utf8")

    # Iterate over results
    for i, result in enumerate(res):
        if result.error:
            print(f"Trial #{i} had an error:", result.error, file=file)
            continue
        if "test_size" in result.metrics:
            print(
                f"Trial #{i} finished successfully with a mean accuracy metric of:",
                result.metrics["test_size"],
                file=file,
            )
        else:
            print(f"Trial #{i} has not been completed yet.", file=file)

    try:
        best_result = res.get_best_result(metric="test_size", mode="min")
        print(f"Best trial config: {best_result.config}", file=file)
        print(f"Best trial final test set size: {best_result.metrics['test_size']}", file=file)
        print(f"Results location, {best_result.path}", file=file)
    except OSError:
        print("No completed trials yet.", file=file)

    if file_name:
        file.close()
