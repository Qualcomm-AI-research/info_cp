# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides functions to perform conformal prediction and compute upper bounds to H(Y|X)."""


import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from info_cp.utils import binary_entropy


def get_conformity_scores(
    logits: Tensor,
    labels: Optional[Tensor] = None,
    sorting_function: Optional[Callable] = None,
    conformal_method: str = "thr",
    reg_vector: Tensor = torch.tensor([0.0]),
    stochastic_scores: bool = False,
    random_noise: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute conformity scores from the logits produced by the classifier.

    Parameters
    ----------
    logits: torch.Tensor with shape (..., number of classes)
        Unnormalized output of the classifier.
    labels: torch.Tensor with shape (..., 1)
        Labels of the correct class for each score, if available.
    sorting_function: Callable
        Differentiable sorting function. If None, regular sorting is used.
    conformal_method: str
        Method used to construct conformal prediction sets.
    reg_vector: torch.Tensor
        Regularization vector is used for RAPS.
    stochastic_scores: bool
        Whether to add noise to the conformity scores
    random_noise: torch.Tensor with shape (..., number of classes)
        Random noise to be added to each score.
        Used for consistency, allowing the exact same noise vector for each method.

    Returns
    -------
    scores: torch.Tensor with shape (..., number of classes) or (..., 1)
        The conformity scores computed according to conformal_method.
        If labels are provided, only the scores corresponding to the correct label are returned.
    """
    if not stochastic_scores:
        random_noise = 0.0
    if random_noise is None:
        random_noise = torch.rand_like(logits)
    # Compute scores
    if conformal_method == "thr-l":
        scores = logits + (random_noise * 1e-6)
    elif conformal_method == "thr":
        scores = torch.softmax(logits, dim=-1) + (random_noise * 1e-6)
    elif conformal_method == "thr-lp":
        scores = torch.log_softmax(logits, dim=-1) + (random_noise * 1e-6)
    elif conformal_method in ["aps", "raps"]:
        probs = torch.softmax(logits, dim=-1)
        if sorting_function is None:
            # Sort the scores from highest to lowest and compute the cumulative sum. The APS (RAPS)
            # score is a function of the cumulative sum of probabilities, starting from the label
            # of highest probability.
            sorting_idx = probs.argsort(1, descending=True)
            sorted_probsum = probs.gather(1, sorting_idx).cumsum(dim=1)
            if conformal_method == "raps":
                sorted_probsum = sorted_probsum + reg_vector.to(probs.device)
            # Revert the sorting to retrieve the original ordering of the labels.
            probsum = sorted_probsum.gather(1, sorting_idx.argsort(1))
        else:
            # Same as above, but using a differentiable function for the sorting operation.
            sorted_scores, permutation_matrix = sorting_function(-probs)
            if permutation_matrix is None:
                raise NotImplementedError("torchsort is not yet supported for APS")
            sorted_probsum = sorted_scores.cumsum(dim=1) * (-1)
            if conformal_method == "raps":
                sorted_probsum = sorted_probsum + reg_vector.to(probs.device)
            probsum = torch.matmul(permutation_matrix, sorted_probsum.unsqueeze(-1)).flatten(-2)
        # In APS (RAPS) we add new labels to the set until their probability sums up to 1 - alpha.
        # The final score takes into account whether each label is added or not to the set, which
        # is equivalent to subtracting its own probability from the cumulative sum. This is done
        # with certain probability so that we achieve the coverage of 1 - alpha.
        if stochastic_scores:
            scores = 1 - (probsum - random_noise * probs)
        else:
            scores = 1 - (probsum - probs)
    else:
        raise ValueError(conformal_method + " is an invalid conformal method.")
    if labels is not None:
        return scores[torch.arange(logits.shape[0]), labels.view(-1)].unsqueeze(1)
    return scores


def get_confidence_sets(
    scores: Tensor,
    q_hat: Tensor,
    temperature: float = 0.0,
    relaxation: str = "smoothing",
) -> Tensor:
    """
    Compute prediction sets from conformity scores and a given threshold q_hat.

    Parameters
    ----------
    scores: torch.Tensor with shape (..., number of classes)
        The conformity score of each class for each data point.
    q_hat: torch.Tensor (scalar)
        The estimated alpha quantile of conformity scores computed on the calibration data.
    temperature: float
        The temperature used in the relaxation of the prediction sets, controlling how smooth the
        sigmoid operation is. If temperature is set to 0, we recover standard "hard" prediction
        sets, i.e. a one-hot encoded vectors. Otherwise, we get "soft" prediction sets with a
        probability in [0,1] for each class.
    relaxation: str
        The type of relaxation. It can be either "smoothing" with a sigmoid function or a "straight"
        through estimator.

    Returns
    -------
    confidence_sets: torch.Tensor with shape (..., number of classes)
        The confidence sets C(x) for each example.
    """
    assert temperature >= 0, "Temperature argument must be non-negative."
    if temperature == 0:
        # Select those above the threshold to construct C(x).
        confidence_sets = torch.as_tensor(scores >= q_hat, dtype=torch.float)
    elif relaxation == "smoothing":
        confidence_sets = torch.sigmoid((scores - q_hat) / temperature)
    elif relaxation == "straight":
        confidence_sets_forward = torch.as_tensor(scores >= q_hat, dtype=torch.float)
        confidence_sets_backward = torch.sigmoid((scores - q_hat) / temperature)
        confidence_sets = (
            confidence_sets_backward + (confidence_sets_forward - confidence_sets_backward).detach()
        )
    else:
        raise ValueError(relaxation + " is an invalid relaxation method.")
    # confidence_sets[i, j]: the probability of class j being in the prediction set of example i.
    return confidence_sets


def calibrate_from_scores(
    scores: Tensor,
    alpha: Union[float, Tensor],
    sorting_function: Optional[Callable],
) -> Tensor:
    """
    Compute the alpha quantile of scores using the given sorting_function.

    Parameters
    ----------
    scores: torch.Tensor with shape (..., number of classes)
        The conformity score of each class for each data point.
    alpha: float or torch.Tensor (scalar)
        The desired miscoverage rate in (0,1).
    sorting_function: Callable
        Differentiable sorting function. If None, regular sorting is used.
    Returns
    -------
    q_hat: torch.Tensor (scalar)
        The alpha quantile of conformity scores computed on the calibration data.
    """
    n = len(scores)
    q_level = torch.as_tensor((1.0 + (1.0 / n)) * alpha)
    q_level = q_level.to(scores.device)
    if sorting_function is None:
        # Compute (hard) quantile.
        # For very small sample sizes, set interpolation to "higher" to ensure proper coverage.
        q_hat = torch.quantile(scores, q_level, interpolation="midpoint")
    else:
        # Compute differentiable quantile.
        q_hat = diff_quantile(scores.T, q_level, sorting_function=sorting_function)
    return q_hat


def get_empirical_coverage_and_size_from_scores(
    scores: Tensor,
    labels: Tensor,
    q_hat: Tensor,
    min_set_size: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the empirical coverage and size from scores.

    Parameters
    ----------
    scores: torch.Tensor with shape (..., number of classes)
        The conformity score of each class for each data point.
    labels: torch.Tensor with shape (...,) or (..., 1)
        Labels of the test dataset.
    q_hat: torch.Tensor (scalar)
        Threshold corresponding to the alpha quantile used to define the confidence sets.
    min_set_size: int
        Minimum set size. Only used for computing bounds at test time. Leave it at zero.

    Returns
    -------
    empirical_coverage: torch.Tensor (scalar)
        The empirical coverage of the method.
    empirical_size: torch.Tensor (..., 1)
        The size of the prediction set of each data point.
    """
    # Select those above the threshold to construct C(x).
    confidence_sets = torch.as_tensor(scores >= q_hat, dtype=torch.float)
    # Compute empirical coverage (sanity check).
    empirical_coverage = confidence_sets.gather(dim=1, index=labels.view(-1, 1)).float().mean()
    # Compute the size of each prediction set.
    empirical_size = confidence_sets.sum(-1)
    # To compute some of the bounds, we clamp empirical sizes at 1 to avoid corner cases with empty
    # sets. An empty set is always incorrect, so adding any label to it does not increase prediction
    # error.
    empirical_size = torch.clamp(empirical_size, min=min_set_size)
    return empirical_coverage, empirical_size


def diff_quantile(scores: Tensor, q: Tensor, sorting_function: Callable) -> Tensor:
    """
    Compute a differentiable quantile function.

    Parameters
    ----------
    scores: torch.Tensor with shape (...,)
        The scores of the correct class for each data point in the calibration data.
    q: torch.Tensor (scalar)
        The desired quantile level.
    sorting_function: Callable
        The differentiable sorting function used to compute the quantile.

    Returns
    -------
    q_hat: torch.Tensor (scalar)
        The q quantile empirical computed of scores.
    """
    if isinstance(scores, np.ndarray):
        scores = torch.as_tensor(scores)
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    _, n = scores.shape  # (m, n) = (number of arrays to be sorted, dimension to be sorted)
    scores_sorted, _ = sorting_function(scores)
    scores_sorted = scores_sorted.flatten()
    q_apr = (n + 1) * q  # We apply a (n+1) correction to the desired quantile level
    # The approximated quantile is given by the mean and of the floor(q_apr) and ceil(q_apr)
    # elements of scores_sorted.
    q_hat = 0.5 * (
        scores_sorted[torch.floor(q_apr).long()] + scores_sorted[torch.ceil(q_apr).long()]
    )
    return q_hat


def compute_coverage_loss(
    confidence_sets: Tensor,
    labels: Tensor,
    alpha: Union[float, Tensor],
    transform: Callable = torch.square,
) -> Tensor:
    """
    Compute the coverage loss as in Stutz et al. 2022 https://arxiv.org/abs/2110.09192.

    Parameters
    ----------
    confidence_sets: torch.Tensor with shape (..., number of classes)
        Confidence sets C(x) constructed for the test dataset.
    labels: torch.Tensor with shape (...,)
        Labels of the test dataset.
    alpha: float or torch.Tensor (scalar)
        The desired miscoverage rate.
    transform: Callable
        Transform to be applied to the resulting loss

    Returns
    -------
    loss: torch.Tensor (scalar)
        The coverage loss.
    """
    labels = torch.as_tensor(labels)
    one_hot_labels = F.one_hot(labels, num_classes=confidence_sets.shape[1])
    loss = transform(torch.mean(torch.sum(confidence_sets * one_hot_labels, dim=1)) - (1 - alpha))
    return loss


def compute_general_classification_loss(
    confidence_sets: Tensor,  # Confidence sets constructed for the test dataset
    labels: Tensor,  # Labels of the test dataset
    loss_matrix: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute the classification loss as in Stutz et al. 2022 https://arxiv.org/abs/2110.09192.

    Used in the ConfTr_class baseline.

    Parameters
    ----------
    confidence_sets: torch.Tensor with shape (..., number of classes)
        Confidence sets C(x) constructed for the test dataset.
    labels: torch.Tensor with shape (...,)
        Labels of the test dataset.
    loss_matrix: torch.Tensor with shape (number of classes, number of classes)
        Matrix of size encoding similarities among classes. Set to None is all of our experiments.

    Returns
    -------
    loss: torch.Tensor (scalar)
        The classification loss
    """
    if loss_matrix is None:
        loss_matrix = torch.eye(confidence_sets.shape[1], device=labels.device)
    one_hot_labels = F.one_hot(labels, confidence_sets.shape[1])
    l1 = (1 - confidence_sets) * one_hot_labels * loss_matrix[labels]
    l2 = confidence_sets * (1 - one_hot_labels) * loss_matrix[labels]
    loss = torch.sum(torch.maximum(l1 + l2, torch.zeros_like(l1)), dim=1)
    return torch.mean(loss)


def compute_hinge_size_loss(
    confidence_sets: Tensor,
    target_size: float = 1.0,
    transform: Callable = lambda x: x,  # Transform to be applied to the resulting loss
) -> Tensor:
    """
    Compute the coverage loss as in Stutz et al. 2022 https://arxiv.org/abs/2110.09192.

    Used in the ConfTr baseline.

    Parameters
    ----------
    confidence_sets: torch.Tensor with shape (..., number of classes)
        Confidence sets C(x) constructed for the test dataset.
    target_size: float
        Labels of the test dataset.
    transform: Callable
        Transform to be applied to the resulting loss

    Returns
    -------
    loss: torch.Tensor (scalar)
        The coverage loss.
    """
    return torch.mean(
        transform(
            torch.maximum(
                torch.sum(confidence_sets, dim=1) - torch.as_tensor(target_size), torch.tensor(1e-6)
            )
        )
    )


def compute_fano_bound(
    confidence_sets: Tensor, labels: Tensor, alpha: Union[float, Tensor], n: int
) -> Tensor:
    """
    Compute the simple Fano bound.

    Parameters
    ----------
    confidence_sets: torch.Tensor with shape (..., number of classes)
        Confidence sets C(x) constructed for the test dataset.
    labels: torch.Tensor with shape (...,)
        Labels of the test dataset.
    alpha: float or torch.Tensor (scalar)
        The desired miscoverage rate.
    n: int
        Number of calibration data points used to compute confidence_sets.
        Not necessarily equal to the number test data points, i.e., confidence_sets.shape[0].

    Returns
    -------
    loss: torch.Tensor (scalar)
        The simple Fano upper bound to the conditional entropy H(Y|X).
    """
    n_classes = confidence_sets.shape[1]
    correct = confidence_sets[torch.arange(confidence_sets.size(0)), labels] > 0.5
    empirical_sizes = torch.sum(confidence_sets, dim=-1)
    # Compute average log sizes for correctly and incorrectly classified examples
    # We need to filter NaNs because there might be no correct (incorrect) examples in the batch
    avg_log_sizes_correct = torch.nan_to_num(torch.mean(empirical_sizes[correct].log()), 0.0)
    avg_log_sizes_incorrect = torch.nan_to_num(
        torch.mean((n_classes - empirical_sizes[~correct]).log()), 0.0
    )
    loss = (
        binary_entropy(alpha)
        + alpha * avg_log_sizes_incorrect
        + (1 - alpha + (1 / (n + 1))) * avg_log_sizes_correct
    )
    return loss


def compute_model_based_fano_bound(
    logits: Tensor,
    confidence_sets: Tensor,
    labels: Tensor,
    alpha: Union[float, Tensor],
    n: int,
    temperature: float,
) -> Tensor:
    """
    Compute the model-based Fano bound.

    Parameters
    ----------
    logits: torch.Tensor with shape (..., number of classes)
        Unnormalized output of the classifier.
    confidence_sets: torch.Tensor with shape (..., number of classes)
        Confidence sets C(x) constructed for the test dataset.
    labels: torch.Tensor with shape (...,)
        Labels of the test dataset.
    alpha: float or torch.Tensor (scalar)
        The desired miscoverage rate.
    n: int
        Number of calibration data points used to compute confidence_sets.
        Not necessarily equal to the number test data points, i.e., confidence_sets.shape[0].
    temperature: float
        If set to 0, confidence_sets are converted into one-hot encoded vectors before computing
        Q(Y in C(X)). Otherwise, Q(Y in C(X)) is computed using the "soft" prediction sets.

    Returns
    -------
    loss: torch.Tensor (scalar)
        The model-based Fano upper bound to the conditional entropy H(Y|X).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    if temperature == 0:
        # If confidence sets are binary (boolean), use hard assignment to compute the scores in and
        # out of C(x)
        scores_q_y_in_cx = log_probs.masked_fill(confidence_sets <= 0.5, -float("inf"))
        scores_q_y_not_in_cx = log_probs.masked_fill(confidence_sets > 0.5, -float("inf"))
    else:
        # Otherwise use soft assignment
        clamped_confidence_sets = torch.clamp(confidence_sets, min=1e-7, max=1 - 1e-7)
        scores_q_y_in_cx = log_probs + clamped_confidence_sets.log()
        scores_q_y_not_in_cx = log_probs + (1 - clamped_confidence_sets).log()
    log_q_y_in_cx = torch.log_softmax(scores_q_y_in_cx, dim=-1)
    log_q_y_not_in_cx = torch.log_softmax(scores_q_y_not_in_cx, dim=-1)
    empirical_ce_pos = torch.nanmean(
        torch.nan_to_num(-log_q_y_in_cx.gather(1, labels.view(-1, 1)), posinf=torch.nan)
    )
    empirical_ce_neg = torch.nanmean(
        torch.nan_to_num(-log_q_y_not_in_cx.gather(1, labels.view(-1, 1)), posinf=torch.nan)
    )
    loss = (
        alpha * torch.nan_to_num(empirical_ce_neg)
        + (1 - alpha + (1 / (n + 1))) * torch.nan_to_num(empirical_ce_pos)
        + binary_entropy(alpha)
    )
    return loss


def compute_dpi_bound(
    logits: Tensor,
    confidence_sets: Tensor,
    labels: Tensor,
    alpha: Union[float, Tensor],
    n: int,
    proper_upper_bound: bool = True,
) -> Tensor:
    """
    Compute the data processing inequality (DPI) bound.

    Parameters
    ----------
    logits: torch.Tensor with shape (..., number of classes)
        Unnormalized output of the classifier.
    confidence_sets: torch.Tensor with shape (..., number of classes)
        Confidence sets C(x) constructed for the test dataset.
    labels: torch.Tensor with shape (...,)
        Labels of the test dataset.
    alpha: float or torch.Tensor (scalar)
        The desired miscoverage rate.
    n: int
        Number of calibration data points used to compute confidence_sets.
        Not necessarily equal to the number test data points, i.e., confidence_sets.shape[0].
    proper_upper_bound: bool
        Whether to include Bernstein's concentration inequality to form a proper bound.

    Returns
    -------
    loss: torch.Tensor (scalar)
        The DPI upper bound to the conditional entropy H(Y|X).
    """
    n_test = logits.size(0)  # Number of test points
    log_q_y = torch.log_softmax(logits, dim=-1)  # Log-probabilities of each class
    ce = torch.mean(-log_q_y[torch.arange(n_test), labels])  # Cross-entropy

    q_e_1 = torch.sum(confidence_sets * log_q_y.exp(), 1)  # Q(Y in C(X))
    q_e_0 = 1.0 - q_e_1  # Q(Y not in C(X))
    if proper_upper_bound:
        # Include the correction given by Bernstein's concentration inequality
        delta = torch.sqrt(2 * torch.var(q_e_1) * math.log(2 / alpha) / n_test) + 7 * math.log(
            2 / alpha
        ) / (3 * (n_test - 1))
        q_e_1, q_e_0 = torch.mean(q_e_1) + delta.detach(), torch.mean(q_e_0) + delta.detach()
    else:
        # Use the (biased) empirical mean directly
        q_e_1, q_e_0 = torch.mean(q_e_1) + 1e-8, torch.mean(q_e_0) + 1e-8
    # Compute the binary KL divergence term
    # d_KL(P(Y in C(x))||Q(Y in C(x))) =
    #   = h_b(alpha) + (1 − alpha) log Q(Y in C(x)) + alpha_n log Q(Y not in C(x))
    binary_kl = (
        -binary_entropy(alpha) - (1 - alpha) * q_e_1.log() - (alpha + (1 / (n + 1))) * q_e_0.log()
    )
    return ce - binary_kl


def get_conformal_loss_batch(
    logits: Tensor,
    labels: Tensor,
    alpha: Union[float, Tensor],
    conformal_fraction: float,
    objective: str,
    conformal_method: str,
    sorting_function: Optional[Callable],
    temperature: float,
    size_loss_weight: float,
    coverage_loss_weight: float,
    cross_entropy_loss_weight: float,
    reg_vector: Tensor,
    target_size: int,
    relaxation: str,
    stochastic_scores: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the conformal training loss for each batch according to given conformal training method.

    Parameters
    ----------
    logits: torch.Tensor with shape (..., number of classes)
        Logits output by the model.
    labels: torch.Tensor with shape (...,)
        The correct class of each data point in the batch.
    alpha: float or torch.Tensor
        Desired miscoverage rate.
    conformal_fraction: float
        Fraction of the batch data to be used for calibration.
    objective: str
        Objective function to be optimized, i.e., an upper bound on the conditional entropy H(Y|X)
    conformal_method: str
        Which conformal method to use, "thr", "aps", or "raps".
    sorting_function: Callable
        Differentiable sorting function. If None, regular sorting is used.
    temperature: float
        Controls how smooth is sigmoid operation
    size_loss_weight: float
        Scalar controlling the influence of the size loss (see compute_hinge_size_loss).
    coverage_loss_weight: float
        Scalar controlling the influence of the coverage loss (see compute_coverage_loss).
    cross_entropy_loss_weight: float
        Scalar controlling the influence of the cross entropy.
    reg_vector: torch.Tensor
        Regularization vector is used for RAPS.
    target_size: int
        Reference prediction set size for hinge size loss.
    relaxation: str
        The relaxation used to compute confidence sets at training time.
    stochastic_scores: bool
        Whether to sample from a uniform distribution in APS and RAPS

    Returns
    -------
    conformal_loss: torch.Tensor (scalar)
        The loss computed on this batch of data with the desired conformal training loss function.
    coverage: torch.Tensor (scalar)
        The empirical coverage rate.
    size: torch.Tensor (scalar)
        The empirical average prediction set size.
    q_hat: torch.Tensor (scalar)
        The estimated alpha quantile of conformity scores.
    """
    # Split batches in two, assigning conformal_fraction to the calibration set
    cal_split = int(conformal_fraction * logits.shape[0])
    cal_logits = logits[:cal_split]
    cal_labels = labels[:cal_split]
    test_logits = logits[cal_split:]
    test_labels = labels[cal_split:]

    return get_conformal_loss(
        cal_logits,
        cal_labels,
        test_logits,
        test_labels,
        alpha,
        objective,
        conformal_method,
        sorting_function,
        temperature,
        size_loss_weight,
        coverage_loss_weight,
        cross_entropy_loss_weight,
        reg_vector,
        target_size,
        relaxation,
        stochastic_scores,
        None,
        None,
    )


def get_conformal_loss(
    cal_logits: Tensor,
    cal_labels: Tensor,
    test_logits: Tensor,
    test_labels: Tensor,
    alpha: Union[float, Tensor],
    objective: str,
    conformal_method: str,
    sorting_function: Optional[Callable],
    temperature: float,
    size_loss_weight: float,
    coverage_loss_weight: float,
    cross_entropy_loss_weight: float,
    reg_vector: Tensor,
    target_size: int = 1,
    relaxation: str = "smoothing",
    stochastic_scores: bool = True,
    cal_random_noise: Optional[Tensor] = None,
    eval_random_noise: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the conformal training loss according to desired conformal training method.

    Parameters
    ----------
    cal_logits: torch.Tensor with shape (number of calibration examples, number of classes)
        Logits output by the model for calibration data
    cal_labels: torch.Tensor with shape (number of calibration examples,)
        The correct class of each data point in the calibration set.
    test_logits: torch.Tensor with shape (number of test examples, number of classes)
        Logits output by the model for test data.
    test_labels: torch.Tensor with shape (number of test examples,)
        The correct class of each data point in the test set.
    alpha: float or torch.Tensor
        Desired miscoverage rate.
    objective: str
        Objective function to be optimized, i.e., an upper bound on the conditional entropy H(Y|X)
    conformal_method: str
        Which conformal method to use, "thr", "aps", or "raps".
    sorting_function: Callable
        Differentiable sorting function. If None, regular sorting is used.
    temperature: float
        Controls how smooth is sigmoid operation
    size_loss_weight: float
        Scalar controlling the influence of the size loss (see compute_hinge_size_loss).
    coverage_loss_weight: float
        Scalar controlling the influence of the coverage loss (see compute_coverage_loss).
    cross_entropy_loss_weight: float
        Scalar controlling the influence of the cross entropy.
    reg_vector: torch.Tensor
        Regularization vector is used for RAPS.
    target_size: int
        Reference prediction set size for hinge size loss.
    relaxation: str
        The relaxation used to compute confidence sets at training time.
    stochastic_scores: bool
        Whether to sample from a uniform distribution in APS and RAPS
    cal_random_noise: torch.Tensor with shape (number of calibration examples, number of classes)
        Random noise used to compute calibration scores.
    eval_random_noise: torch.Tensor with shape (number of test examples, number of classes)
        Random noise used to compute test scores.

    Returns
    -------
    conformal_loss: torch.Tensor (scalar)
        The loss computed on this batch of data with the desired conformal training loss function.
    coverage: torch.Tensor (scalar)
        The empirical coverage rate.
    size: torch.Tensor (scalar)
        The empirical average prediction set size.
    q_hat: torch.Tensor (scalar)
        The estimated alpha quantile of conformity scores.
    """
    n = cal_logits.shape[0]  # Number of calibration points
    if conformal_method != "raps":
        reg_vector = torch.tensor([0.0])
    cal_conformity_scores = get_conformity_scores(
        cal_logits,
        cal_labels,
        sorting_function,
        conformal_method,
        reg_vector,
        stochastic_scores,
        cal_random_noise,
    )
    q_hat = calibrate_from_scores(cal_conformity_scores, alpha, sorting_function)
    test_conformity_scores = get_conformity_scores(
        test_logits,
        None,
        sorting_function,
        conformal_method,
        reg_vector,
        stochastic_scores,
        eval_random_noise,
    )
    test_confidence_sets = get_confidence_sets(
        test_conformity_scores, q_hat, temperature, relaxation
    )

    if objective == "conftr":
        coverage_loss = compute_general_classification_loss(test_confidence_sets, test_labels)
        size_loss = compute_hinge_size_loss(
            test_confidence_sets, target_size=target_size, transform=lambda x: x
        )
        conformal_loss = torch.log(
            torch.as_tensor(size_loss_weight) * size_loss
            + torch.as_tensor(coverage_loss_weight) * coverage_loss
            + 1e-8
        )
    elif objective == "model_fano":
        conformal_loss = compute_model_based_fano_bound(
            test_logits, test_confidence_sets, test_labels, alpha, n, temperature
        )
    elif objective == "simple_fano":
        conformal_loss = compute_fano_bound(test_confidence_sets, test_labels, alpha, n)
    elif objective == "dpi":
        conformal_loss = compute_dpi_bound(
            test_logits, test_confidence_sets, test_labels, alpha, n, proper_upper_bound=True
        )
    elif objective == "ce":
        conformal_loss = torch.tensor(0.0)
        cross_entropy_loss_weight = 1.0
    else:
        raise ValueError("Invalid conformal method.")

    # Add the cross entropy loss on calibration data
    all_logits = torch.cat([cal_logits, test_logits], dim=0)
    all_labels = torch.cat([cal_labels, test_labels], dim=0)
    cross_entropy = F.cross_entropy(all_logits, all_labels)
    conformal_loss = conformal_loss + cross_entropy_loss_weight * cross_entropy

    test_confidence_sets = torch.greater_equal(
        test_confidence_sets, torch.ones_like(test_confidence_sets) * 0.5
    )
    coverage = test_confidence_sets.gather(1, test_labels.view(-1, 1)).float().mean()
    size = torch.sum(test_confidence_sets.float(), dim=1).mean()
    return conformal_loss, coverage, size, q_hat
