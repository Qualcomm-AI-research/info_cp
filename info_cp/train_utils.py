# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides the functions used for model training and evaluation."""

import os
from copy import copy
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from info_cp.configs import TrainingConfig
from info_cp.conformal_utils import (
    calibrate_from_scores,
    get_conformal_loss,
    get_conformal_loss_batch,
    get_conformity_scores,
    get_empirical_coverage_and_size_from_scores,
)
from info_cp.data_utils import (
    get_cifar10_train_test_datasets,
    get_cifar100_train_test_datasets,
    get_emnist_train_test_datasets,
    get_fashion_train_test_datasets,
    get_mnist_train_test_datasets,
)
from info_cp.utils import get_sorting_function, seed_everything


def setup(cfg: TrainingConfig) -> dict:
    """
    Set up an experiment according to a config file.

    Parameters
    ----------
    cfg: TrainingConfig
        The configuration parameters used to define the data, model and training procedure.
        See info_cp/configs.py

    Returns
    -------
    setup_dict: dict
        Dictionary containing all the necessary objects for training or testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    print("Using device:", device)
    # Path for pretrained model
    pretrained_model_path = os.path.join(
        cfg.original_working_dir,
        "pretrained_models",
        str(cfg.dataset) + "_seed" + str(cfg.seed) + "_" + "model.pt",
    )
    if cfg.job_name == "pretrain":
        model_path = (
            pretrained_model_path  # If a pretraining job, save model to the pretrained model path
        )
    else:
        model_path = os.path.join(
            cfg.output_dir, "model.pt"
        )  # Otherwise, save the model to model_path
    torch.hub.set_dir(cfg.original_working_dir)

    # Set seed and collect datasets
    seed_everything(cfg.seed)

    if cfg.dataset == "cifar10":
        n_classes = 10
        train_dataset, test_dataset = get_cifar10_train_test_datasets(
            os.path.join(cfg.original_working_dir, cfg.datapath)
        )
        model = torchvision.models.resnet34(num_classes=n_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    elif cfg.dataset == "cifar100":
        n_classes = 100
        train_dataset, test_dataset = get_cifar100_train_test_datasets(
            os.path.join(cfg.original_working_dir, cfg.datapath)
        )
        model = torchvision.models.resnet50(num_classes=n_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    elif cfg.dataset == "fashion":
        n_classes = 10
        train_dataset, test_dataset = get_fashion_train_test_datasets(
            os.path.join(cfg.original_working_dir, cfg.datapath),
        )
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    elif cfg.dataset == "mnist":
        n_classes = 10
        train_dataset, test_dataset = get_mnist_train_test_datasets(
            os.path.join(cfg.original_working_dir, cfg.datapath),
        )
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, n_classes))

    elif cfg.dataset == "emnist":
        n_classes = 62
        train_dataset, test_dataset = get_emnist_train_test_datasets(
            os.path.join(cfg.original_working_dir, cfg.datapath),
        )
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    else:
        raise ValueError("Invalid dataset")

    model = model.to(device)

    if cfg.pretrained:
        if os.path.exists(pretrained_model_path):
            print("Loading pretrained model from " + pretrained_model_path)
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            raise ValueError(
                pretrained_model_path + " is an invalid path for the pretrained model. "
                "Pretrain the model using the pretrain config or set pretrained to False."
            )
    if cfg.hypersearch:
        n_total = len(train_dataset)
        n_val = cfg.val_examples * 2  # We set aside twice as many validation examples
        n_train = n_total - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])
        # Ensure validation and test data have the same transform
        val_dataset.dataset = copy(val_dataset.dataset)  # This is a somewhat dirty solution
        val_dataset.dataset.transform = test_dataset.transform
        # Replace the test dataset with half of the validation dataset
        val_dataset, test_dataset = torch.utils.data.random_split(
            val_dataset, [int(n_val / 2), int(n_val / 2)]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
    else:
        n_total = len(train_dataset)
        n_val = cfg.val_examples
        n_train = n_total - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])
        # Ensure validation and test data have the same transform
        val_dataset.dataset = copy(val_dataset.dataset)  # This is a somewhat dirty solution
        val_dataset.dataset.transform = test_dataset.transform
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )

    params = list(model.parameters())
    alpha = cfg.alpha

    if cfg.optimizer == "sgd":
        optimizer = SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adam":
        optimizer = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("Invalid optimizer")

    if cfg.scheduler == "none":
        scheduler = None
    elif cfg.scheduler == "step":
        scheduler = MultiStepLR(
            optimizer,
            milestones=[
                int((2.0 / 5.0) * cfg.n_epochs),
                int((3.0 / 5.0) * cfg.n_epochs),
                int((4.0 / 5.0) * cfg.n_epochs),
            ],
            gamma=cfg.gamma,
        )
    elif cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=0, last_epoch=-1, verbose=False
        )
    else:
        raise ValueError("Invalid scheduler")

    reg_vector = torch.tensor([cfg.k_reg * [0.0] + (n_classes - cfg.k_reg) * [cfg.lam_reg]])

    sorting_function = get_sorting_function(
        cfg.sorting_method,
        steepness=cfg.steepness,
    )

    if cfg.eval_only:
        print("Loading trained model from " + model_path)
        model.load_state_dict(torch.load(model_path))

    if cfg.eval_only or cfg.hypersearch:
        logger = None
    else:
        logger = SummaryWriter()

    setup_dict = {
        "model": model,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "alpha": alpha,
        "reg_vector": reg_vector,
        "sorting_function": sorting_function,
        "logger": logger,
        "model_path": model_path,
        "cfg.output_dir": cfg.output_dir,
    }
    return setup_dict


@torch.no_grad()
def evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    alpha_test: Union[float, Tensor],
    device: str,
    objective: str,
    conformal_methods: List[str],
    size_loss_weight: float,
    coverage_loss_weight: float,
    cross_entropy_loss_weight: float,
    n_splits: int,
    n_sel: int,
    reg_vector: Tensor,
    target_size: int,
    relaxation: str,
    **kwargs,
) -> dict:
    """
    Evaluate a given model in terms of coverage and prediction efficiency.

    Parameters
    ----------
    model: nn.Module
        The classifier.
    train_loader: DataLoader
        The data loader containing the training data.
    val_loader: DataLoader
        The data loader containing the calibration data.
    test_loader: DataLoader
        The data loader containing the test data.
    alpha_test: float
        Desired miscoverage rate. This might be different to the alpha used for conformal training.
    device: str
        The device where the model will be loaded for evaluation.
    objective: str
        Objective function to be optimized, i.e., an upper bound on the conditional entropy H(Y|X).
    conformal_methods: List[str]
        List of conformal methods to use, "thr", "aps", or "raps".
    size_loss_weight: float
        Scalar controlling the influence of the size loss (see compute_hinge_size_loss).
    coverage_loss_weight: float
        Scalar controlling the influence of the coverage loss (see compute_coverage_loss).
    cross_entropy_loss_weight: float
        Scalar controlling the influence of the cross entropy.
    n_splits: int
        Number of different calibration-test splits to consider at test time.
    n_sel: int
        Number of data points reserved for model selection.
    reg_vector: torch.Tensor
        Regularization vector is used for RAPS.
    target_size: int
        Reference prediction set size for hinge size loss.
    relaxation: str
        The relaxation used to compute confidence sets at training time.

    Returns
    -------
    res: dict
        Dictionary containing relevant performance statistics.
    """
    model.eval()
    res = {}
    # Compute accuracy and loss on training data
    _, res["train_acc"] = get_ce_and_accuracy_dataloader(model, train_loader, device)
    # Get scores and labels for calibration and test sets
    val_logits, val_labels = compute_logits_dataloader(model, val_loader, device)
    res["val_logits"], res["val_labels"] = val_logits.cpu(), val_labels.cpu()
    test_logits, test_labels = compute_logits_dataloader(model, test_loader, device)
    res["test_logits"], res["test_labels"] = test_logits.cpu(), test_labels.cpu()
    # Group the scores to simulate different calibration/test splits
    logits = torch.cat([val_logits[n_sel:, ...], test_logits], dim=0).to("cpu")
    labels = torch.cat([val_labels[n_sel:], test_labels], dim=0).to("cpu")
    assert len(logits) == len(labels)
    n_total = len(logits)
    n_val = len(val_logits) - n_sel
    n_eval = n_total - n_val
    res["val_acc"] = (
        torch.as_tensor(torch.argmax(val_logits, dim=-1) == val_labels, dtype=torch.float)
        .mean()
        .item()
    )
    res["test_acc"] = (
        torch.as_tensor(torch.argmax(test_logits, dim=-1) == test_labels, dtype=torch.float)
        .mean()
        .item()
    )
    for i in range(n_splits):
        # Define a new split
        perm = torch.randperm(n_total)
        cal_logits, eval_logits = torch.split(logits[perm, ...], [n_val, n_eval], dim=0)
        cal_labels, eval_labels = torch.split(labels[perm, ...], [n_val, n_eval], dim=0)
        cal_random_noise = torch.rand_like(cal_logits)
        eval_random_noise = torch.rand_like(eval_logits)
        for conformal_method in conformal_methods:
            if i == 0:
                # Compute conformal quantile
                val_conformity_scores = get_conformity_scores(
                    val_logits, val_labels, None, conformal_method, reg_vector
                )
                q_hat = calibrate_from_scores(
                    val_conformity_scores, alpha=alpha_test, sorting_function=None
                )
                res["q_hat_" + conformal_method] = q_hat
                # Initialize metrics
                res["test_loss_" + conformal_method] = torch.zeros(n_splits)
                res["test_coverage_" + conformal_method] = torch.zeros(n_splits)
                res["test_size_" + conformal_method] = torch.zeros(n_splits)
            # Evaluate conformal prediction results
            test_loss, test_coverage, test_size, _ = get_conformal_loss(
                cal_logits,
                cal_labels,
                eval_logits,
                eval_labels,
                alpha=alpha_test,
                objective=objective,
                conformal_method=conformal_method,
                sorting_function=None,
                temperature=0.0,
                size_loss_weight=size_loss_weight,
                coverage_loss_weight=coverage_loss_weight,
                cross_entropy_loss_weight=cross_entropy_loss_weight,
                reg_vector=reg_vector,
                target_size=target_size,
                relaxation=relaxation,
                stochastic_scores=True,
                cal_random_noise=cal_random_noise,
                eval_random_noise=eval_random_noise,
            )
            res["test_loss_" + conformal_method][i] = test_loss
            res["test_coverage_" + conformal_method][i] = test_coverage
            res["test_size_" + conformal_method][i] = test_size.float().mean()
            if i == n_splits - 1:
                res["test_loss_" + conformal_method] = res["test_loss_" + conformal_method].mean()
                res["test_coverage_" + conformal_method] = res[
                    "test_coverage_" + conformal_method
                ].mean()
                res["test_size_std_" + conformal_method] = res[
                    "test_size_" + conformal_method
                ].std()
                res["test_size_" + conformal_method] = res["test_size_" + conformal_method].mean()
    return res


def train_conformal(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Union[CosineAnnealingLR, MultiStepLR],
    max_epochs: int,
    logger: SummaryWriter,
    alpha: float,
    alpha_test: float,
    conformal_fraction: float,
    objective: str,
    sorting_function: Callable,
    temperature: float,
    size_loss_weight: float,
    coverage_loss_weight: float,
    cross_entropy_loss_weight: float,
    early_stopping_patience: int,
    evaluation_splits: int,
    n_sel: int,
    conformal_method_train: str,
    conformal_methods_test: List[str],
    reg_vector: torch.Tensor,
    target_size: int,
    relaxation: str,
    verbose: bool,
    **kwargs,
):
    """
    Train a model using conformal training with the given objective.

    Parameters
    ----------
    model: nn.Module
        The classifier.
    device: str
        The device where the model is supposed to run.
    train_loader: DataLoader
        The data loader containing the training data.
    val_loader: DataLoader
        The data loader containing the calibration data.
    test_loader: DataLoader
        The data loader containing the test data.
    optimizer: Optimizer
        The pytorch optimizer associated with the model.
    scheduler: LRScheduler
        Learning rate scheduler.
    max_epochs: int
        The maximum number of epochs the model is trained for.
        Training might halt earlier due to early stopping.
    logger: SummaryWriter
        Logger used to write intermediary results to tensorboard.
    alpha: float
        The desired miscoverage rate used to compute the conformal training objectives.
    alpha_test: float
        Desired miscoverage rate. This might be different to the alpha used for conformal training.
    conformal_fraction: float
        Fraction of the batch data to be used for calibration.
    objective: str
        The training objective, either "ce", "conftr", "conftr_class", "simple_fano", "model_fano"
        or "dpi".
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
    early_stopping_patience: int
        Number of epochs to wait for a performance improvement before early stopping.
    evaluation_splits: int
        Number of different random calibration/test splits to use to estimate performance.
    n_sel:int
        Number of data points reserved for model selection.
    conformal_method_train: str
        The conformal method to use for training. Note this is a single method not a list like
        conformal_methods_test.
        In all experiments in the paper, we use "thr-l" for training.
    conformal_methods_test: List[str]
        List of conformal methods to use, "thr-l", "thr", "aps", or "raps".
    reg_vector: torch.Tensor
        Regularization vector is used for RAPS.
    target_size: int
        Reference prediction set size for hinge size loss.
    relaxation: str
        The relaxation used to compute confidence sets at training time.
    verbose: bool
        Whether to print extra information during training.
    """
    best_model_statedict = model.state_dict().copy()
    best_model_epoch = 0
    best_model_score = np.inf
    running_conformal_loss = 0.0
    for e in range(max_epochs):
        with torch.no_grad():
            # Evaluate the model using conformal prediction
            res = evaluate(
                model,
                train_loader,
                val_loader,
                test_loader,
                alpha_test,
                device,
                objective,
                conformal_methods_test,
                size_loss_weight,
                coverage_loss_weight,
                cross_entropy_loss_weight,
                evaluation_splits,
                n_sel,
                reg_vector,
                target_size,
                relaxation,
            )
            model_score = selection_criterion(res, n_sel)

        if model_score <= best_model_score:
            best_model_score = model_score
            best_model_epoch = e
            best_model_statedict = model.state_dict().copy()
        else:
            # check if early stopping patience exceeded
            if e - best_model_epoch >= early_stopping_patience:
                break

        if logger is not None:
            logger.add_scalar("train_accuracy", res["train_acc"], e)
            logger.add_scalar("test_accuracy", res["test_acc"], e)
            logger.add_scalar("selection_score", model_score, e)

        print_str = (
            f"Training Epoch: {e} \tLoss: {running_conformal_loss:.6f} \tTrain acc: "
            f"{res['train_acc']:.3f} \tTest acc: {res['test_acc']:.3f}"
        )

        for cp_method in conformal_methods_test:
            print_str += (
                f"\t{cp_method.upper()} ({res['test_coverage_' + cp_method]:.3f}, "
                f"{res['test_size_' + cp_method]:.3f})"
            )
            if logger is not None:
                logger.add_scalar(
                    "test_coverage_" + cp_method, res["test_coverage_" + cp_method], e
                )
                logger.add_scalar("test_size_" + cp_method, res["test_size_" + cp_method], e)
                logger.add_scalar("test_loss_" + cp_method, res["test_loss_" + cp_method], e)
                logger.add_scalar("q_hat_" + cp_method, res["q_hat_" + cp_method], e)
        if verbose:
            print(print_str)

        model.train()
        running_conformal_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.ndim > 1:
                dist = torch.distributions.Categorical(y)
                y = dist.sample([1]).squeeze()
            optimizer.zero_grad()
            logits = model(x)
            # If alpha is learnable, recompute it every iteration
            if isinstance(alpha, nn.Module):
                this_step_alpha = alpha()
            else:
                this_step_alpha = alpha
            # Compute the training loss and optimize it
            conformal_loss, _, _, _ = get_conformal_loss_batch(
                logits,
                y,
                this_step_alpha,
                conformal_fraction,
                objective,
                conformal_method_train,
                sorting_function,
                temperature,
                size_loss_weight,
                coverage_loss_weight,
                cross_entropy_loss_weight,
                reg_vector,
                target_size,
                relaxation,
                stochastic_scores=False,
            )
            conformal_loss.backward()
            optimizer.step()
            running_conformal_loss.append(x.shape[0] * conformal_loss.detach().item())
        running_conformal_loss = (np.sum(running_conformal_loss) / len(train_loader.dataset)).item()

        if logger is not None:
            logger.add_scalar("training_loss", running_conformal_loss, e)
            logger.add_scalar("alpha", this_step_alpha, e)

        if scheduler is not None:
            scheduler.step()
    with torch.no_grad():
        # Load best performing model
        model.load_state_dict(best_model_statedict)


@torch.no_grad()
def compute_logits_dataloader(
    model: nn.Module, dataloader: DataLoader, device: str
) -> Tuple[Tensor, Tensor]:
    """
    Loop through the dataloader computing the scores for all datapoints.

    Returns the corresponding scores and labels.

    Parameters
    ----------
    model: nn.Module
        The classifier.
    dataloader: DataLoader
        The data loader containing the data we want to evaluate the method on.
    device: str
        The device where the model is supposed to run.

    Returns
    -------
    logits: torch.Tensor with shape (..., number of classes)
        The logits output by the model.
    ys: torch.Tensor with shape (...,)
        The correct labels.
    """
    logits = []
    ys = []
    for x, y in dataloader:
        # Compute the scores
        logits.append(model(x.to(device)).to("cpu"))
        if y.ndim > 1:
            dist = torch.distributions.Categorical(y)
            y = dist.sample([1]).squeeze()
        ys.append(y)
    logits = torch.concat(logits, dim=0).squeeze()
    ys = torch.concat(ys, dim=0).squeeze()
    return logits, ys


def get_ce_and_accuracy_dataloader(
    model: nn.Module, dataloader: DataLoader, device: str
) -> Tuple[float, float]:
    """
    Loop through the dataloader computing the scores for all datapoints.

    Returns the cross-entropy and accuracy of the model for the data contained in the dataloader.

    Parameters
    ----------
    model: nn.Module
        The classifier.
    dataloader: DataLoader
        The data loader containing the data we want to evaluate the method on.
    device: str
        The device where the model is supposed to run.

    Returns
    -------
    avg_loss: torch.Tensor (scalar)
        Average loss of the model on the data contained in the data loader.
    acc: torch.Tensor (scalar)
        Accuracy of the model on the data contained in the data loader.
    """
    n_samples = len(dataloader.dataset)
    n_correct_pred = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += F.cross_entropy(pred, y)
            if y.ndim > 1:
                y = torch.argmax(y, dim=-1)
            n_correct_pred += torch.as_tensor(torch.argmax(pred, dim=-1) == y).sum()
    avg_loss = total_loss / n_samples
    acc = n_correct_pred / n_samples
    return avg_loss, acc


def selection_criterion(res: dict, n_sel: int) -> Tensor:
    """
    Compute the loss on the calibration set, using one half of it for calibration and the other half
    for testing.

    Parameters
    ----------
    res: dict
        Dictionary returned by the evaluate function.
    n_sel: int
        Number of data points reserved for model selection.

    Returns
    -------
    cross_entropy: torch.Tensor (scalar)
        The cross-entropy of the model evaluated on the data points used for model selection.
    """
    if n_sel <= 0:
        return np.inf
    logits, labels = res["val_logits"], res["val_labels"]
    sel_logits, sel_labels = logits[:n_sel, ...], labels[:n_sel]
    return F.cross_entropy(sel_logits, sel_labels)


def alpha_sweep(res: dict, conformal_methods_test: List[str], logger: SummaryWriter) -> None:
    """
    Evaluate the methods at different alpha values using the logits stored in the res dictionary.

    Parameters
    ----------
    res: dict
        Dictionary returned by the evaluate function.
    conformal_methods_test: List[str]
        List of conformal methods to use, "thr-l", "thr", "aps", or "raps".
    logger: SummaryWriter
        Logger used to write intermediary results to tensorboard.
    """
    alphas = torch.arange(1e-12, 0.5, 0.001)
    for conformal_method in conformal_methods_test:
        for i, alpha in enumerate(alphas):
            # Compute conformal quantile
            val_conformity_scores = get_conformity_scores(
                res["val_logits"], res["val_labels"], None, conformal_method
            )
            q_hat = calibrate_from_scores(val_conformity_scores, alpha=alpha, sorting_function=None)
            test_conformity_scores = get_conformity_scores(
                res["test_logits"], None, None, conformal_method
            )
            _, test_size = get_empirical_coverage_and_size_from_scores(
                test_conformity_scores, res["test_labels"], q_hat, 0
            )
            logger.add_scalar("alpha_sweep_" + conformal_method, test_size.float().mean(), i)
