# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides the functions used to run conformal prediction with side information on CIFAR100 and
EMNIST."""


import copy
import os
from typing import List, Tuple, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from torch import Tensor, nn
from torch.utils.data import DataLoader

from info_cp.configs import TrainingConfig
from info_cp.conformal_utils import (
    calibrate_from_scores,
    get_confidence_sets,
    get_conformity_scores,
)
from info_cp.train_utils import evaluate, setup
from info_cp.utils import print_results, seed_everything

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=TrainingConfig, group="experiment")


class SideInformationModel(nn.Module):
    """
    SideInformationModel object defines a model computing a probability distribution over labels Y
    given input data x and side information z.

    Parameters
    ----------
    model_base: nn.Module
        The classifier computing Q(Y|x).
    n_classes: int
        Number of classes the label Y can take.
    n_side: int
        Number of classes the side information Z can take.

    Attributes
    ----------
    encoder: nn.Module
        Neural network that maps input x to a latent representation given by the last layer of
        model_base.
    in_features: int
        The dimensionality of the representation defined by the encoder.
    fc_side: nn.Module
        A linear model mapping the representation given by the encoder to a distribution over side
        information Z.
    """

    def __init__(self, model_base: nn.Module, n_classes: int, n_side: int):
        """Initialize the base model as well as the encoder and additional classifier head."""
        super().__init__()
        self.model_base = model_base
        self.model_base.eval()
        for param in self.model_base.parameters():
            param.requires_grad = False

        # We define all but the last layer of the base model as encoder.
        self.encoder = nn.Sequential(*list(self.model_base.children())[:-1])
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.in_features = list(self.model_base.children())[-1].in_features
        self.n_classes = n_classes
        self.n_side = n_side
        self.fc_side = nn.Linear(self.in_features + self.n_classes, self.n_side)

    def forward_side(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute logits of Q(Z|x, y), the probability of side information Z given x, y.

        Parameters
        ----------
        x: torch.Tensor
            The input data.
        y: torch.Tensor
            The ground-truth labels.

        Returns
        -------
        p_z: torch.Tensor with shape (..., n_side)
            The logits of the conditional categorical distribution Q(Z|x, y).
        """
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        y = y.long()
        y_one_hot = F.one_hot(y.squeeze(), num_classes=self.n_classes)
        p_z = self.fc_side(torch.cat([x, y_one_hot], dim=1))
        return p_z

    def predict_with_side(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Compute logits of Q(Y|x, z), the probability of label Y given x, z.

        Parameters
        ----------
        x: torch.Tensor
            The input data.
        y: torch.Tensor
            The ground-truth labels.

        Returns
        -------
        p_z: torch.Tensor with shape (..., n_side)
            The logits of the conditional categorical distribution Q(Y|x, z).
        """
        # dim_x * dim_y
        logits_base = torch.log_softmax(self.predict_base(x), 1)
        logits_side_per_class = []
        for y_i in range(self.n_classes):
            y_for_x = torch.ones(x.size(0), 1) * y_i
            y_for_x = y_for_x.to(x.device)
            z_pred = torch.log_softmax(self.forward_side(x, y_for_x), 1)
            logits_side_per_class.append(z_pred[:, None, :])

        # dim_x * dim_y * dim_z
        logits_side_per_class = torch.cat(logits_side_per_class, dim=1)
        logits_conditioned_on_side = torch.log_softmax(
            logits_base[:, :, None] + logits_side_per_class, dim=1
        )
        return logits_conditioned_on_side[torch.arange(x.size(0)), :, z]

    def predict_base(self, x: Tensor) -> Tensor:
        """
        Compute log Q(Y|x) using the base model (no side information is used).

        Parameters
        ----------
        x: torch.Tensor
            The input data.

        Returns
        -------
        p_y: torch.Tensor
            The conditional log-probabilities Q(Y|x).
        """
        p_y = torch.log_softmax(self.model_base(x), 1)
        return p_y

    def forward(self, x: Tensor, z: Tensor):
        """Compute logits of Q(Y|x, z), the probability of label Y given x, z."""
        return self.predict_with_side(x, z)


def side_information_training(
    side_model: SideInformationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
) -> SideInformationModel:
    """
    Train auxiliary model Q(Z|x,y).

    Parameters
    ----------
    side_model: SideInformationModel
        The model to be trained to predict Q(Z|x,y).
    train_loader: DataLoader
        Data loader containing the training data.
    val_loader: Dataloader
        Data loader containing the validation data.
    cfg: TrainingConfig
        Config file specifying the experiment configuration

    Returns
    -------
    side_model: SideInformationModel
        The trained model Q(Z|x,y).
    """
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    print("Using device:", device)
    side_model = side_model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in side_model.parameters() if p.requires_grad], lr=cfg.lr_side
    )
    for e in range(cfg.n_epochs_side):
        running_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits_z = side_model.forward_side(x, y[:, 0].unsqueeze(1))
            loss_z = F.cross_entropy(logits_z, y[:, 1])
            loss_z.backward()
            optimizer.step()
            running_loss.append(loss_z.detach().item())
        running_loss = (np.sum(running_loss) / len(running_loss)).item()

        # evaluate
        with torch.no_grad():
            num_corr, seen = 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred_z = torch.argmax(side_model.forward_side(x, y[:, 0].unsqueeze(1)), 1)
                num_corr += (pred_z == y[:, 1]).sum()
                seen += x.shape[0]
            acc = (num_corr / seen).item()

        print(f"Training Epoch: {e} \tLoss: {running_loss:.6f} \tTest acc: {acc:.3f}")
    return side_model


@torch.no_grad()
def compute_logits_dataloader_with_side_information(
    model: nn.Module,  # The classifier
    dataloader: DataLoader,
    device: str,
    use_side: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Loop through the dataloader computing the scores for all datapoints.

    Returns the corresponding scores and labels.
    """
    logits, ys, zs = [], [], []

    for x, y in dataloader:
        # Compute the scores
        x, y = x.to(device), y.to(device)
        if np.random.random() <= use_side:
            logits_pred = model.predict_with_side(x, y[:, 1])
        else:
            logits_pred = model.predict_base(x)
        logits.append(logits_pred)
        ys.append(y[:, 0])
        zs.append(y[:, 1])

    logits = torch.concat(logits, dim=0).squeeze()
    ys = torch.concat(ys, dim=0).squeeze()
    zs = torch.concat(zs, dim=0).squeeze()
    return logits, ys, zs


@torch.no_grad()
def evaluate_side(
    side_model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    alpha_test: Union[float, Tensor],
    device: str,
    conformal_methods: List[str],
    n_splits: int,
    n_sel: int,
    reg_vector: Tensor,
    use_side_cal: float,
    use_side_test: float,
    group_balanced: bool = True,
    **kwargs,
) -> dict:
    """
    Evaluate a model using conformal prediction with side information.

    Parameters
    ----------
    side_model: nn.Module
        The classifier with optional side information.
    val_loader: DataLoader,
        Data loader containing the validation data.
    test_loader: DataLoader,
        Data loader containing the test data.
    alpha_test: float | Tensor
        Desired miscoverage rate at test time.
    device: str
        The device where to load and run the model.
    conformal_methods: List[str]
        Which conformal methods to use, "thr", "aps", or "raps".
    n_splits: int
        Number of different validation-test splits to consider at test time.
    n_sel: int
        Number of data points reserved for model selection.
    reg_vector: Tensor
        Regularization vector is used for RAPS.
    use_side_cal: float
        A number in [0,1] corresponding to the fraction of side information to use for calibration.
    use_side_test: float
        A number in [0,1] corresponding to the fraction of side information to use for testing.
    group_balanced: bool
        Whether to run group balanced CP.

    Returns
    -------
    res: dict
        Dictionary containing relevant performance statistics.
    """
    side_model.eval()
    res = {}
    # Get scores and labels for validation and test sets.
    val_logits, val_labels, val_side = compute_logits_dataloader_with_side_information(
        side_model, val_loader, device, use_side_cal
    )
    res["val_logits"], res["val_labels"], res["val_side"] = (
        val_logits.cpu(),
        val_labels.cpu(),
        val_side.cpu(),
    )

    test_logits, test_labels, test_side = compute_logits_dataloader_with_side_information(
        side_model, test_loader, device, use_side_test
    )
    res["test_logits"], res["test_labels"], res["test_side"] = (
        test_logits.cpu(),
        test_labels.cpu(),
        test_side.cpu(),
    )

    # Compute how many states we have for the side information.
    n_side = torch.unique(test_side).size(0)

    # Group the scores to simulate different validation/test splits.
    logits = torch.cat([val_logits[n_sel:, ...], test_logits], dim=0).to("cpu")
    labels = torch.cat([val_labels[n_sel:], test_labels], dim=0).to("cpu")
    side = torch.cat([val_side[n_sel:], test_side], dim=0).to("cpu")
    assert len(logits) == len(labels) == len(side)
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
        # Define a new split.
        perm = torch.randperm(n_total)
        cal_logits, eval_logits = torch.split(logits[perm, ...], [n_val, n_eval], dim=0)
        cal_labels, eval_labels = torch.split(labels[perm, ...], [n_val, n_eval], dim=0)
        cal_side, eval_side = torch.split(side[perm, ...], [n_val, n_eval], dim=0)
        for conformal_method in conformal_methods:
            if group_balanced:
                if i == 0:
                    # Initialize metrics.
                    res["test_coverage_" + conformal_method] = torch.zeros(n_splits, n_side)
                    res["test_size_" + conformal_method] = torch.zeros(n_splits, n_side)

                for z_i in range(n_side):
                    if conformal_method != "raps":
                        reg_vector = torch.tensor([0.0])
                    cal_conformity_scores = get_conformity_scores(
                        cal_logits[cal_side == z_i],
                        cal_labels[cal_side == z_i],
                        None,
                        conformal_method,
                        reg_vector,
                    )
                    q_hat = calibrate_from_scores(cal_conformity_scores, alpha_test, None)
                    test_conformity_scores = get_conformity_scores(
                        eval_logits[eval_side == z_i], None, None, conformal_method, reg_vector
                    )
                    test_confidence_sets = get_confidence_sets(test_conformity_scores, q_hat, 0)

                    coverage = (
                        test_confidence_sets.gather(1, eval_labels[eval_side == z_i].view(-1, 1))
                        .float()
                        .mean()
                    )
                    size = torch.sum(test_confidence_sets.float(), dim=1).mean()

                    res["test_coverage_" + conformal_method][i, z_i] = coverage
                    res["test_size_" + conformal_method][i, z_i] = size

                    if i == n_splits - 1:
                        res["test_coverage_" + conformal_method + "_z=" + str(z_i)] = res[
                            "test_coverage_" + conformal_method
                        ][:, z_i].mean()
                        res["test_coverage_std_" + conformal_method + "_z=" + str(z_i)] = res[
                            "test_coverage_" + conformal_method
                        ][:, z_i].std()

                        res["test_size_" + conformal_method + "_z=" + str(z_i)] = res[
                            "test_size_" + conformal_method
                        ][:, z_i].mean()
                        res["test_size_std_" + conformal_method + "_z=" + str(z_i)] = res[
                            "test_size_" + conformal_method
                        ][:, z_i].std()
                if i == n_splits - 1:
                    res["test_coverage_std_" + conformal_method] = (
                        res["test_coverage_" + conformal_method].mean(1).std()
                    )
                    res["test_coverage_" + conformal_method] = res[
                        "test_coverage_" + conformal_method
                    ].mean()

                    res["test_size_std_" + conformal_method] = (
                        res["test_size_" + conformal_method].mean(1).std()
                    )
                    res["test_size_" + conformal_method] = res[
                        "test_size_" + conformal_method
                    ].mean()
            else:
                if i == 0:
                    # Initialize metrics
                    res["test_coverage_" + conformal_method] = torch.zeros(n_splits)
                    res["test_size_" + conformal_method] = torch.zeros(n_splits)

                if conformal_method != "raps":
                    reg_vector = torch.tensor([0.0])
                cal_conformity_scores = get_conformity_scores(
                    cal_logits, cal_labels, None, conformal_method, reg_vector
                )
                q_hat = calibrate_from_scores(cal_conformity_scores, alpha_test, None)
                test_conformity_scores = get_conformity_scores(
                    eval_logits, None, None, conformal_method, reg_vector
                )
                test_confidence_sets = get_confidence_sets(test_conformity_scores, q_hat, 0)

                coverage = test_confidence_sets.gather(1, eval_labels.view(-1, 1)).float().mean()
                size = torch.sum(test_confidence_sets.float(), dim=1).mean()

                res["test_coverage_" + conformal_method][i] = coverage
                res["test_size_" + conformal_method][i] = size

                if i == n_splits - 1:
                    res["test_coverage_std_" + conformal_method] = res[
                        "test_coverage_" + conformal_method
                    ].std()
                    res["test_coverage_" + conformal_method] = res[
                        "test_coverage_" + conformal_method
                    ].mean()

                    res["test_size_std_" + conformal_method] = res[
                        "test_size_" + conformal_method
                    ].std()
                    res["test_size_" + conformal_method] = res[
                        "test_size_" + conformal_method
                    ].mean()
    return res


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: TrainingConfig) -> None:
    """Run conformal prediction with side information."""
    # Set dirs
    output_dir = os.getcwd()
    print("output dir is", output_dir)
    original_working_dir = get_original_cwd()
    cfg.output_dir = output_dir
    cfg.original_working_dir = original_working_dir
    cfg.eval_only = True  # We want to load an existing trained model

    # Set the experiment up
    setup_dict = setup(cfg)
    seed_everything(cfg.seed)

    # Evaluate final results for all conformal prediction methods without side information
    conformal_methods = ["thr", "thr-l", "aps", "raps"]
    print(type(setup_dict["model"]))

    if cfg.pretrained_path_for_side != "":
        setup_dict["model"].load_state_dict(torch.load(cfg.pretrained_path_for_side))

    if cfg.dataset == "cifar100":
        side_model = SideInformationModel(
            model_base=copy.deepcopy(setup_dict["model"]), n_classes=100, n_side=20
        )
    elif cfg.dataset == "emnist":
        side_model = SideInformationModel(
            model_base=copy.deepcopy(setup_dict["model"]), n_classes=62, n_side=3
        )
    else:
        raise Exception()

    train_loader, val_loader, test_loader = (
        setup_dict["train_loader"],
        setup_dict["val_loader"],
        setup_dict["test_loader"],
    )
    if cfg.dataset == "cifar100":

        def sparse2joint(targets):
            """
            Convert Pytorch CIFAR100 sparse targets to coarse targets.

            Usage:
                training_data = torchvision.datasets.CIFAR100(path)
                training_data.targets = sparse2coarse(training_data.targets)

            Parameters
            ----------
            targets: torch.Tensor with shape (..., 1)
                The ground-truth labels.

            Returns
            -------
            targets: torch.Tensor (..., 2)
                The ground-truth labels followed by the coarser super-class labels.
            """
            coarse_labels = np.array(
                [
                    4,
                    1,
                    14,
                    8,
                    0,
                    6,
                    7,
                    7,
                    18,
                    3,
                    3,
                    14,
                    9,
                    18,
                    7,
                    11,
                    3,
                    9,
                    7,
                    11,
                    6,
                    11,
                    5,
                    10,
                    7,
                    6,
                    13,
                    15,
                    3,
                    15,
                    0,
                    11,
                    1,
                    10,
                    12,
                    14,
                    16,
                    9,
                    11,
                    5,
                    5,
                    19,
                    8,
                    8,
                    15,
                    13,
                    14,
                    17,
                    18,
                    10,
                    16,
                    4,
                    17,
                    4,
                    2,
                    0,
                    17,
                    4,
                    18,
                    17,
                    10,
                    3,
                    2,
                    12,
                    12,
                    16,
                    12,
                    1,
                    9,
                    19,
                    2,
                    10,
                    0,
                    1,
                    16,
                    12,
                    9,
                    13,
                    15,
                    13,
                    16,
                    19,
                    2,
                    4,
                    6,
                    19,
                    5,
                    5,
                    8,
                    19,
                    18,
                    1,
                    2,
                    15,
                    6,
                    0,
                    17,
                    8,
                    14,
                    13,
                ]
            )

            return np.concatenate(
                [np.array(targets)[:, None], coarse_labels[targets][:, None]], axis=1
            )

        train_dataset, val_dataset = copy.deepcopy(train_loader.dataset), copy.deepcopy(
            val_loader.dataset
        )
        test_dataset = copy.deepcopy(test_loader.dataset)

        train_dataset.dataset.targets = sparse2joint(train_dataset.dataset.targets)
        val_dataset.dataset.targets = sparse2joint(val_dataset.dataset.targets)
        test_dataset.targets = sparse2joint(test_dataset.targets)

        total_data = len(train_dataset)
        subset_for_coarse, _ = torch.utils.data.random_split(
            train_dataset, [10000, total_data - 10000]
        )

        coarse_dataloader = DataLoader(
            subset_for_coarse, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_with_side_dataloader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
        )
        test_with_side_dataloader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
    elif cfg.dataset == "emnist":

        def sparse2joint(targets):
            """
            Convert Pytorch EMNIST sparse targets to coarse targets.

            Usage:
                training_data = torchvision.datasets.EMNIST(path)
                training_data.targets = sparse2coarse(training_data.targets)

            Parameters
            ----------
            targets: torch.Tensor with shape (..., 1)
                The ground-truth labels.

            Returns
            -------
            targets: torch.Tensor (..., 2)
                The ground-truth labels followed by the coarser super-class labels.
            """
            coarse_labels = np.array(10 * [0] + 26 * [1] + 26 * [2])
            return np.concatenate(
                [np.array(targets)[:, None], coarse_labels[targets][:, None]], axis=1
            )

        train_dataset, val_dataset = copy.deepcopy(train_loader.dataset), copy.deepcopy(
            val_loader.dataset
        )
        test_dataset = copy.deepcopy(test_loader.dataset)

        train_dataset.dataset.targets = sparse2joint(train_dataset.dataset.targets)
        val_dataset.dataset.targets = sparse2joint(val_dataset.dataset.targets)
        test_dataset.targets = sparse2joint(test_dataset.targets)

        total_data = len(train_dataset)
        subset_for_coarse, _ = torch.utils.data.random_split(
            train_dataset, [10000, total_data - 10000]
        )

        coarse_dataloader = DataLoader(
            subset_for_coarse, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_with_side_dataloader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
        )
        test_with_side_dataloader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
    else:
        raise Exception()

    print()
    side_model = side_information_training(
        side_model, coarse_dataloader, val_with_side_dataloader, cfg
    )

    print()
    print("No side information")
    seed_everything(cfg.seed)
    res = evaluate(
        **setup_dict,
        alpha_test=cfg.alpha_test,
        objective=cfg.objective,
        conformal_methods=["thr", "thr-l", "aps", "raps"],
        size_loss_weight=cfg.size_loss_weight,
        coverage_loss_weight=cfg.coverage_loss_weight,
        cross_entropy_loss_weight=cfg.cross_entropy_loss_weight,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        target_size=cfg.target_size,
        relaxation=cfg.relaxation,
    )
    print_results(res)

    print()
    print("Side CP 10%")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=0.1,
        use_side_test=0.1,
        group_balanced=False,
    )
    res["train_acc"] = None
    print_results(res)

    print()
    print("Side CP 30%")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=0.3,
        use_side_test=0.3,
        group_balanced=False,
    )
    res["train_acc"] = None
    print_results(res)

    print()
    print("Side CP 100%")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=1,
        use_side_test=1,
        group_balanced=False,
    )
    res["train_acc"] = None
    print_results(res)

    print()
    print("Group CP")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=0.0,
        use_side_test=0.0,
        group_balanced=True,
    )
    res["train_acc"] = None
    print_results(res)

    print()
    print("Group Side CP 10%")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=0.1,
        use_side_test=0.1,
        group_balanced=True,
    )
    res["train_acc"] = None
    print_results(res)

    print()
    print("Group Side CP 30%")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=0.3,
        use_side_test=0.3,
        group_balanced=True,
    )
    res["train_acc"] = None
    print_results(res)

    print()
    print("Group Side CP 100%")
    seed_everything(cfg.seed)
    res = evaluate_side(
        side_model,
        val_with_side_dataloader,
        test_with_side_dataloader,
        alpha_test=cfg.alpha_test,
        device=setup_dict["device"],
        conformal_methods=conformal_methods,
        n_splits=cfg.evaluation_splits,
        n_sel=cfg.sel_examples,
        reg_vector=setup_dict["reg_vector"],
        use_side_cal=1,
        use_side_test=1,
        group_balanced=True,
    )
    res["train_acc"] = None
    print_results(res)


if __name__ == "__main__":
    main()
