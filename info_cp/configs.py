# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides a dataclass with the configuration parameters used for training and testing."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """Config class used for configuring experiments with Hydra."""

    n_epochs: int = 50
    lr: float = 0.001  # Learning rate
    optimizer: str = "sgd"  # Type of optimizer
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0005
    scheduler: str = "step"  # Type of scheduler
    gamma: float = 0.1  # Multiplicative factor of learning rate decay used by the scheduler
    early_stopping_patience: int = 50
    job_name: str = "empty"  # Name used in the logs
    # conformal params
    objective: str = "ce"  # Training objective: "ce", "conftr", "conftr_class", "simple_fano",
    # "model_fano" or "dpi"
    conformal_method_train: str = "thr-lp"  # We train using thresholding on log-probabilities
    conformal_methods_test: List[str] = field(
        default_factory=lambda: list(["aps", "raps", "thr", "thr-l"])
    )
    alpha: float = 0.01  # alpha value used during training
    alpha_test: float = 0.01  # alpha value used at test time
    k_reg: int = 1  # Number of prediction set elements that are not penalized
    lam_reg: float = 0.01  # Penalization term for elements added to sets already larger than k_reg
    conformal_fraction: float = 0.5  # Percentage of batch data to be used for calibration
    temperature: float = 1.0  # Thresholding smoothing factor
    size_loss_weight: float = 0.05  # Weight of the size loss in conftr
    coverage_loss_weight: float = 0.0  # Weight of the coverage loss in conftr
    cross_entropy_loss_weight: float = 0.0  # Weight of the cross-entropy loss in conftr
    target_size: int = 1
    relaxation: str = "smoothing"
    device: str = "cuda"
    # data params
    dataset: str = "cifar10"
    datapath: str = "data"
    batch_size: int = 500
    val_examples: int = 5000  # How many validation examples to take from the training data
    sel_examples: int = 1000  # How many of the validation examples are used for model selection
    # model
    pretrained: bool = False
    # sorting
    sorting_method: str = "diffsort"  # only diffsort is supported
    steepness: float = 10.0  # diffsort smoothing factor
    regularization_strength: float = 1.0  # torchsort smoothing factor
    # reproducibility
    seed: int = 0
    evaluation_splits: int = 10
    eval_only: bool = False
    # path
    output_dir: str = ""
    original_working_dir: str = ""
    # hypersearch, ray tune parameters
    grace_period: int = 10
    ray_cpu: float = 1.0
    ray_gpu: float = 0.16
    hypersearch: bool = False
    # Side information parameters
    n_epochs_side: int = 20
    lr_side: float = 0.01
    pretrained_path_for_side: str = ""
