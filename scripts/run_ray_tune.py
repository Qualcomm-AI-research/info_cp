# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides functions for hyperparameter optimization with ray tune."""


import os
import pickle

import hydra
import numpy as np
import ray
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from ray import train, tune
from ray.train import CheckpointConfig, RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler

from info_cp.configs import TrainingConfig
from info_cp.train_utils import evaluate, setup, train_conformal
from info_cp.utils import print_tune_results

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
__MAX_CONCURRENCY__ = 16


def set_raytune_global_env() -> None:
    """Configure some Raytune's global variables."""
    # Disable Ray's automatic reports when running in cluster mode
    # see https://docs.ray.io/en/latest/cluster/usage-stats.html
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

    # Syncer are used in multi-node experiments to synchronize logs and checkpoint
    # between the head and worker nodes.
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"

    # Disable auto-logging
    # By default Ray always defines CVS, JSON and Tensorboard callbacks. Here we
    # disable this so we can manually select which callback to add.
    # see https://docs.ray.io/en/latest/tune/api/env.html#tune-env-vars
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # Immediately stop on SIGINT errors
    os.environ["TUNE_DISABLE_SIGINT_HANDLER"] = "1"

    # Limit concurrency by capping the maximum number of pending trials
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(__MAX_CONCURRENCY__)


def get_config_column_params() -> dict:
    """Create the search space.

    Returns
    -------
    config: dict
        Configuration parameters.
    parameter_columns: List
        List of metrics to show on terminal.
    """
    config = {
        "batch_size": tune.grid_search([1000, 500, 100]),
        "lr": tune.grid_search([0.05, 0.01, 0.005]),
        "temperature": tune.grid_search([0.01, 0.1, 0.5, 1.0]),
        "steepness": tune.grid_search([1.0, 10.0, 100.0]),
    }
    return config


def tune_fn(cfg: dict, hydra_conf: OmegaConf) -> None:
    """Run training.

    Parameters
    ----------
    cfg: dict
        Dictionary containing the search space for tune.
    hydra_conf: OmegaConf
        Configuration parameters defined via Hydra.
    """
    for k, v in cfg.items():
        hydra_conf[k] = v
    cfg = ConfigDict(hydra_conf)
    setup_dict = setup(cfg)

    for _ in range(cfg.n_epochs):
        train_conformal(
            **setup_dict,
            max_epochs=1,
            alpha_test=cfg.alpha_test,
            objective=cfg.objective,
            conformal_method_train=cfg.conformal_method_train,
            conformal_methods_test=cfg.conformal_methods_test,
            conformal_fraction=cfg.conformal_fraction,
            temperature=cfg.temperature,
            size_loss_weight=cfg.size_loss_weight,
            coverage_loss_weight=cfg.coverage_loss_weight,
            cross_entropy_loss_weight=cfg.cross_entropy_loss_weight,
            early_stopping_patience=cfg.early_stopping_patience,
            evaluation_splits=cfg.evaluation_splits,
            n_sel=cfg.sel_examples,
            target_size=cfg.target_size,
            relaxation=cfg.relaxation,
            verbose=False,
        )

        res = evaluate(
            **setup_dict,
            alpha_test=cfg.alpha_test,
            objective=cfg.objective,
            conformal_methods=[cfg.conformal_methods_test[0]],  # ["thr", "thr-l", "aps", "raps"],
            size_loss_weight=cfg.size_loss_weight,
            coverage_loss_weight=cfg.coverage_loss_weight,
            cross_entropy_loss_weight=cfg.cross_entropy_loss_weight,
            n_splits=cfg.evaluation_splits,
            n_sel=cfg.sel_examples,
            target_size=cfg.target_size,
            relaxation=cfg.relaxation,
        )

        if (
            torch.isnan(res[f"test_size_{cfg.conformal_methods_test[0]}"])
            or res[f"test_size_{cfg.conformal_methods_test[0]}"] < 0.5
        ):
            # Something off, discard trial
            res[f"test_size_{cfg.conformal_methods_test[0]}"] = torch.tensor(float("inf"))

        train.report(
            {"test_size": res[f"test_size_{cfg.conformal_methods_test[0]}"].item()},
        )


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=TrainingConfig, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def tune_hp(cfg: TrainingConfig) -> None:
    """Run hyperparameter optimization with ray tune."""
    ray.init(num_cpus=10, num_gpus=1)
    set_raytune_global_env()

    # Set dirs
    if cfg.output_dir == "":
        output_dir = os.getcwd()
        cfg.output_dir = output_dir
    original_working_dir = get_original_cwd()
    cfg.original_working_dir = original_working_dir

    scheduler = AsyncHyperBandScheduler(max_t=cfg.n_epochs, grace_period=cfg.grace_period)

    param_space = get_config_column_params()
    trainable = tune.with_resources(
        tune.with_parameters(tune_fn, hydra_conf=cfg),
        resources={"cpu": cfg.ray_cpu, "gpu": cfg.ray_gpu},
    )
    tune_config = tune.TuneConfig(
        metric="test_size",
        mode="min",
        scheduler=scheduler,
        num_samples=1,
    )
    callbacks = [
        tune.logger.JsonLoggerCallback(),
        tune.logger.CSVLoggerCallback(),
    ]

    run_config = RunConfig(
        storage_path=output_dir,
        name="tune_results",
        stop={"training_iteration": cfg.n_epochs},
        log_to_file=True,
        callbacks=callbacks,
        checkpoint_config=CheckpointConfig(
            checkpoint_at_end=False,
            checkpoint_frequency=0,
            num_to_keep=1,
        ),
    )

    exp_dir = os.path.join(output_dir, "tune_results")
    if tune.Tuner.can_restore(exp_dir):
        tuner = tune.Tuner.restore(exp_dir, trainable=trainable, resume_errored=True)
        print("Restoring Ray Tune run")
    else:
        print("Restarting optimization from scratch")
        # Run the search
        tuner = tune.Tuner(
            trainable=trainable,
            tune_config=tune_config,
            param_space=param_space,
            run_config=run_config,
        )

    if cfg.eval_only:
        final_results_file = os.path.join(output_dir, "tune_results.pkl")
        if os.path.exists(final_results_file):
            with open(final_results_file, "rb") as fp:
                results = pickle.load(fp)
        else:
            print("Run not finished yet. Collecting partial results")
            results = tuner.get_results()
    else:
        results = tuner.fit()
        with open(os.path.join(output_dir, "tune_results.pkl"), "wb") as fp:
            pickle.dump(results, fp)

    print_tune_results(results, file_name=None)
    print_tune_results(results, file_name=os.path.join(output_dir, "tune_results.txt"))


if __name__ == "__main__":
    tune_hp()
