# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides the main function that runs conformal training."""


import json
import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from info_cp.configs import TrainingConfig
from info_cp.train_utils import alpha_sweep, evaluate, setup, train_conformal
from info_cp.utils import print_results, seed_everything

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=TrainingConfig, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: TrainingConfig) -> None:
    """Run conformal training according to a given Hydra config file."""
    # Set dirs
    output_dir = os.getcwd()
    original_working_dir = get_original_cwd()
    cfg.output_dir = output_dir
    cfg.original_working_dir = original_working_dir

    # Set the experiment up
    setup_dict = setup(cfg)

    if not cfg.eval_only:
        train_conformal(
            **setup_dict,
            max_epochs=cfg.n_epochs,
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
            verbose=True,
        )
        torch.save(setup_dict["model"].state_dict(), setup_dict["model_path"])

    # Evaluate final results for all conformal prediction methods
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
    print_results(res, os.path.join(output_dir, "final_results.txt"))
    print_results(res)

    # Compute prediction set size on test data using different values for alpha
    if setup_dict["logger"] is not None:
        alpha_sweep(res, ["thr", "thr-l", "aps", "raps"], setup_dict["logger"])

    if cfg.hypersearch:
        res_dir = os.path.join(
            original_working_dir,
            "outputs",
            cfg.dataset,
            "seed" + str(cfg.seed),
            cfg.job_name,
            "hypersearch",
        )
        os.makedirs(res_dir, exist_ok=True)
        filename = (
            f"batch_size_{cfg.batch_size}_lr_{cfg.lr}"
            f"_temp_{cfg.temperature}_steep_{cfg.steepness}.json"
        )
        res_dict = {}
        for method in cfg.conformal_methods_test:
            res_dict[method] = res["test_size_" + method].item()
        print("Writing to ", os.path.join(res_dir, filename))
        with open(os.path.join(res_dir, filename), "w", encoding="utf8") as f:
            json.dump(res_dict, f)


if __name__ == "__main__":
    main()
