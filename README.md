# An Information Theoretic Perspective on Conformal Prediction
Alvaro H.C. Correia, Fabio Valerio Massoli, Christos Louizos and Arash Behboodi. "An Information Theoretic Perspective on Conformal Prediction." [[Arxiv]](https://arxiv.org/abs/2405.02140)

Qualcomm AI Research, Qualcomm Technologies Netherlands B.V. (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.).

## Abstract

Conformal Prediction (CP) is a distribution-free uncertainty estimation framework that constructs prediction sets guaranteed to contain the true answer with a user-specified probability. Intuitively, the size of the prediction set encodes a general notion of uncertainty, with larger sets associated with higher degrees of uncertainty. In this work, we leverage information theory to connect conformal prediction to other notions of uncertainty. More precisely, we prove three different ways to upper bound the intrinsic uncertainty, as described by the conditional entropy of the target variable given the inputs, by combining CP with information theoretical inequalities. Moreover, we demonstrate two direct and useful applications of such connection between conformal prediction and information theory: (i) more principled and effective conformal training objectives that generalize previous approaches and enable end-to-end training of machine learning models from scratch, and (ii) a natural mechanism to incorporate side information into conformal prediction. We empirically validate both applications in centralized and federated learning settings, showing our theoretical results translate to lower inefficiency (average prediction set size) for popular CP methods.

## Getting started

### Environment Setup
First, let's define the path where we want the repository to be downloaded.
```
REPO_PATH=<path/to/repo>
```
Now we can clone the repository.
```
git clone git@github.com:Qualcomm-AI-research/info_cp.git $REPO_PATH
cd $REPO_PATH
```
Next, create a virtual environment.
```
python3 -m venv env
source env/bin/activate
```
Make sure to have Python ≥3.10 (tested with Python 3.10.12) and ensure the latest version of pip (tested with 24.2).
```
pip install --upgrade --no-deps pip
```
Next, install PyTorch 2.3.1 with the appropriate CUDA version (tested with CUDA 11.7).
```
python -m pip install torch==2.3.1 torchvision==0.18.1
```  
Finally, install the remaining dependencies using pip.
 ```
pip install -r requirements.txt
```
To run the code as indicated below, the project root directory needs to be added to your PYTHONPATH.
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

### Repository Structure

```
|- info_cp/
    |- configs.py         (data class with all configuration parameters.)
    |- conformal_utils.py (all utils related to conformal prediction, including our upper bounds to H(Y|X).)
    |- data_utils.py      (utilities for downloading and preprocessing data.)
    |- train_utils.py     (utilities for setting up and running the experiments.)
    |- utils.py           (common utilities.)
|- scripts/
    |- run_conformal_training.py (run conformal training experiments.)
    |- run_ray_tune.py           (run hyperparameter search with ray tune.)
    |- side_information.py       (run side information experiments.)
    |- conf/
        |- experiment/ (directory with all config files organized by dataset/confomal_method/training_objective.)
            |- <dataset>/
                |- <conformal_method>/
                    |- <training_objective>.yaml
```

## Running experiments

### Conformal Training
The main run file to reproduce the conformal training experiments is `scripts/run_conformal_training.py`. We use [Hydra](https://hydra.cc/) to configure experiments, so you can retrain our models as follows.
```bash
python scripts/run_conformal_training.py +experiment=<config_file>
```
where <config_file> is a yaml file defining the experiment configuration. The experiments in the paper are configured via the config files in the `scripts/conf/experiment` folder, which are named as `<dataset>/<conformal_method>/<training_objective>.yaml`. For instance, `mnist/aps/dpi` is the config used to train a model on MNIST using the DPI bound and the best hyperparameters for APS.
```bash
python scripts/run_conformal_training.py +experiment=mnist/aps/dpi
```
After training, this script also evaluates the trained model on the test data and prints the results to stdout. The model as well as the test data results are saved in the folder `outputs/<dataset>/seed<seed>/<training_objective>/<conformal_method>` as `model.pt` and `final_results.txt`, respectively. It is also possible to evaluate the trained model on the test data again by setting `eval_only=True` in the command line. 

```bash
python scripts/run_conformal_training.py +experiment=mnist/aps/dpi ++eval_only=True
```
Note that this overwrites `final_results.txt`, but the results will remain the same unless config parameters are changed (see below).

> Note that the code is not fully deterministic, so training results might vary slightly with differences in hardware/software stack.

#### Different configurations

To experiment with different configurations, you can either create a new yaml file and use it on the command line as above, or you can change specific variables with [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). As an example, if you want to keep the same configuration of the MNIST experiments with APS and DPI bound but change the batch size to 100, you can do so as follows

```bash
python scripts/run_conformal_training.py +experiment=mnist/aps/dpi ++batch_size=100
```

See `info_cp/configs.py` for an overview of the different configuration variables. Since changing the config variables also changes the final trained model, a new output folder, named after the extra config variable passed in the command line, is created inside of `outputs/<dataset>/seed<seed>/<training_objective>/<conformal_method>`. In the example, above the folder would be named `++batch_size=100`. The exception to this rule is config variables that only change test time behavior, namely the target miscoverage at test time `alpha_test` and [RAPS](https://arxiv.org/abs/2009.14193) hyperparameters `lam_reg` and `k_reg`. These configuration parameters, as well as `eval_only`, do not change the folder name.

For instance, the following command only evaluates a previously trained model with a target miscoverage rate `alpha_test=0.1`.

```bash
python scripts/run_conformal_training.py +experiment=mnist/aps/dpi  # creates a new folder, only needed if you haven't run that already
python scripts/run_conformal_training.py +experiment=mnist/aps/dpi ++alpha_test=0.1 ++eval_only=True  # does not create a new folder, overwrites the results in final_results.txt
```
This will not create a new folder and thus will overwrite the results in `final_results.txt`.

### Side Information
The experiments with side information can be run with `scripts/side_information.py`. This script relies on the same config structure but assumes a model with the corresponding config has already been trained using `scripts/run_conformal_training.py`. The experiments in the paper can be rerun with

**CIFAR100**
```bash
python scripts/run_conformal_training.py +experiment=cifar100/thr/ce  # only needed if you haven't run that already 
python scripts/side_information.py +experiment=cifar100/thr/ce
```
**EMNIST**
```bash
python scripts/run_conformal_training.py +experiment=emnist/thr/ce  # only needed if you haven't run that already
python scripts/side_information.py +experiment=emnist/thr/ce
```

In the paper, we considered classifiers trained with cross-entropy for the side information experiments, and that is why we pass config files `/thr/ce` as above. Nonetheless, any other config file should work provided the corresponding model has already been trained.

### Hyperparameter search
We used ray tune to search for the best values for the learning rate, batch size, temperature and steepness hyperparameters. This can be done via `scripts/run_ray_tune.py` using the same config structure from before. For example, to find the best hyperparameters for MNIST with APS and the DPI bound, we ran

```bash
python scripts/run_ray_tune.py +experiment=mnist/aps/dpi
```
This will do a grid search as described in the paper and output the best set of hyperparameters for a specific dataset, conformal method and training objective combination.

## Citation

If you find our work useful, please cite
```
@article{correia2024information,
  title={An Information Theoretic Perspective on Conformal Prediction},
  author={Correia, Alvaro HC and Massoli, Fabio Valerio and Louizos, Christos and Behboodi, Arash},
  journal={Advances in neural information processing systems},
  volume={37},
  year={2024}
}
```
