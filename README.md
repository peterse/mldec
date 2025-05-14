# Codebase for "Example importance for data-driven decoding"

#### Installation

You will need to install the `mldec` python package locally. From this directory, call `python -m pip install -e .`

#### Running experiments

The two main experiment scripts are:
 - `mldec/pipelines/main` (repetition code and surface code DDD with FNN, CNN, transformer) 
 - `mldec/pipelines/reps_main` (Detector DDD with GNN).

Each file has a `mode` ("train" or "tune"), `dataset_module` (to determine the type of experiment), `MODEL` to determine the architecture, and a `config` dictionary that specifies fixed run parameters. If in train mode, you can specify all `config` entries to run a specific experiment with the above options. In tune mode, you can specify sets of hyperparameters and dataset parameters to use for large parallel-processing runs using a yaml configuration file in `mldec/hyper_config`. Further instructions are provided in `mldec/pipelines/README.md`.

#### Generating figures

Notebooks for generating all figures are in `src/mldec/analysis`.