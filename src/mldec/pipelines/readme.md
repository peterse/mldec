#### Instructions for running the pipeline(s)

There are two modes: 'tune' and 'train'. The 'train' mode trains a single model, single-threaded, and is good for 
initial debugging and discovery. The 'tune' mode dispatches 'train' to many different places with stochastically
sampled hyperparameters and dataset config parameters.

### Train mode
In 'train' mode, a `config` dictionary will be populated with all of the details of the trauning run and the model hyperparameters, statically. Every parameter must be specified, there are no default parameters. This interface is intentionally frustrating to use. I do not want to run a model that I have not explicitly specified the details of.

To switch from tune mode to train mode, you need to explicitly comment out hyperparams that had been specified by the tune yaml before. To help you get started, you should comment out anything with `#!OVERWRITE` comment on that line.


### Tune mode

The 'tune' mode is even more intentionally frustrating to use. A yaml file in `hyper_config/`is read in to populate each different instance of `config` that is sent to a thread doing a training run. If there are any duplicated parameters across both dictionaries, the run will fail. If there are missing parameters in the hyper config that are not present in `config`, the run will fail. If the yaml file header contains the wrong model name or has a single hyperparameter that doesn't make sense for the model specified in `config`, the run will fail. I do not want to burn 300 cpu hours with a massive hyper run if I have not explicitly specified the set of model instances that might be trained and I should be able to reference the log files to determine which set of model instances I was sampling from without any possible ambiguity between `config` and `hyper_config`.

To help you be slightly less frustrated, here are tips for running in tune mode:
 1. All modifications are done either to `config` and/or `model_config` in `main.py`, or to the `.yaml` file in `mldec/hyper_config/`
 1a. Since you will fail the first time, set `epochs` to be small in `config`, and `num_samples` to be small in the `yaml` file.
 2. in `config`, set model name and the hyper_config to point to the correct hyper_config for that model name
 3. in the `config` dictionary and `model_config` dictionaries, comment out anything that might conflict with the `hyper_config`. The possible issues are highlighted with `# !OVERWRITE` comment
 4. Try to run tune and correct the errors that inevitably pop up in the validation scripts