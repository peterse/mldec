import os
from mldec.datasets import toy_problem_data
from mldec.models import initialize, train_model, tune_model
import torch
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))


def make_tune_directory(config):
    tune_directory = os.path.join(abs_path, "tune_results")
    model = config.get("model")
    dataset_module = config.get("dataset_module")
    only_good_examples = config.get("only_good_examples")
    model_subdir = f"{model}_{dataset_module}_good{only_good_examples}"
    tune_directory = os.path.join(tune_directory, model_subdir)
    if not os.path.exists(tune_directory):
        os.makedirs(tune_directory)
    return tune_directory

def main(config):


    # Configure the dataset.
    # Why pass a module? The main observation is that actually
    # training on real data is comparatively slow; we instead
    # reweight a loss function according to a sample from the 
    # underlying data distribution. This virtual sampling is done
    # just-in-time.
    dataset_module = config.get("dataset_module")
    if dataset_module == "toy_problem":
        n = config['n']
        dataset_config = {
            'p1': 0.1,
            'p2': 0.07,
            'pcm': toy_problem_data.repetition_pcm(n),
            "sos_eos": config.get("sos_eos", None),
        }
    else:
        raise ValueError("Unknown dataset module")
    
        
    if config.get("mode") == "tune":

        tune_directory = make_tune_directory(config)
        # tune_path = os.path.join(tune_directory, "tune_results.csv")
        # these are the parameterized hyperparameters we want to tune over
        # They vary by model, so be aware!
        hyper_config = {
            'lr': np.logspace(-4,-2, num=10, base=10.0),
            'hidden_dim': np.array([8, 16, 32, 64]),
            'n_layers': np.array([1, 2, 3, 4]),
        }
        
        # these specify how tune will work
        hyper_settings = {
            "total_cpus": 2,
            "total_gpus": 0,
            "num_samples": 2, # this is equal to total trials if no grid search
            "tune_directory": tune_directory,
        }
        tune_model.tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, dataset_module, config, dataset_config)
    else:
        model_wrapper = initialize.initialize_model(config)
        train_model.train_model(model_wrapper, dataset_module, config, dataset_config)

if __name__ == "__main__":
    only_good_examples = True
    mode = "tune" # options: train, tune
    n = 8
    input_dim = n - 1

    # Some notes:
    # scale up lr with batchsize in general.
    # patience = early stopping patience, triggered on no improvement in val_acc for `patience` epochs (with checking every X epochs)\
    # n_train = virtual training samples, i.e. noise in a histogram of the error distribution
    # !OVERWRITE indiicates a hyperparam that may be overwritten by raytune `hyper_config` or raytune internals
    # only_good_examples = uniform distribution over good examples
    # SERIALIZABILITY: All of the config options, hyper options, dataset_config options must be serializable (json)
    config = {
        "device": "cpu", # !OVERWRITE
        # Dataset config
        "n": n,
        "only_good_examples": only_good_examples, 
        "n_train": 1000,
        "dataset_module": "toy_problem",
        # Training config: 
        "max_epochs": 300,
        "batch_size": 500,
        "patience": 2000,  
        "lr": 0.003, # !OVERWRITE
        "opt": "adam",
        "mode": mode,
        # fixed model config
        "input_dim": input_dim,
        "output_dim": n,
        "dropout": 0.05, # !OVERWRITE
    }

    MODEL_CHOICE = "cnn"
    if MODEL_CHOICE == "ffnn":
        model_config = {
            "model": "ffnn",
            "hidden_dim": 16, # !OVERWRITE
            "n_layers": 3, # !OVERWRITE
            "dropout": 0, # !OVERWRITE
        }
    elif MODEL_CHOICE == "cnn":
        model_config = {
            "model": "cnn",
            "conv_channels": 4, # !OVERWRITE
            "n_layers": 3, # !OVERWRITE
        }
    elif MODEL_CHOICE == "encdec":
        config["sos_eos"] = (0, 0)
        model_config = {
            "model": "encdec",
            "d_model": 16, # !OVERWRITE
            "nhead": 4, # !OVERWRITE
            "num_encoder_layers": 2, # !OVERWRITE
            "num_decoder_layers": 2, # !OVERWRITE
            "dim_feedforward": 8, # !OVERWRITE
        }

    config.update(model_config)
    main(config)