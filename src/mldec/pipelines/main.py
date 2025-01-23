import os
from mldec.datasets import toy_problem_data
from mldec.models import initialize, train_model

from ray import tune

abs_path = os.path.dirname(os.path.abspath(__file__))
tune_directory = os.path.join(abs_path, "ray_results")
tune_path = os.path.join(tune_directory, "tune_results.csv")

def main(config):


    # Configure the dataset.
    # Why pass a module? The main observation is that actually
    # training on real data is comparatively slow; we instead
    # reweight a loss function according to a sample from the 
    # underlying data distribution. This virtual sampling is done
    # just-in-time.
    if config.get("dataset_module") == "toy_problem":
        dataset_module = toy_problem_data
        n = config['n']
        dataset_config = {
            'p1': 0.1,
            'p2': 0.07,
            'pcm': toy_problem_data.repetition_pcm(n),
        }
    else:
        raise ValueError("Unknown dataset module")
    

    if config.get("mode") == "tune":
        # these are the parameterized hyperparameters we want to tune over
        hyper_config = {
            'lr': tune.loguniform(1e-4, 1e-2),
            'hidden_dim': tune.choice([16, 32, 64]),
            'n_layers': tune.choice([1, 2, 3]),
        }
        # these specify how tune will work
        
        hyper_settings = {
            "total_cpus": 2,
            "total_gpus": 0,
            "cpus_per_worker": 1, #i.e. cpus per trial
            "gpus_per_worker": 0,
            "max_concurrent_trials": 1,
            "max_epochs": 100, # this is the max epochs any trial is allowed to run
            "num_samples": 2, # this is equal to total trials if no grid search
            "tune_directory": tune_directory,
            "tune_path": tune_path,
        }
        train_model.tune_hyperparameters(hyper_config, hyper_settings, dataset_module, config, dataset_config)
    else:
        model = initialize.initialize_model(config)
        train_model.train_model(model, dataset_module, config, dataset_config)

if __name__ == "__main__":
    # mode options: train, verify, tune
    # mode = "verify"
    mode = "tune"
    n = 8
    input_dim = n - 1
    config = {
        "model": "ffnn",
        "input_dim": input_dim,
        "hidden_dim": 16,
        "output_dim": n,
        "n_layers": 3,
        "max_epochs": 10000,
        "batch_size": 500,
        "learning_rate": 0.003,
        "patience": 2000,
        "n": n,
        "mode": mode,
        "n_train": 1000,
        "lr": 0.01, # scales up with batch size
        "opt": "adam",
        "sgd_decay_rate": 0.5,
        "sgd_decay_patience": 100,
        "dataset_module": "toy_problem",
    }
    main(config)