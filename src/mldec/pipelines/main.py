from mldec.datasets import toy_problem_data
from mldec import models

def main():

    config = {
        "model": "ffnn",
        "input_dim": 8,
        "hidden_dim": 8,
        "output_dim": 8,
        "n_layers": 1,
        "max_epochs": 1000,
        "batch_size": 100,
        "learning_rate": 0.01,
        "patience": 10,
        "n": 8,
        "mode": "train",
        "n_train": 1000,
        "lr": 0.01,
        "opt": "adam",
        "sgd_decay_rate": 0.5,
        "sgd_decay_patience": 10,
        "dataset_module": "toy_problem",
    }

    # Configure the dataset
    if config.get("datset_module") == "toy_problem":
        dataset_module = toy_problem_data
        n = 8
        dataset_config = {
            'p1': 0.1,
            'p2': 0.07,
            'pcm': toy_problem_data.repetition_pcm(n),
        }
    else:
        raise ValueError("Unknown dataset module")
    
    # make the model
    model = models.initialize.initialize_model(config)

    # train the model
    models.train_model(model, dataset_module, config, dataset_config)