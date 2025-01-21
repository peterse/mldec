from mldec.datasets import toy_problem_data
from mldec.models import initialize, train_model

def main(config):


    # Configure the dataset
    if config.get("dataset_module") == "toy_problem":
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
    model = initialize.initialize_model(config)

    # train the model
    train_model.train_model(model, dataset_module, config, dataset_config)

if __name__ == "__main__":

    n = 8
    input_dim = n - 1
    config = {
        "model": "ffnn",
        "input_dim": input_dim,
        "hidden_dim": 8,
        "output_dim": 8,
        "n_layers": 1,
        "max_epochs": 1000,
        "batch_size": 100,
        "learning_rate": 0.01,
        "patience": 10,
        "n": n,
        "mode": "train",
        "n_train": 1000,
        "lr": 0.01,
        "opt": "adam",
        "sgd_decay_rate": 0.5,
        "sgd_decay_patience": 10,
        "dataset_module": "toy_problem",
    }
    main(config)