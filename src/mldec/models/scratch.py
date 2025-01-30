import multiprocessing as mp
import numpy as np




def trainable(package):
    # Note: Function wrapping within `tune_hyperparameters_multiprocessing` is dangerous
    # because of serializability issues, e.g. pickling a function that is not defined at the module level
    (hyper_config, config, dataset_module, dataset_config) = package
    return initialize_and_train_model(hyper_config, config, dataset_module, dataset_config)

def main():
    # List of parameters for which f needs to be evaluated
    params_list = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    
    # Number of CPUs to use
    num_cpus = mp.cpu_count()  # Get the number of available CPUs
    print(f"Number of available CPUs: {num_cpus}")
    # You can adjust this number based on your requirement or limit
    num_cpus_to_use = 3  # For example, use 3 CPUs, adjust as needed

    # Create a multiprocessing pool with the desired number of CPUs


def tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, dataset_module, config, dataset_config):
    """
    hyper_config should contain a list for each hyperparameter in the search, e.g.
    hyper_config = {
        'lr': np.loguniform(1e-4, 1e-2),
        'hidden_dim': [8, 16, 32, 64],
        'n_layers': [1, 2, 3, 4],
}
    """
    num_cpus=hyper_settings.get("total_cpus")
    num_gpus=hyper_settings.get("total_gpus") 
    if num_gpus != 0:
        raise NotImplementedError("GPU support not yet implemented")

    # build the hyperparameter space by sampling (gridsearch not yet supported)
    package_list = []
    for _ in range(hyper_settings.get("num_samples")):
        hyperparameter_slice = {}
        for key, value in hyper_config.items():
            hyperparameter_slice[key] = np.random.choice(value)
        package = (hyperparameter_slice, config, dataset_module, dataset_config)
        package_list.append(package)



    with mp.Pool(processes=num_cpus) as pool:
        # Map f to the list of parameters
        results = pool.map(trainable, package_list)
        print(results)

    # using the hyper_config, build a pool