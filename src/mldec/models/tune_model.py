import torch
import numpy as np
import os
import datetime
import json
import time
import multiprocessing as mp

from mldec.models import initialize, train_model
from mldec.pipelines import loggingx

def initialize_and_train_model(hyper_config, config, dataset_module, dataset_config, manager=None):
    device = torch.device("cpu")
    manager.log_print(f"initializing multiprocessing on: {device}")
    manager.log_print(f"Process ID: {os.getpid()}, Process Name: {mp.current_process().name}")
    config["device"] = "cpu"
    # merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
    for k, v in hyper_config.items():
        config[k] = v
    model = initialize.initialize_model(config)
    train_model.train_model(model, dataset_module, config, dataset_config, manager=manager)


class ThreadManager:
    """
    Each thread has its own manager to handle file I/O operations.

    We will write to `tune_directory`, which is a directory that is unique to this thread
    (timestamp for uniqueness). This directory will be populated with:
    - a CSV file for the results of each epoch: tune_results_{thread_id}.csv
    - a JSON file for the model config: config_{thread_id}.json
    - a JSON file for the hyperparameters: hyper_config_{thread_id}.json
    """
    def __init__(self, thread_id, tune_path, logger_name):
        self.thread_id = thread_id
        self.tune_path = tune_path
        self.tune_results_path = os.path.join(self.tune_path, f"tune_results.csv")
        self.log_path = os.path.join(self.tune_path, f"log.txt")

        # initialize logging
        log_path = os.path.join(tune_path, "log.txt") # where the train_model will log
        logger_name = f"train_model_{thread_id}"
        self.logger = loggingx.init_logger(logger_name, log_path)

        self.logger.info(f"Thread {thread_id} writing to {self.tune_path}")
        with open(self.tune_results_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    def report(self, epoch_results):
        with open(self.tune_results_path, "a") as f:
            f.write(f"{epoch_results['epoch']},{epoch_results['train_loss']},{epoch_results['train_acc']},{epoch_results['val_loss']},{epoch_results['val_acc']}\n")

    def save_configs(self, config, hyper_config):
        with open(os.path.join(self.tune_path, f"config_{self.thread_id}.json"), "w") as f:
            f.write(json.dumps(config))
        with open(os.path.join(self.tune_path, f"hyper_config_{self.thread_id}.json"), "w") as f:
            f.write(json.dumps(hyper_config))

    def log_print(self, message):
        self.logger.info(message)

def trainable(package):
    """This function is now inside its own thread"""
    # Note: Function wrapping within `tune_hyperparameters_multiprocessing` is dangerous
    # because of serializability issues, e.g. pickling a function that is not defined at the module level
    (hyper_config, config, dataset_module, dataset_config, local_vars) = package

    # Write a file to the tune directory with filename and id
    # This is to allow for easy tracking of the results
    tune_path = local_vars.get("tune_path")
    thread_id = local_vars.get("thread_id")
    manager = ThreadManager(thread_id, tune_path, local_vars.get("logger_name"))
    results = initialize_and_train_model(hyper_config, config, dataset_module, dataset_config, manager=manager)
    # wrap-up operations: save config as json, save hyper_config as json
    manager.save_configs(config, hyper_config)


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

    # Number of CPUs to use
    tot_cpus = mp.cpu_count()  # Get the number of available CPUs
    print(f"Number of available CPUs: {tot_cpus}")
    print(f"Number of CPUs to use: {num_cpus}")

    # build the hyperparameter space by sampling (gridsearch not yet supported)
    hyper_list = []
    for _ in range(hyper_settings.get("num_samples")):
        hyperparameter_slice = {}
        for key, value in hyper_config.items():
            hyperparameter_slice[key] = np.random.choice(value).item() # keep serializable

        hyper_list.append(hyperparameter_slice)

    # Assemble local variables to be called from within a thread
    tune_directory = hyper_settings.get("tune_directory") # where all the tuning results live
    package_list = []
    for hyperparameter_slice in hyper_list:
        # generate a datetime stample YYYY-MM-DD-HH-MM-SS-MS
        thread_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        tune_path = os.path.join(tune_directory, f"thread_{thread_id}") # where this thread's results live
        os.makedirs(tune_path)

        local_vars = {"thread_id": thread_id, "tune_path": tune_path}
        package = (hyperparameter_slice, config, dataset_module, dataset_config, local_vars)
        package_list.append(package)
        time.sleep(0.1) # to ensure unique timestamps

    with mp.Pool(processes=num_cpus) as pool:
        # Map f to the list of parameters
        results = pool.map(trainable, package_list)
        print(results)

    # using the hyper_config, build a pool


# def tune_hyperparameters(hyper_config, hyper_settings, dataset_module, config, dataset_config):


#     dataset = config.get("dataset_module")
#     only_good_examples = config.get("only_good_examples")
#     prefix = f"{dataset}_"
#     if only_good_examples:
#         prefix += "only_good_examples_"
#     prefix += f"{config.get('model')}_"
#     # create a MMDDHHMMSS timestamp
#     suffix = datetime.datetime.now().strftime("%m%d%H%M%S")

#     def trial_dirname_creator(trial):
#         return f"{prefix}{trial.trial_id}"

#     ray.init(
#         include_dashboard=False, 
#             num_cpus=hyper_settings.get("total_cpus"), 
#             num_gpus=hyper_settings.get("total_gpus"), 
#             _temp_dir=None, 
#             ignore_reinit_error=True)
    
#     scheduler = FIFOScheduler()

#     reporter = CLIReporter(
#         metric_columns=["loss", "training_iteration", "mean_accuracy"],
#         print_intermediate_tables=False,
#         )

#     tune_config = tune.TuneConfig(
#         num_samples=hyper_settings.get("num_samples"),
#         scheduler=scheduler,
#         trial_dirname_creator=trial_dirname_creator,
#         max_concurrent_trials=hyper_settings.get("max_concurrent_trials"),
#         )
        
#     run_config = RunConfig(
#         progress_reporter=reporter,
#         storage_path=hyper_settings.get("tune_directory"),
#         stop={"epoch": hyper_config.get("max_epochs")}, # because both the trainable and raytune need to see max_epochs.
#     )

#     resources = tune.with_resources(
#                 tune.with_parameters(
#                     initialize_and_train_model, 
#                     config=config,
#                     dataset_module=dataset_module,
#                     dataset_config=dataset_config,
#                     ),
#                 resources={"cpu": hyper_settings.get("cpus_per_worker"), "gpu": hyper_settings.get("gpus_per_worker")}
#     )

#     tuner = tune.Tuner(
#         resources,
#         param_space=hyper_config,
#         tune_config=tune_config,
#         run_config=run_config,
#     )
#     result = tuner.fit()
#     df = result.get_dataframe()
#     dest = os.path.join(hyper_settings.get("tune_directory"), f"{prefix}{suffix}_tune.csv")
#     with open(dest, 'a') as f:
#         f.write(df.to_string(header=True, index=False))
        
#     return result

