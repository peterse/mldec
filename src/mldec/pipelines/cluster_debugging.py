# PYTHON PACKAGE REQUIREMENTS: ray[tune], 
import ray
from ray import tune

import psutil

def main():
    # `num_cpus` is the number of CPUs that the program tries to allocate in advance.
    # This can limit the resources available, e.g. when using a SLURM partition
    num_cpus = 20
    ray.init(num_cpus=num_cpus, _temp_dir=None)
    
    trainable = lambda x: print("hello from cpu:", psutil.Process().cpu_num())

    # `max_concurrent_trials` is how many of `num_cpus` the program will try to use
    tune_config = tune.TuneConfig(
        num_samples=40,
        max_concurrent_trials=40,
        trial_dirname_creator=lambda x: "scratchwork",
    )
    config = {"a": tune.uniform(0, 1)}

    tuner = tune.Tuner(
        trainable,
        param_space=config,
        tune_config=tune_config,
    )
    tuner.fit()

if __name__ == "__main__":
    main()