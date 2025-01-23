# REQUIREMENTS: ray[tune], 
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.train import RunConfig
from ray.tune.schedulers import FIFOScheduler

import psutil

def main():
    num_cpus = 4
    ray.init(num_cpus=num_cpus, _temp_dir=None)
    
    trainable = lambda x: print(psutil.Process().cpu_num())

    # control how much paralelism to invoke
    tune_config = tune.TuneConfig(
        num_samples=4,
        trial_dirname_creator=lambda x: "scratchwork",
        max_concurrent_trials=4,
    )
    config = {"a": tune.uniform(0, 1)}

    # Uncomment this to enable distributed execution

    tuner = tune.Tuner(
        trainable,
        param_space=config,
        tune_config=tune_config,
    )
    results = tuner.fit()
