# requirements grpcio-1.67.1 ray-2.41.0, protobuf 3.20
import os
import time
os.environ["OMP_NUM_THREADS"] = "1" # also set via export on command line
import ray
num_cpus = 10
ray.init(
        num_cpus=num_cpus, 
         _temp_dir=None, 
         include_dashboard=False,
    )



# import time

# filepath = os.path.abspath(__file__)
# tmp_dir = os.path.join(os.path.dirname(filepath), "temp")
#     runtime_env={
#         "worker_env": {
#             "RAY_WORKER_MIN_PORT": "6000",
#             "RAY_WORKER_MAX_PORT": "19999"
#         }
# }
#          object_store_memory=BYTES_MEMORY, 

BYTES_MEMORY = 80 * 1e6 # 5 MB

print("finished init")

from ray import tune
from ray.tune import CLIReporter
from ray.train import RunConfig
from ray.tune.schedulers import FIFOScheduler
import psutil
import time

# object_store_memory= The amount of memory (in bytes) to start the object store with. 
# By default, this is 30% (ray_constants.DEFAULT_OBJECT_STORE_MEMORY_PROPORTION) of 
# available system memory capped by the shm size and 200G; This was 600MB on the cluster...
# (ray_constants.DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES) but can be set higher. 


print("...........tuner.fit()")

trainable = lambda x: print("hello from cpu:", psutil.Process().cpu_num())
# TRYME:
# tune.with_resources(train_candidate, {"cpu": 1, "gpu": 1, "memory": <memory_limit_in_bytes>})

# control how much paralelism to invoke
tune_config = tune.TuneConfig(
    num_samples=40,
    trial_dirname_creator=lambda x: "scratchwork",
    max_concurrent_trials=20,
)
config = {"a": tune.uniform(0, 1)}

# Uncomment this to enable distributed execution

tuner = tune.Tuner(
    trainable,
    param_space=config,
    tune_config=tune_config,
)
results = tuner.fit()
