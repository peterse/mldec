import torch
import numpy as np
import os
import datetime
import json
import time
import multiprocessing as mp
import pandas as pd

from mldec.models import initialize, train_model
from mldec.pipelines import loggingx


def make_tune_results_path(path):
	return os.path.join(path, "tune_results.csv")

def hyper_config_path(path):
	return os.path.join(path, f"hyper_config.json")

def get_header():
	return "epoch,train_loss,train_acc,val_loss,val_acc"


class ThreadManager:
	"""
	Each thread has its own manager to handle file I/O operations.

	We will write to `tune_directory`, which is a directory that is unique to this thread
	(timestamp for uniqueness). This directory will be populated with:
	- a CSV file for the results of each epoch: tune_results_{thread_id}.csv
	- a JSON file for the model config: config_{thread_id}.json
	- a JSON file for the hyperparameters: hyper_config_{thread_id}.json
	"""
	def __init__(self, thread_info):
		self.thread_id = thread_info.get("thread_id")
		self.tune_path = thread_info.get("tune_path")  # tune_results/{model}_{dataset}/run_{timestamp}/threads/job_<thread_id>/
		self.logger_name = thread_info.get("logger_name")
		self.tune_results_path = make_tune_results_path(self.tune_path)
		self.log_path = os.path.join(self.tune_path, f"log.txt")

		# initialize logging
		log_path = os.path.join(self.tune_path, "log.txt") # where the train_model will log
		logger_name = f"train_model_{self.thread_id}"
		self.logger = loggingx.init_logger(logger_name, log_path)

		self.logger.info(f"Thread {self.thread_id} writing to {self.tune_path}")
		with open(self.tune_results_path, "w") as f:
			f.write(get_header() + "\n")

	def report(self, epoch_results):
		with open(self.tune_results_path, "a") as f:
			f.write(f"{epoch_results['epoch']},{epoch_results['train_loss']},{epoch_results['train_acc']},{epoch_results['val_loss']},{epoch_results['val_acc']}\n")

	def save_configs(self, config, hyper_config):
		# with open(os.path.join(self.tune_path, f"config_{self.thread_id}.json"), "w") as f:
		# 	f.write(json.dumps(config))
		with open(hyper_config_path(self.tune_path), "w") as f:
			f.write(json.dumps(hyper_config))

	def log_print(self, message):
		self.logger.info(message)


def make_tune_directory(config, abs_path):
	#new
	"""Build top-level tuning directories: tune_results/{model}_{dataset_module}/run_<run_id>"""
	tune_directory = os.path.join(abs_path, "tune_results")
	model_name = config.get("model")
	dataset_module = config.get("dataset_module")
	model_subdir = f"{model_name}_{dataset_module}"
	if config.get("only_good_examples"):
		model_subdir += "_only_good_examples"
	tune_directory = os.path.join(tune_directory, model_subdir)

	# stamp this run with YYYY-MM-DD-HH-MM-SS-MS
	run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
	run_directory = os.path.join("run_" + run_id) # makes tune_results/{model}_{dataset}/run_{timestamp}
	tune_directory = os.path.join(tune_directory, run_directory)
	if not os.path.exists(tune_directory):
		os.makedirs(tune_directory)
	return tune_directory


def trainable(package):
	"""This function is now inside its own thread"""
	# Note: Function wrapping within `tune_hyperparameters_multiprocessing` is dangerous
	# because of serializability issues, e.g. pickling a function that is not defined at the module level
	(hyper_config, config, dataset_module, dataset_config, thread_vars) = package
	manager = ThreadManager(thread_vars)

	device = torch.device("cpu")
	manager.log_print(f"initializing multiprocessing on: {device}")
	manager.log_print(f"Process ID: {os.getpid()}, Process Name: {mp.current_process().name}")
	config["device"] = "cpu"
	# merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
	for k, v in hyper_config.items():
		config[k] = v
	model = initialize.initialize_model(config)
	results = train_model.train_model(model, dataset_module, config, dataset_config, manager=manager)
	manager.save_configs(config, hyper_config)
	return results


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
	logger = loggingx.get_logger(config.get("log_name"))
	logger.info(f"Number of available CPUs: {tot_cpus}")
	logger.info(f"Number of CPUs to use: {num_cpus}")

	# build the hyperparameter space by sampling (gridsearch not yet supported)
	hyper_list = []
	for _ in range(hyper_settings.get("num_samples")):
		hyperparameter_slice = {}
		for key, value in hyper_config.items():
			hyperparameter_slice[key] = np.random.choice(value).item() # keep serializable
		hyper_list.append(hyperparameter_slice)

	# Assemble local variables to be called from within a thread
	tune_directory = config.get("tune_directory") # where all the tuning results live
	threads_directory = os.path.join(tune_directory, "threads/")
	package_list = []
	paths_list = []

	for thread_id, hyperparameter_slice in enumerate(hyper_list):
		# generate a datetime stample YYYY-MM-DD-HH-MM-SS-MS
		tune_path = os.path.join(tune_directory, f"job_{thread_id}") # where this thread's results live
		os.makedirs(tune_path)
		paths_list.append(tune_path)
		logger_name = f"thread_{thread_id}" # we cannot serialize a logger effectively, so we pass around its name
		thread_pkg = {"thread_id": thread_id, "tune_path": tune_path, "logger_name": logger_name}
		package = (hyperparameter_slice, config, dataset_module, dataset_config, thread_pkg)
		package_list.append(package)
		time.sleep(0.1) # to ensure unique timestamps

	with mp.Pool(processes=num_cpus) as pool:
		# Map f to the list of parameters
		all_results = pool.map(trainable, package_list)

	#cleanup
	logger.debug("All threads have completed. Cleaning up...")

	hyper_keys = list(hyper_config.keys())
	header_keys = get_header().split(",")
	columns = header_keys + hyper_keys
	# columns = hyper_keys + header_keys
	data = []
	for i in range(len(all_results)):
		hyper_setting = [hyper_list[i].get(k) for k in hyper_keys]
		best_result = [all_results[i].get(k) for k in header_keys]
		data.append(best_result + hyper_setting)
	df = pd.DataFrame(data, columns=columns)

	model_name = config.get("model")
	dataset_module = config.get("dataset_module")
	only_good_examples = config.get("only_good_examples")
	res_fname = f"{model_name}_{dataset_module}"
	if only_good_examples:
		res_fname += "_only_good_examples"
	res_fname += "_results.csv"
	df.to_csv(os.path.join(tune_directory, res_fname), index=False)

	config_dict = {k: v for k, v in config.items() if not k.startswith("__")}
	for k in hyper_keys:
		config_dict[k] = None
	with open(os.path.join(tune_directory, f"config.json"), "w") as f:
		f.write(json.dumps(config_dict))
	logger.debug("Tuning run completed successfully...")
