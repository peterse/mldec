import torch
import numpy as np
import os
import shutil
import datetime
import json
import time
import multiprocessing as mp
import pandas as pd
import yaml

from mldec.models import initialize, train_model, reps_train_model
from mldec.pipelines import loggingx


def make_tune_results_path(path):
	return os.path.join(path, "tune_results.csv")

def hyper_config_path(path):
	return os.path.join(path, f"hyper_config.json")

def get_header():
	return "job_id,epoch,train_loss,train_acc,val_loss,val_acc,vs_lookup,vs_minweight"

def load_hyperparameters(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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
		self.job_id = self.thread_id
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
		write_str = f"{self.thread_id},"
		# skip the job_id, which is the first column in the header
		split_header = get_header().split(",")[1:]
		for i, k in enumerate(split_header):
			write_str += f"{epoch_results[k]}"
			if i < len(split_header) - 1:
				write_str += ","
			else:
				write_str += "\n"
		with open(self.tune_results_path, "a") as f:
			f.write(write_str)

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
	tune_directory = os.path.abspath(os.path.join(abs_path, "..", "tune_results"))
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


class MakeTrainable:
	"""Dispatching a trainer in a serializable way."""
	def __init__(self, trainer):
		self.trainer = trainer
	def __call__(self, package):
		"""This function is now inside its own thread"""
		# Note: Function wrapping within `tune_hyperparameters_multiprocessing` is dangerous
		# because of serializability issues, e.g. pickling a function that is not defined at the module level
		(hyper_config, config, dataset_module, dataset_config, knob_settings, thread_vars) = package
		manager = ThreadManager(thread_vars)

		device = torch.device("cpu")
		manager.log_print(f"initializing multiprocessing on: {device}")
		manager.log_print(f"Process ID: {os.getpid()}, Process Name: {mp.current_process().name}")
		config["device"] = "cpu"
		# merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
		for k, v in hyper_config.items():
			config[k] = v
		model = initialize.initialize_model(config)
		results = self.trainer(model, dataset_module, config, dataset_config, knob_settings, manager=manager)
		manager.save_configs(config, hyper_config)
		return results


def distribute_hyperparameters_evenly(num_samples, knob_settings, key, fixed_dict={}):
	"""Evenly distribute a list of knob settings over a number of samples.
	
	Potentially also concatenating each setting with `fixed_dict`.

	Example usage
		num_samples = 3
		knob_settings = [1, 2, 4]
		key = "a"
		fixed_dict = {"b": 3}
	The results will be
		[{"a": 1, "b": 3}, {"a": 2, "b": 3}, {"a": 4, "b": 3}]
	Returns: A list of dictionaries, each containing a single knob setting and the fixed settings.
	"""
	knob_list = []
	div = len(knob_settings.get(key)) 
	rem = num_samples % div
	quot = num_samples // div
	for i in range(div):
		# we will get about `div` samples for each knob setting
		for _ in range(quot):
			temp = {key: knob_settings.get(key)[i]}
			temp.update(fixed_dict) 
			knob_list.append(temp)
	# distribute the remainder
	for j in range(rem):
		temp = {key: knob_settings.get(key)[j]}
		temp.update(fixed_dict)
		knob_list.append(temp)
	return knob_list

def tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, dataset_module, config, dataset_config, knob_settings, delete_intermediate_dirs=True):
	"""
	hyper_config should contain a list for each hyperparameter in the search, e.g.
	hyper_config = {
		'lr': [0.00001, 0.00005, ...],
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

	# get knob settings to pass directly into train func
	# we distribute them evenly among all of the samples for this run.
	num_samples = hyper_settings.get("num_samples")
	if dataset_module == "toy_problem" or dataset_module == "toy_problem_unbiased":
		knob_list = distribute_hyperparameters_evenly(num_samples, knob_settings, "p")
	elif dataset_module == "toric_code":
		knob_list = distribute_hyperparameters_evenly(num_samples, knob_settings, "beta", fixed_dict={"var": knob_settings.get("var")})
	elif dataset_module == "reps_toric_code":
		knob_list = distribute_hyperparameters_evenly(num_samples, knob_settings, "beta")
	elif dataset_module == "reps_exp_rep_code":
		knob_list = distribute_hyperparameters_evenly(num_samples, knob_settings, "beta")
	else:
		raise NotImplementedError
	logger.debug("Distributing these hyperparameters: {}".format(knob_list))


	# Assemble local variables to be called from within a thread
	tune_directory = config.get("tune_directory") # where all the tuning results live
	threads_directory = os.path.join(tune_directory, "threads/")
	package_list = []
	paths_list = []

	for thread_id, hyperparameter_slice in enumerate(hyper_list):
		# generate a datetime stample YYYY-MM-DD-HH-MM-SS-MS
		knob_slice = knob_list[thread_id]
		tune_path = os.path.join(tune_directory, f"zjob_{thread_id}") # where this thread's results live
		os.makedirs(tune_path)
		paths_list.append(tune_path)
		logger_name = f"thread_{thread_id}" # we cannot serialize a logger effectively, so we pass around its name
		thread_pkg = {"thread_id": thread_id, "tune_path": tune_path, "logger_name": logger_name}
		package = (hyperparameter_slice, config, dataset_module, dataset_config, knob_slice, thread_pkg)
		logger.info(package)
		package_list.append(package)
		time.sleep(0.1) # to ensure unique timestamps
	
	# Different training schemes for with/without multiple code cycles.
	if dataset_module in ["toy_problem", "toric_code", "toy_problem_unbiased"]:
		trainable = MakeTrainable(trainer=train_model.train_model)
	elif dataset_module in ["reps_toric_code", "reps_exp_rep_code"]:
		trainable = MakeTrainable(trainer=reps_train_model.train_model)

	with mp.Pool(processes=num_cpus) as pool:
		# Map f to the list of parameters
		all_results = pool.map(trainable, package_list)

	hyper_keys = list(hyper_config.keys())
	knob_keys = list(knob_settings.keys())
	header_keys = get_header().split(",")
	auxiliary_keys = ["job_id", "total_parameters", "total_time", "total_epochs"]
	columns = header_keys + hyper_keys + knob_keys + auxiliary_keys
	data = []
	for i in range(len(all_results)):
		hyper_setting = [hyper_list[i].get(k) for k in hyper_keys]
		knob_setting = [knob_list[i].get(k) for k in knob_keys]
		best_result = [all_results[i].get(k) for k in header_keys]
		auxiliary_result = [all_results[i].get(k) for k in auxiliary_keys]
		data.append(best_result + hyper_setting + knob_setting + auxiliary_result)
	df = pd.DataFrame(data, columns=columns)

	model_name = config.get("model")
	dataset_module = config.get("dataset_module")
	only_good_examples = config.get("only_good_examples")
	res_fname = f"{model_name}_{dataset_module}"
	if only_good_examples:
		res_fname += "_only_good_examples"
	res_fname += "_results.csv"
	df.to_csv(os.path.join(tune_directory, res_fname), index=False)

	#cleanup
	logger.debug("All threads have completed. Cleaning up...")
	if delete_intermediate_dirs:
		for path in paths_list:
			shutil.rmtree(path)


	config_dict = {k: v for k, v in config.items() if not k.startswith("__")}
	for k in hyper_keys:
		config_dict[k] = None
	with open(os.path.join(tune_directory, f"config.json"), "w") as f:
		f.write(json.dumps(config_dict))
	logger.debug("Tuning run completed successfully...")



def validate_tuning_parameters(config, hyper_config, logger):
	"""As you might guess, this exists because of a major mistake"""
	ALLOWED_HYPERS = {
		# "rnn": ["cell_type", "emb_size", "hidden_size", "depth", "dropout"],
		"transformer": ["d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward", "dropout"],
		"cnn": ["conv_channels", "n_layers", "kernel_size", "dropout"],
		"ffnn": ["hidden_dim", "n_layers", "dropout"],
		"gnn": ["gcn_depth", "gcn_min", "mlp_depth", "mlp_max"],
	}
	error = 0
	err_out = ""
	allowed_hypers = ALLOWED_HYPERS.get(config["model"])
	for k in allowed_hypers:
		if config.get(k) is not None and hyper_config.get(k) is not None:
			err_str = "Hyperparameter {} is specified in `config` and `hyper_config`".format(k)
			err_out += err_str + "\n"
			logger.error(err_str)
			error = 1
		if config.get(k) is None and hyper_config.get(k) is None:
			err_str = "Hyperparameter {} is not specified in `config` or `hyper_config`".format(k)
			err_out += err_str + "\n"
			logger.error(err_str)
			error = 1
	for k, v in ALLOWED_HYPERS.items():
		if k == config["model"]:
			continue
		for hyper_k in v:
			if hyper_k in hyper_config and hyper_k not in allowed_hypers:
				err_str = "Hyperparameter {} is specified in `hyper_config`, but `model` is {}".format(hyper, k)
				err_out += err_str + "\n"
				logger.error(err_str)
				error = 1
	# force an error if hyper_config is overwriting a set config option
	for k, v in hyper_config.items():
		if k in config:
			err_str = "Hyperparameter {} is specified in `hyper_config`, but `config` is also set. Delete that setting in config".format(k)
			err_out += err_str + "\n"
			logger.error(err_str)
			error = 1
	if error:
		raise ValueError("Hyperparameters are specified in both commandline and hyperparameters:\n{}".format(err_out))
	
	# Handling training amounts
	if hyper_config.get("batch_size") is not None and hyper_config.get("n_batches") is not None:
		if config.get("n_train"):
			for batch_size in hyper_config.get("batch_size"):
				if batch_size * config.get("n_batches") != config.get("n_train"):
					err_str = "Batch size * n_batches must be equal to n_train, or do not specify n_train"
					logger.error(err_str)
					error = 1

	logger.debug("Hyperparameters validated")


def validate_knob_settings(config, knob_settings, dataset_config, logger):
	# check that the dataset config is valid
	return 
	# dataset_module = config.get("dataset_module")
	# error = 0
	# if knob_settings.get("dataset_module") != dataset_module:
	# 	err_str = f"Dataset module {dataset_module} does not match dataset module in hyperparameters {dataset_module}"
	# 	err_out += err_str + "\n"
	# 	error = 1
	# if error:
	# 	raise ValueError("Hyperparameters are specified in both commandline and hyperparameters:\n{}".format(err_out))