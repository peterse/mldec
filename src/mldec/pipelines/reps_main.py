import os
from mldec.datasets import reps_toric_code_data
from mldec.pipelines import loggingx
from mldec.models import initialize, reps_train_model, tune_model
import torch
import numpy as np
import logging
import datetime


def main(config):

	dataset_module = config.get("dataset_module")
	if dataset_module == "reps_toric_code":
		n = config['n']
		# this dataset describes what data the model will be evaluated on.
		dataset_config = {
			'p': 0.001,
			'repetitions': 5,
			'code_size': 3,
			'beta': 1, # this gets overwritten by the knob settings
		}
	else:
		raise ValueError("Unknown dataset module")
	
	abs_path = os.path.dirname(os.path.abspath(__file__))
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	config['log_name'] = timestamp
	# use a timestamp YYYYMMDDHHMMSS to identify the run
	if config.get("mode") == 'train':
		config["log_path"] = 'train_results/'
		if not os.path.exists(config["log_path"]):
			os.makedirs(config["log_path"])
		log_file = os.path.join(config.get("log_path"), f'{timestamp}.txt')
	elif config.get("mode") == 'tune':
		config["tune_directory"] = tune_model.make_tune_directory(config, abs_path) # makes tune_results/{model}_{dataset}/run_{timestamp}
		log_file = os.path.join(config.get("tune_directory"), 'log.txt')

	logger = loggingx.init_logger(timestamp, log_file_path=log_file, logging_level=logging.DEBUG)

	if config.get("mode") == "tune":
		# Load in the hyperparameters from a config path
		hyper_config_dir = os.path.join(abs_path, "..", "hyper_config")
		hyper_config_path = os.path.abspath(os.path.join(hyper_config_dir, config.get("hyper_config_path")))
		yaml = tune_model.load_hyperparameters(hyper_config_path)
		model_type = config.get("model")
		model_type_from_yaml = yaml["model_type"] 
		if model_type != model_type_from_yaml:
			raise ValueError(f"Model type {model_type} from args is different from model type {model_type_from_yaml} in hyperparameter config file")
		
		# Load in the tuning deck parameters
		knob_settings = yaml["knob_settings"]
		tune_model.validate_knob_settings(config, knob_settings, dataset_config, logger)
		hyper_config = yaml["hyperparameters"]
		tune_model.validate_tuning_parameters(config, hyper_config, logger)
		hyper_settings = yaml["settings"]
		tune_model.tune_hyperparameters_multiprocessing(
			hyper_config, hyper_settings, dataset_module, 
			config, dataset_config, knob_settings,
			delete_intermediate_dirs=False)
	else:
		model_wrapper = initialize.initialize_model(config)
		reps_train_model.train_model(model_wrapper, dataset_module, config, dataset_config, knob_settings)

if __name__ == "__main__":

	# Some notes:
	# scale up lr with batchsize in general.
	# patience = early stopping patience, triggered on no improvement in val_acc for `patience` epochs (with checking every X epochs)\
	# !OVERWRITE indiicates a hyperparam that may be overwritten by raytune `hyper_config` or raytune internals
	# only_good_examples = uniform distribution over good examples
	# SERIALIZABILITY: All of the config options, hyper options, dataset_config options must be serializable (json)

	# # # important stuff # # # # # # # # # 
	mode = "tune" # options: train, tune
	dataset_module = "reps_toric_code" # options: reps_toric_code
	MODEL = "gnn" # options: gnn
	# # # # # # # ## # # # # # # # # # # # # 
	if dataset_module == "reps_toric_code":
		# FIXME
		n = 8
		input_dim = 5 # 5-dimensional vectors to represent graph coordinates
		output_dim = 1 # binary clf

	config = {
		"model" : MODEL,
		"hyper_config_path": f"{MODEL}_{dataset_module}.yaml",
		"device": "cpu", 
		"n": n,
		# "n_train": 100000, # !OVERWRITE
		"n_test": 1000000,
		"dataset_module": dataset_module,
		# Training config: 
		"max_epochs": 3000,
		"patience": 100,  
		"opt": "adam",
		"mode": mode,
		"input_dim": input_dim,
		"output_dim": output_dim,
		# "lr": 0.003, # !OVERWRITE
		# "batch_size": 250, # !OVERWRITE
		# "dropout": 0.05, # 
	}

	if config.get("model") == "gnn":
		model_config = {
			"model": "gnn",
			# "gcn_depth": 5,# !OVERWRITE
			# "gcn_min": 32,# !OVERWRITE
			# "mlp_depth": 3,# !OVERWRITE
			# "mlp_max": 64,# !OVERWRITE
		}


	config.update(model_config)
	main(config)