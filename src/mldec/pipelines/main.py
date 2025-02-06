import os
from mldec.datasets import toy_problem_data
from mldec.pipelines import loggingx
from mldec.models import initialize, train_model, tune_model
import torch
import numpy as np
import logging
import datetime


def main(config):

	# Configure the dataset.
	# Why pass a module? The main observation is that actually
	# training on real data is comparatively slow; we instead
	# reweight a loss function according to a sample from the 
	# underlying data distribution. This virtual sampling is done
	# just-in-time.
	dataset_module = config.get("dataset_module")
	if dataset_module == "toy_problem":
		n = config['n']
		dataset_config = {
			'p1': 0.1,
			'p2': 0.07,
			'pcm': toy_problem_data.repetition_pcm(n),
			"sos_eos": config.get("sos_eos", None),
		}
	else:
		raise ValueError("Unknown dataset module")
	
	abs_path = os.path.dirname(os.path.abspath(__file__))
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	config['log_name'] = timestamp
	# use a timestamp YYYYMMDDHHMMSS to identify the run
	if config.get("mode") == 'train':
		config["log_path"] = 'train_results/'
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
		hyper_config = yaml["hyperparameters"]
		hyper_settings = yaml["settings"]

		tune_model.validate_tuning_parameters(config, hyper_config, logger)
		tune_model.tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, dataset_module, config, dataset_config)
	else:
		model_wrapper = initialize.initialize_model(config)
		train_model.train_model(model_wrapper, dataset_module, config, dataset_config)

if __name__ == "__main__":
	only_good_examples = True
	mode = "tune" # options: train, tune
	n = 8
	input_dim = n - 1

	# Some notes:
	# scale up lr with batchsize in general.
	# patience = early stopping patience, triggered on no improvement in val_acc for `patience` epochs (with checking every X epochs)\
	# n_train = virtual training samples, i.e. noise in a histogram of the error distribution
	# !OVERWRITE indiicates a hyperparam that may be overwritten by raytune `hyper_config` or raytune internals
	# only_good_examples = uniform distribution over good examples
	# SERIALIZABILITY: All of the config options, hyper options, dataset_config options must be serializable (json)
	config = {
		"model" : "cnn",
		"device": "cpu", 
		# Dataset config
		"n": n,
		"only_good_examples": only_good_examples, 
		"n_train": 1000,
		"dataset_module": "toy_problem",
		# Training config: 
		"max_epochs": 10,
		"batch_size": 50,
		"patience": 5000,  
		# "lr": 0.003, # !OVERWRITE
		"opt": "adam",
		"mode": mode,
		# fixed model config
		"input_dim": input_dim,
		"output_dim": n,
		# "dropout": 0.05, # !OVERWRITE
		"hyper_config_path": "cnn_toyproblem_only_good.yaml",

	}

	if config.get("model") == "ffnn":
		model_config = {
			"model": "ffnn",
			"hidden_dim": 16, # !OVERWRITE
			"n_layers": 3, # !OVERWRITE
			"dropout": 0, # !OVERWRITE
		}
	elif config.get("model") == "cnn":
		model_config = {
			"model": "cnn",
			# "conv_channels": 4, # !OVERWRITE
			"kernel_size": 3, # !OVERWRITE
			# "n_layers": 3, # !OVERWRITE
		}
	elif config.get("model") == "encdec":
		config["sos_eos"] = (0, 0)
		model_config = {
			"model": "encdec",
			"d_model": 16, # !OVERWRITE
			"nhead": 4, # !OVERWRITE
			"num_encoder_layers": 2, # !OVERWRITE
			"num_decoder_layers": 2, # !OVERWRITE
			"dim_feedforward": 8, # !OVERWRITE
		}

	config.update(model_config)
	main(config)