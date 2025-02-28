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
		# this dataset describes what data the model will be evaluated on.
		dataset_config = {
			'p': 0.1,
			'alpha': 0.7,
			'pcm': toy_problem_data.repetition_pcm(n),
			"sos_eos": config.get("sos_eos", None),
		}
		# KNOB SETTINGS: This is an additional description of what data the model will be trained on.
		# train mode:
		# anything not set in `knob_settings` will be defaulted to the `dataset_config`.
		# tune mode:
		# Anything not set in this can be written with knob_settings in the hyperparameter config.
		# anything set in this that is attempted to be overwritten by the hyperparameter config will raise an error.
		knob_settings = {
			# 'p': dataset_config.get('p'), # !OVERWRITE # how much to scale 'p' by
			'alpha': dataset_config.get('alpha'),
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
		
		knob_settings = yaml["knob_settings"]
		tune_model.validate_knob_settings(config, knob_settings, dataset_config, logger)

		hyper_config = yaml["hyperparameters"]
		tune_model.validate_tuning_parameters(config, hyper_config, logger)
		
		hyper_settings = yaml["settings"]
		tune_model.tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, dataset_module, config, dataset_config, knob_settings)
	else:
		model_wrapper = initialize.initialize_model(config)
		train_model.train_model(model_wrapper, dataset_module, config, dataset_config, knob_settings)

if __name__ == "__main__":

	# Some notes:
	# scale up lr with batchsize in general.
	# patience = early stopping patience, triggered on no improvement in val_acc for `patience` epochs (with checking every X epochs)\
	# n_train = virtual training samples, i.e. noise in a histogram of the error distribution
	# !OVERWRITE indiicates a hyperparam that may be overwritten by raytune `hyper_config` or raytune internals
	# only_good_examples = uniform distribution over good examples
	# SERIALIZABILITY: All of the config options, hyper options, dataset_config options must be serializable (json)

	only_good_examples = False
	mode = "tune" # options: train, tune
	n = 8
	input_dim = n - 1
	MODEL = "cnn"
	config = {
		"model" : MODEL,
		"hyper_config_path": f"{MODEL}_toyproblem.yaml",
		"device": "cpu", 
		# Dataset config
		"n": n,
		"only_good_examples": only_good_examples, 
		"n_train": 2000,
		"dataset_module": "toy_problem",
		# Training config: 
		"max_epochs": 10000,
		# "batch_size": 250, # !OVERWRITE
		"patience": 4000,  
		# "lr": 0.003, # !OVERWRITE
		"opt": "adam",
		"mode": mode,
		# fixed model config
		"input_dim": input_dim,
		"output_dim": n,
		# "dropout": 0.05, # !OVERWRITE
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
			# "kernel_size": 3, # !OVERWRITE
			# "n_layers": 3, # !OVERWRITE
		}
	elif config.get("model") == "transformer":
		config["sos_eos"] = (0, 0)
		model_config = {
			"model": "transformer",
			# "d_model": 16, # !OVERWRITE
			# "nhead": 4, # !OVERWRITE
			# "num_encoder_layers": 2, # !OVERWRITE
			# "num_decoder_layers": 2, # !OVERWRITE
			# "dim_feedforward": 8, # !OVERWRITE
		}

	config.update(model_config)
	main(config)