import os
from collections import OrderedDict

import numpy as np
import pickle
import torch

class Corpus(object):
	def __init__(self, path, voc, debug=False):
		self.voc = voc
		self.debug= debug
		self.data, self.targets= self.create_ids(path)
		self.nlabels = max(self.targets)+1

	def create_ids(self, path):
		assert os.path.exists(path)
		label_tensors = []
		with open(path, 'rb') as handle:
			dct = pickle.load(handle)
		
		# the type of dct is a dictionary with keys 'line' and 'label'
		# dct['line'] is a list of strings
		# dct['label'] is a list of output strings strings, in order
		lines = dct['line']
		lines = [x.strip() for x in lines]   # Adding last symbol for classification
		lines = [list(x) for x in lines]

		labels = dct['label']
		label_types= list(set(labels))
		label_types.sort()
		label_keys = OrderedDict()

		for k in range(len(label_types)):
			label_keys[label_types[k]] = k
		
		labels = [label_keys[labels[i]] for i in range(len(labels))]
		label_tensors = torch.tensor(labels).type(torch.int64)

		if self.debug:
			return lines[:100], label_tensors[:100]

		return lines, label_tensors


class Sampler(object):
	def __init__(self, corpus, voc, batch_size):
		self.corpus= corpus
		self.batch_size = batch_size
		self.voc = voc
		self.data =corpus.data
		self.targets = corpus.targets
		self.num_batches = np.ceil(len(self.data)/batch_size)


	def get_batch(self, i):
		batch_size= min(self.batch_size, len(self.data) - i)
		
		word_batch = self.data[i: i+batch_size]
		target_batch = self.targets[i:i+batch_size]

		word_lens= torch.tensor([len(x) for x in word_batch], dtype = torch.long)

		try:
			batch_ids= sents_to_idx(self.voc, word_batch)
		except:
			print("ERROR?")
			pdb.set_trace()

		# FIXME: This seemed wrong, it seems to be throwing away
		# the last bit of the 
		# source = batch_ids[:,:-1].transpose(0,1)
		source = batch_ids.transpose(0,1)

		targets= target_batch.clone()
		
		return source, targets, word_lens

	def __len__(self):
		return len(self.data)


def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	if config.mode == 'train' or config.mode == 'tune':
		logger.debug('Loading Training Data...')

		'''Create Vocab'''
		train_path = os.path.join(data_path, config.dataset, 'train.pkl')
		val_path = os.path.join(data_path, config.dataset, 'val.pkl')
		# test_path = os.path.join(data_path, config.dataset, 'test.tsv')


		'''Load Datasets'''
		train_corpus = Corpus(train_path, voc, debug = config.debug)
		train_loader = Sampler(train_corpus, voc, config.batch_size)

		val_corpus = Corpus(val_path, voc, debug = config.debug)		
		val_loader = Sampler(val_corpus, voc, config.batch_size)

		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(len(train_corpus.data), len(val_corpus.data))
		logger.info(msg)
		
		return voc, train_loader, val_loader
	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))
