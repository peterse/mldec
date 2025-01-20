import os
import torch
import numpy as np
import pickle


class Sampler(object):
	def __init__(self, path, batch_size, mode='train', add_sos_eos=True, preload_device=None):
		self.weights = None
		self.source, self.targets = self._load_data(path, mode=mode, add_sos=add_sos_eos)
		self.batch_size = batch_size
		self.num_batches = int(np.ceil(len(self.source)/batch_size))
		self.preload = False
		if preload_device is not None:
			self.preload = True
			self.source = self.source.to(preload_device)
			self.targets = self.targets.to(preload_device)
			if mode == 'test':
				self.weights = self.weights.to(preload_device)

	def _load_data(self, path, mode, add_sos):
		""" Load the data from a predetermined path.
		
		Args:
			path (str): path to the data file. Format of this file is a 
				pickle file with a dictionary containing keys 'line' and 'label'.
			mode: This indicates whether we are working with train or test data,
				in the latter case we choose to do weighted accuracy evals.
		"""
            
		assert os.path.exists(path)
        
		if mode == 'train':
			lines = np.load(os.path.join(path, 'X_train.npy'))
			labels = np.load(os.path.join(path, 'y_train.npy'))
                  
		elif mode == 'test':
			lines = np.load(os.path.join(path, 'X_test.npy'))
			labels = np.load(os.path.join(path, 'y_test.npy'))
			weights = np.load(os.path.join(path, 'weights.npy'))
			self.weights = torch.tensor(weights).type(torch.float32)

		if add_sos:
			# If we are doing autoregressive prediction, we might want an SOS on our labels.
			self.sos_token = 2
			self.eos_token = 3
			labels = np.concatenate( (np.full((labels.shape[0], 1), self.sos_token), labels), axis=1)
			labels = np.concatenate( (labels, np.full((labels.shape[0], 1), self.eos_token)), axis=1)
			
		self.n_data = len(lines)
		self.output_shape = labels.shape

		# label_tensors = []
		# with open(path, 'rb') as handle:
		# 	dct= pickle.load(handle)

		# # FIXME: is this the right way to store data?
		# lines = dct['line']
		# lines = [x.strip() for x in lines]  
		# lines = [list(x) for x in lines]

		# labels = dct['label']
		# labels = [x.strip() for x in labels]
		# labels = [list(x) for x in labels]

		line_tensors = torch.tensor(lines).type(torch.int64)
		label_tensors = torch.tensor(labels).type(torch.int64)


		return line_tensors, label_tensors

	def get_batch(self, i):

		batch_size= min(self.batch_size, len(self.source) - i)
		if self.preload:
			source_batch = self.source[i:i+batch_size]
			target_batch = self.targets[i:i+batch_size]
		else:
			source_batch = self.source[i:i+batch_size]
			target_batch = self.targets[i:i+batch_size]
		# batch_size= min(self.batch_size, len(self.source) - i)
		# source_batch = self.source[i:i+batch_size]
		# target_batch = self.targets[i:i+batch_size]
		# sources = source_batch.clone()
		# targets = target_batch.clone()
		return source_batch, target_batch

	def __len__(self):
		return len(self.source)


def save_numpy_as_dict(data, data_path):
    """Save the output of a data generator as a set of dictionaries.
    
    Args:
        data (np.array): The data (n_data, n_bits) array
    """
    data_dict = {'line':[], 'label':[]}
    for i in range(data.shape[0]):
        data_dict['line'].append(''.join([str(x) for x in data[i,:-1]]))
        data_dict['label'].append(str(data[i][-1]))
    with open(data_path, 'wb') as handle:
        pickle.dump(data_dict, handle)

def load_dict_as_numpy(data_path):
    with open(data_path, 'rb') as handle:
        data_dct = pickle.load(handle)
    data = []
    for i in range(len(data_dct['line'])):
        x = [int(x) for x in data_dct['line'][i]]
        y = int(data_dct['label'][i])
        data.append(x + [y])
    data = np.array(data)
    return data