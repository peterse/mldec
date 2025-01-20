import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn import functional as F
from torch import optim

from mldec.pipelines.utils import EarlyStopping


class TokenEmbedding(nn.Module):
	"""from https://pytorch.org/tutorials/beginner/translation_transformer.html
	
		Args:
			vocab_size: (int) number of tokens in alphabet
			emb_size: (int) model dimension
	"""
	def __init__(self, vocab_size, emb_size):
		super(TokenEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, emb_size)
		self.emb_size = emb_size

	def forward(self, tokens):
		"""
		Input:
			tokens: (batch_size, m) tensor of bits or token indices (m=n or 2n)
		Returns:
			Tensor: (batch_size, n, emb_size), final dimension indexes the embedding vector
		"""
		# FIXME: use lower precision?
		return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
	"""from https://pytorch.org/tutorials/beginner/translation_transformer.html
	
	Note: This has been heavily modified for BATCH FIRST mode.

	Args:
		emb_size: dimension of the embedding, i.e. d_model. MUST BE EVEN
		dropout: dropout rate
	"""
	def __init__(self,
				 emb_size: int,
				 dropout: float,
				 maxlen: int = 5000,
				 disable = False):
		super(PositionalEncoding, self).__init__()
		# this just rearranges the equation from Vaswani et al. (2017)
		den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
		pos = torch.arange(0, maxlen).reshape(maxlen, 1)
		pos_embedding = torch.zeros((maxlen, emb_size))
		pos_embedding[:, 0::2] = torch.sin(pos * den)
		pos_embedding[:, 1::2] = torch.cos(pos * den)

		# insert batch dimension up front for batch_first convention
		pos_embedding = pos_embedding.unsqueeze(0) # (1, maxlen, emb_size)
		# This lets me turn off positional encoding.
		if disable:
			pos_embedding = torch.zeros_like(pos_embedding)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer('pos_embedding', pos_embedding)

	def forward(self, token_embedding):
		"""
		Input:
			token_embedding: (batch_size, n, emb_size)
		Returns:
			Tensor: (batch_size, n, emb_size), with positional encoding
		"""
		# NOTE: dropout has a normalization subroutine so this object might 
		# have a weird norm. For instance, if the token embedding is all zeros you 
		# might get values larger than 1 (the maximum of sin, cos)
		sliced = self.pos_embedding[:, :token_embedding.size(-2)] # (1, sequence_len, emb_size)
		return self.dropout(token_embedding + sliced)
	

def generate_square_subsequent_mask(sz, device):
	mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

class BinarySeq2Seq(nn.Module):
	"""Wrapper class for a bitstring-to-bitstring (seq2seq) transformer, with fixed-length data.
	
	Inputs and outputs are both binary, and no tokenizer is expected.
	"""
	def __init__(self, config=None, device=None, logger=None):
		super(BinarySeq2Seq, self).__init__()

		self.config = config
		self.device = device
		self.logger = logger

		if self.logger:
			self.logger.debug('Initalizing Model...')
		self._initialize_model()

		if self.logger:
			self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		self.criterion = torch.nn.CrossEntropyLoss()

	def _initialize_model(self):
		"""Initializes the model from the configuration"""
		self.model = EncoderDecoderTransformer(
			emb_size=self.config.d_model,
			nhead=self.config.heads, 
			num_encoder_layers=self.config.depth, 
			num_decoder_layers=self.config.depth,
			dim_feedforward=self.config.d_ffn,
			dropout=self.config.dropout,
			norm_first=False,
			src_vocab_size=4,
			tgt_vocab_size=4,
			positional_encoding=True
			).to(self.device)
		# `forward` signature: (src, trg, src_mask, tgt_mask, **kwargs)

	def _initialize_optimizer(self):
		self.params = self.model.parameters()

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		elif self.config.opt =='rmsprop':
			self.optimizer = optim.RMSprop(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=self.config.decay_rate, patience=self.config.decay_patience, verbose=True)

	def trainer(self, source, targets):
		"""Single training step for the model. For seq2seq, we are training on last bit prediction basically."""
		self.optimizer.zero_grad()
		# the prediction scheme is to predict the next bit in the sequence
		# at every place. Thanks Andrej Karpathy.
		tgt_input = targets[:, :-1]
		tgt_out = targets[:, 1:]
		logits = self.model(source, tgt_input)
		loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
		loss.backward()
		if self.config.max_grad_norm > 0:   
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
		self.optimizer.step()     
		return loss.item()
	
	def predict(self, source, sos_token=2):
		"""Predicts the output of the model given a source sequence."""
		max_len = self.config.n + 2 # +2 for SOS and EOS
		# FIXME: shouldn't this be vectorized or something?

		# column-wise recursion
		x = source
		memory = self.model.encode(x).to(self.device) # now this is phi(x), shape (1, 2*N_BITS + 2, emb_dim)
		ys = torch.ones(x.shape[0], 1).fill_(sos_token).type(torch.long).to(self.device)
		for i in range(max_len - 1): # -1 since we start with SOS
			out = self.model.decode(tgt=ys, memory=memory) # (1, tgt_seq_len, emb_dim)
			prob = self.model.generator(out[:, -1])
			_, next_bits = torch.max(prob, dim=1)
			ys = torch.cat([ys, next_bits.reshape(-1, 1)], dim=1)
		
		return ys
	
	def evaluator(self, source, targets, weights=None):
		"""The accuracy per row is 1 if every single bit is correct, zero otherwise.
		
		We ignore the SOS and EOS tokens in the evaluation.

		weights: this will be used to weight the accuracy of each example in the 
		contribution to the final accuracy, e.g. when the number of bits is small.
		"""
		self.optimizer.zero_grad()
		preds = self.predict(source)
		Y_pred = preds[:, 1:-1] # remove SOS, EOS
		Y = targets[:, 1:-1] # remove SOS, EOS
		diff = (Y_pred + Y) % 2
		correct = diff.sum(axis=1) == 0
		# print("truth", Y)
		# print("preds", Y_pred)
		# print("correct", correct)
		if weights is not None:
			correct = torch.multiply(correct, weights)

		# print("weighted correct", correct, weights)
		# raise(Exception("stop"))
		acc = correct.sum()/len(correct)
		return acc    
	
		return None
		# tgt_input = targets[:, :-1]
		# tgt_out = targets[:, 1:]
		# logits = self.model(source, tgt_input)
		# acc = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
		# return acc.item()
	
		# TODO: carefuly choose the evaluator for comparing bitstrings.
		# # if config.model_type == 'RNN':
		# # 	output, hidden = self.model(source, hidden, lengths)
		
		# output = self.model(source)
		# preds = output.cpu().numpy()
		# preds = preds.argmax(axis=1)
		# labels= targets.cpu().numpy()
		# acc= np.array(preds==labels, np.int32).sum() / len(targets)

		# return acc
		return


class EncoderDecoderTransformer(nn.Module):
	"""from https://pytorch.org/tutorials/beginner/translation_transformer.html"""
	def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size,
				 dim_feedforward=512, positional_encoding=True, norm_first = False, dropout=0.1):
		super(EncoderDecoderTransformer, self).__init__()
		self.transformer = Transformer(d_model=emb_size,
									   nhead=nhead,
									   num_encoder_layers=num_encoder_layers,
									   num_decoder_layers=num_decoder_layers,
									   dim_feedforward=dim_feedforward,
									   dropout=dropout,
									   norm_first=norm_first,                                       
									   bias=True,
									   batch_first=True) # (batch, seq_len, d_model)
		
		self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
		self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
		self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, disable=(not positional_encoding))

		# Final layer for output decoder
		self.generator = nn.Linear(emb_size, tgt_vocab_size)

	def forward(self, src, trg, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
		"""
		Let S be the source seq length, T the target seq length, N the batch size, E the embedding dimension.

		Args:
			src: input token embeddings. Shape: (N,S,E) (consistent with Transformer.batch_first=True)
			trg: target token embeddings. Shape: (N,T,E) 
			src_mask: Encoder self-attention mask. Shape is (S,S) or (N⋅num_heads,S,S)
			tgt_mask: Decoder self-attention mask. Shape is (T,T) or (N⋅num_heads,T,T)
			src_padding_mask: This removes padding for ragged seqences, specified per example
			tgt_padding_mask: See above 
			memory_key_padding_mask: See above
		
		Returns:
			Tensor: (N, T, num_tokens) logits for the target sequence
		"""
		src_emb = self.positional_encoding(self.src_tok_emb(src))
		tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
		outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
								src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
		logits = self.generator(outs)

		# Compute loss
		# Forward is only called during training/validation, so this is fine
		B, T, C = logits.shape
		logits = logits.view(B*T, C)
		return logits

	def encode(self, src, src_mask=None):
		src_pos_emb = self.positional_encoding(self.src_tok_emb(src))
		return self.transformer.encoder(src_pos_emb, src_mask)

	def decode(self, tgt, memory, tgt_mask=None):
		tgt_pos_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
		return self.transformer.decoder(tgt_pos_emb, memory, tgt_mask)
	

def build_model(config, device, logger):
	model = BinarySeq2Seq(config, device, logger)
	model = model.to(device)
	# print total number of model parameters
	if logger:
		logger.debug('Model has {} trainable parameters'.format(
			sum(p.numel() for p in model.parameters() if p.requires_grad)))
	return model


def run_validation(model, data_loader, device, weights=None):
	"""Compute the accuracy of the model on a dataset, by exhausting the batch sampler."""
	model.eval()
	batch_num = 0
	val_acc_epoch = 0.0
	if weights is not None:
		weights = weights.to(device)

	with torch.no_grad():
		for batch, i in enumerate(range(0, len(data_loader), data_loader.batch_size)):
			source, targets = data_loader.get_batch(i)
			source, targets = source.to(device), targets.to(device) # to(device) is cost-free if the tensor is already on the requested device
			acc = model.evaluator(source, targets, weights)
			val_acc_epoch+= acc
			batch_num+=1

	if batch_num != data_loader.num_batches:
		raise Exception("the fuck?")
		# pdb.set_trace()
	val_acc_epoch = val_acc_epoch/data_loader.num_batches
	
	return val_acc_epoch


def run_weighted_validation(model, data_loader, weights, device):
	"""Compute the accuracy of the model with respect to a set of weights"""
	model.eval()
	batch_num = 0
	acc = 0.0

	with torch.no_grad():
		for batch, i in enumerate(range(0, len(data_loader), data_loader.batch_size)):
			source, targets = data_loader.get_batch(i)
			source, targets = source.to(device), targets.to(device)
			acc += model.evaluator(source, targets) * weights # FIXME

	acc = acc/data_loader.num_batches
	
	return acc


def train_model(model, train_loader, test_loader, device, 
				config, logger, epoch_offset= 0):

	max_val_acc = 0
	best_epoch = 0
	curr_train_acc = 0.0
	itr= 0
	data_size = len(train_loader)
	# Insert my early stopping criteria/class...
	early_stopping = EarlyStopping(patience=1000, delta=0.0, logger=logger)

	for epoch in range(1, config.epochs):
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		test_acc_epoch = 0.0
		model.train()
		start_time = time.time()
		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']
		
		for i in range(train_loader.num_batches):
			source, targets = train_loader.get_batch(i)			
			source, targets = source.to(device), targets.to(device)
			loss = model.trainer(source, targets)
			train_loss_epoch += loss 
			itr +=1
		
		train_loss_epoch = train_loss_epoch/train_loader.num_batches
		time_taken = time.time() - start_time
		time_mins = int(time_taken / 60)
		time_secs= time_taken % 60

		logger.debug('Training for epoch {} completed...\nTime Taken: {} mins and {} secs'.format(epoch, time_mins, time_secs))
		logger.debug('Starting Validation')

		test_acc_epoch = run_validation(model, test_loader, device, weights=test_loader.weights)
		train_acc_epoch = run_validation(model, train_loader, device)

		early_stopping( (-1) * test_acc_epoch, model)

		if config.opt == 'sgd':
			model.scheduler.step(test_acc_epoch)

		# if config.mode == 'tune':
			# train.report({
			# 	"train_loss": (train_loss_epoch),
			# 	"train_acc": train_acc_epoch,
			# 	"val_acc": test_acc_epoch
			# })
		else:
			logger.debug('Epoch: {} | Train Loss: {} | Train Acc: {} | Val Acc: {} | LR: {}'.format(
				epoch + epoch_offset, train_loss_epoch, train_acc_epoch, test_acc_epoch, lr_epoch))

		if test_acc_epoch > max_val_acc :
			max_val_acc = test_acc_epoch
			best_epoch= epoch
			curr_train_acc= train_acc_epoch

		# Break if we haven't had consistent progress 
		if early_stopping.early_stop:
			break

		# od = OrderedDict()
		# od['Epoch'] = epoch + epoch_offset
		# od['train_loss'] = train_loss_epoch
		# od['train_acc'] = train_acc_epoch
		# od['test_acc_epoch']= test_acc_epoch
		# od['max_val_acc']= max_val_acc
		# od['lr_epoch'] = lr_epoch
		# od['conv_time'] = conv_time
		# print_log(logger, od)

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	# if config.results:
	# 	store_results(config, max_val_acc, curr_train_acc, best_epoch)
	# 	logger.info('Scores saved at {}'.format(config.result_path))

	
# def build_and_train_model_raytune(hyper_config, config, train_loader, val_loader, voc, 
# 				logger, epoch_offset= 0, min_val_loss=1e7, 
# 				max_val_acc=0.0,
# 				):
# 	"""Build and train a model with the given hyperparameters
	
# 	Reasons why this exists:
# 	 - ray (or pickle) cannot serialize torch models
# 	 - We need to distribute the model training/building pipeline to multiple workers
# 	"""
# 	device = get_device()
# 	for key, value in hyper_config.items():
# 		setattr(config, key, value)
# 	model = build_model(config, voc, device, logger)
# 	train_model(model, train_loader, val_loader, voc, device, config, logger, epoch_offset, min_val_loss, max_val_acc)

# def trial_dirname_creator(trial):
#     return f"{trial.trainable_name}_{trial.trial_id}"

# def tune_model(hyper_settings, hyper_config, train_loader, val_loader, voc, 
# 				config, logger, epoch_offset= 0, min_val_loss=1e7, 
# 				max_val_acc=0.0):	
	
	
# 	# config should have tune=True
# 	scheduler = ASHAScheduler(
# 		metric="train_loss",
# 		mode="min",
# 		max_t=hyper_settings.get("epochs"), # I'm not sure what this kwarg does and neither is the documentation
# 		grace_period=1,
# 		reduction_factor=2)

# 	reporter = CLIReporter(
# 		metric_columns=["loss", "training_iteration", "mean_accuracy"],
# 		print_intermediate_tables=False,
# 		)

# 	tune_config = tune.TuneConfig(
# 		num_samples=hyper_settings.get("num_samples"),
# 		trial_dirname_creator=trial_dirname_creator,
# 		scheduler=scheduler,
# 		max_concurrent_trials=hyper_settings.get("max_concurrent_trials"),
# 		)
		
# 	run_config = RunConfig(
# 		progress_reporter=reporter,
# 		stop={"training_iteration": config["epochs"], "mean_accuracy": 0.8},
# 	)
# 	trainable = tune.with_parameters(
# 					build_and_train_model_raytune, 
# 					config=config,
# 					train_loader=train_loader,
# 					val_loader=val_loader,
# 					voc=voc,
# 					logger=logger,
# 					epoch_offset=epoch_offset,
# 					min_val_loss=min_val_loss,
# 					max_val_acc=max_val_acc
# 					)
	
# 	resources = tune.with_resources(
# 		trainable,
# 		{"cpu": hyper_settings.get("cpus_per_worker"), "gpu": hyper_settings.get("gpus_per_worker")},
# 	)
	
# 	tuner = Tuner(
# 		resources,
# 		param_space=hyper_config,
# 		tune_config=tune_config,
# 		run_config=run_config,
# 	)
# 	result = tuner.fit()

# 	return result			