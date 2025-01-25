"""encdec.py - Encoder Decoder transformer"""
import torch
import torch.nn as nn
from torch.nn import Transformer
import math


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

		input_dim: (int) number of bits in the input sequence
		output_dim: (int) number of bits in the output sequence
	"""
	def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0, device=None, sos_token=0):
		super(BinarySeq2Seq, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.d_model = d_model
		self.nhead = nhead
		self.num_encoder_layers = num_encoder_layers
		self.num_decoder_layers = num_decoder_layers
		self.dim_feedforward = dim_feedforward
		self.dropout = dropout
		self.device = device
		self.sos_token = sos_token

		self.cached_memory = None
		self._initialize_model()

		# create the mask manually, accounting for sos/eos concats, minus one for the trianing scheme
		self.tgt_mask = generate_square_subsequent_mask(output_dim + 1, device='cpu')

	def _initialize_model(self):
		"""Initializes the model from the configuration"""
		self.model = EncoderDecoderTransformer(
			d_model=self.d_model,
			nhead=self.nhead, 
			num_encoder_layers=self.num_encoder_layers, 
			num_decoder_layers=self.num_decoder_layers,
			dim_feedforward=self.dim_feedforward,
			dropout=self.dropout,
			norm_first=False,
			src_vocab_size=2,
			tgt_vocab_size=2,
			positional_encoding=True
			).to(self.device)
		# `forward` signature: (src, trg, src_mask, tgt_mask, **kwargs)

	def training_step(self, X, Y, weights, optimizer, criterion):
		"""Single training step for the model. For seq2seq, we are training on last bit prediction."""
		optimizer.zero_grad()
		# the prediction scheme is to predict the next bit in the sequence
		# at every place. Thanks Andrej Karpathy.
		tgt_input = Y[:, :-1] # shape [batch, output_dim - 1]
		tgt_out = Y[:, 1:] # shape [batch, output_dim - 1]
		logits = self.model(X, tgt_input, tgt_mask=self.tgt_mask) # [B, T, C]
		# We are calling the criterion on the output logits vs. a _slice_
		loss = criterion(logits, tgt_out, weights)
		loss.backward()
		optimizer.step()     
		return loss
	
	def predict(self, X, use_cached_memory=False, return_activations=False):
		"""Predicts the output of the model given a source sequence.
		WARNING: This will output the entire sequence, including SOS and EOS tokens.

		`is_memory`: You can provide the encoded X as memory to save time.
		Returns the predicted sequence and the memory from the encoder. You can use the 
		cached memory to evaluate accuracy, e.g.
		"""
		max_len = self.output_dim + 2 # +2 for SOS and EOS

		# X = self.model.encode(X).to(self.device) # now this is phi(x), shape [B, output_dim , d_model]
		# column-wise recursion: Note that this takes about T times longer than the encoder step
		# TODO: There might be a caching technique to make this faster
		Y_pred = torch.ones(X.shape[0], 1).fill_(self.sos_token).type(torch.float).to(self.device)
		acts_out = torch.ones(X.shape[0], 1).fill_(-100).type(torch.float).to(self.device)

		for _ in range(max_len - 1): # -1 since we start with SOS
			logits = self.model(X, Y_pred) # [B, i]
			last_logit = logits[:, -1] 
			next_bit = (last_logit >= 0).float()
			# next_bit = torch.sigmoid(last_logit >= 0).float()
			Y_pred = torch.cat([Y_pred, next_bit.view(-1, 1)], dim=1)
			# out = self.model.decode(tgt=Y_pred, memory=X) # [B, output_dim + 2, d_model]
			# prob = self.model.generator(out[:, -1]) # [B, C] with C=num_tokens
			# next_acts = torch.matmul(prob, torch.tensor([-1, 1], dtype=prob.dtype)) # [B]
			# next_bits = (next_acts > 0).float()

			# Y_pred = torch.cat([Y_pred, next_bits.view(-1, 1)], dim=1)

		return Y_pred
	


	# def evaluator(self, source, targets, weights=None):
	# 	"""The accuracy per row is 1 if every single bit is correct, zero otherwise.
		
	# 	We ignore the SOS and EOS tokens in the evaluation.

	# 	weights: this will be used to weight the accuracy of each example in the 
	# 	contribution to the final accuracy, e.g. when the number of bits is small.
	# 	"""
	# 	self.optimizer.zero_grad()
	# 	preds = self.predict(source)
	# 	Y_pred = preds[:, 1:-1] # remove SOS, EOS
	# 	Y = targets[:, 1:-1] # remove SOS, EOS
	# 	diff = (Y_pred + Y) % 2
	# 	correct = diff.sum(axis=1) == 0

	# 	if weights is not None:
	# 		correct = torch.multiply(correct, weights)

	# 	# print("weighted correct", correct, weights)
	# 	# raise(Exception("stop"))
	# 	acc = correct.sum()/len(correct)
	# 	return acc    


class EncoderDecoderTransformer(nn.Module):
	"""from https://pytorch.org/tutorials/beginner/translation_transformer.html"""
	def __init__(self, num_encoder_layers, num_decoder_layers, d_model, nhead, src_vocab_size, tgt_vocab_size,
				 dim_feedforward=512, positional_encoding=True, norm_first = False, dropout=0.1):
		super(EncoderDecoderTransformer, self).__init__()
		self.transformer = Transformer(d_model=d_model,
									   nhead=nhead,
									   num_encoder_layers=num_encoder_layers,
									   num_decoder_layers=num_decoder_layers,
									   dim_feedforward=dim_feedforward,
									   dropout=dropout,
									   norm_first=norm_first,                                       
									   bias=True,
									   batch_first=True) # (batch, seq_len, d_model)
		if tgt_vocab_size != 2:
			raise ValueError("Only binary output supported.")
		self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
		self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
		self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, disable=(not positional_encoding))
		# Final layer for output decoder
		self.generator = nn.Linear(d_model, tgt_vocab_size) # [B, T, C]
		self.register_buffer("collapse_weight", torch.tensor([-1, 1], dtype=torch.float))
		self.init_output_layer()

	def init_output_layer(self):
		# Initialize weights to bias our output
		init_val = -0.1
		nn.init.constant_(self.generator.weight, init_val)
		nn.init.constant_(self.generator.bias, init_val)

	def forward(self, src, trg, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None, verbose=False):
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
			Tensor: (N, T) scalars that will sigmoid into binary values.
		"""
		src_emb = self.positional_encoding(self.src_tok_emb(src))
		tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
		outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
								src_padding_mask, tgt_padding_mask, memory_key_padding_mask) # [B, T, d_model]
		logits = self.generator(outs) # [B, T, C] with C=num_tokens
		# Trick for just binary sequences with no unique SOS, EOS: 
		# Just collapse the final dimension into C[1] - C[0]. Sigmoid 
		# will take care of the rest later.
		logits = torch.matmul(logits, self.collapse_weight) #[B, T]
		return logits

	def encode(self, src, src_mask=None):
		src_pos_emb = self.positional_encoding(self.src_tok_emb(src))
		return self.transformer.encoder(src_pos_emb, src_mask)

	def decode(self, tgt, memory, tgt_mask=None):
		tgt_pos_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
		return self.transformer.decoder(tgt_pos_emb, memory, tgt_mask)
	