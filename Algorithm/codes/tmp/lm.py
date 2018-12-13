# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:28:51 2018

@author: win 10
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class RNNLM(nn.Module):
	def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
		super(RNNLM, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.)
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0.0)

	def forward(self, input, lengths):
		embed_input = self.embed(input)
		lens, indices = torch.sort(embed_input.data.new(lengths).long(), 0 ,True)
		embed_input = embed_input[indices]
		lstm_input = pack_padded_sequence(embed_input, lens.tolist(), batch_first=True)
		output, (h, c) = self.lstm(lstm_input)
		output = pad_packed_sequence(output, batch_first=True)[0]
		_, _indices = torch.sort(indices, 0)
		ordered_output = output[_indices] # batch in dim-0
		h, c = h[:, _indices, :], c[:, _indices, :] # batch in dim-1
		final_output = F.softmax(self.linear(ordered_output),dim=2) # (batch, seqlen, vocab_size)

		return final_output,h