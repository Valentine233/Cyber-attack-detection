import sys
sys.path.append("..")
from data_preprocessing.data_preprocessing import *
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.nn.utils import clip_grad_norm_
from networks.rnnlm import RNNLM
from utils import *
from config import *
import os

if not os.path.exists(model_root):
	os.makedirs(model_root)

# data preprocessing
# char
# train_sents, test_sents, train_labels, test_labels = getSent(train_fn, test_fn, train_label_fn, test_label_fn)
# data_processor = One_hot(train_sents, test_sents, train_labels, test_labels, batch_size)
# bpe
train_sents, test_sents, train_labels, test_labels = getSent(train_bpe_fn, test_bpe_fn, train_label_fn, test_label_fn)
data_processor = One_hot(train_sents, test_sents, train_labels, test_labels, batch_size, mode='bpe')

trainlen = data_processor.all_len()
testlen = data_processor.test_len()
vocab_size = data_processor.voc_len()
step_num = int(trainlen / batch_size)
test_step_num = int(testlen / batch_size)

# create net
net = RNNLM(embed_size, hidden_size, num_layers, vocab_size)
if torch.cuda.is_available():
	net.cuda(cuda_num)

# create criterion and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# log file
log = open(rnnlm_log_file,'w+')

# initiate
state = (to_var(torch.zeros(num_layers, batch_size, hidden_size), cuda_num), to_var(torch.zeros(num_layers, batch_size, hidden_size), cuda_num))

# train
net.train()

for epoch in range(epoch_num):
	step_loss = []
	for step in range(step_num):
		# get mini-batch
		sents, labels, lens = data_processor.getAllBatch(step, train=True)

		input_sents = copy.deepcopy(sents)
		target_sents = copy.deepcopy(sents)

		for s in input_sents:
			s.insert(0, data_processor.unigram2index['<sos>'])
		input_sents = [torch.LongTensor(s) for s in input_sents]

		for s in target_sents:
			s.append(data_processor.unigram2index['<eos>'])
		target_sents = [torch.LongTensor(s) for s in target_sents]

		lens = [l+1 for l in lens]
		
		padded_input_sents = pad(input_sents, lens, padding_value=0)
		padded_target_sents = pad(target_sents, lens, padding_value=0)

		if torch.cuda.is_available():
			padded_input_sents = to_var(padded_input_sents, cuda_num)
			padded_target_sents = to_var(padded_target_sents, cuda_num)

		# forward pass
		# outputs: (batch_size, seq_len, vocab_size)
		# h: (num_layers*num_directions, batch, hidden_size)
		outputs, h = net(padded_input_sents, lens, state) # len+1 for adding <sos>
		predicted_sent = torch.transpose(outputs, 1, 2)
		loss = criterion(predicted_sent, padded_target_sents)
		step_loss.append(loss.data)
		log.write("[epoch%d, step%d]: loss %f\n" % (epoch+1, step+1, loss.data))

		# log
		if step == step_num-1:
			print("[epoch%d]: avg loss %f" % (epoch+1, np.mean(step_loss)))
			log.write("\n[epoch%d]: avg loss %f\n\n" % (epoch+1, np.mean(step_loss)))

		# backward and optimize
		net.zero_grad()
		loss.backward()
		clip_grad_norm_(net.parameters(), 0.5)
		optimizer.step()


# save model
torch.save(net.state_dict(), rnnlm_model_file)


# test
# step_loss = []
# net.eval()
# for step in range(test_step_num):
# 	# get mini-batch
# 	sents, labels, lens = data_processor.getBatch(step, train=False)
# 	input_sents = copy.deepcopy(sents)
# 	target_sents = copy.deepcopy(sents)

# 	for s in input_sents:
# 		s.insert(0, data_processor.unigram2index['<sos>'])
# 	input_sents = [torch.LongTensor(s) for s in input_sents]

# 	for s in target_sents:
# 		s.append(data_processor.unigram2index['<eos>'])
# 	target_sents = [torch.LongTensor(s) for s in target_sents]
		
# 	padded_input_sents = pad_sequence(input_sents, batch_first=True, padding_value=0)
# 	padded_target_sents = pad_sequence(target_sents, batch_first=True, padding_value=0)

# 	if torch.cuda.is_available():
# 		padded_input_sents = to_var(padded_input_sents, cuda_num)
# 		padded_target_sents = to_var(padded_target_sents, cuda_num)

# 	# forward pass
# 	# outputs: (batch_size, seq_len, vocab_size)
# 	# h: (num_layers*num_directions, batch, hidden_size)
# 	outputs, h = net(padded_input_sents, [l+1 for l in lens], state) # len+1 for adding <sos>
# 	predicted_sent = torch.transpose(outputs, 1, 2)
# 	loss = criterion(predicted_sent, padded_target_sents)

# 	step_loss.append(loss.data)
# 	log.write("[TEST step%d]: loss %f\n" % (step+1, loss.data))

# 	# log
# 	if step % 100 == 0:
# 		print("[TEST step%d]: avg loss %f" % (step+1, np.mean(step_loss)))

# 	if step == test_step_num-1:
# 		print("[TEST]: avg loss %f" % (np.mean(step_loss)))
# 		log.write("\n[TEST]: avg loss %f\n\n" % (np.mean(step_loss)))

log.close()












