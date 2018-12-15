import sys
sys.path.append("..")
from data_preprocessing.data_preprocessing import *
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from networks.rnnlm import RNNLM
from utils import *
from config import *

# data preprocessing
train_sents, test_sents, train_labels, test_labels = getSent(train_fn, test_fn, train_label_fn, test_label_fn)
data_processor = One_hot(train_sents, test_sents, train_labels, test_labels, 1)
all_sents = data_processor.all_features
all_labels = data_processor.all_y

vocab_size = data_processor.voc_len()

# create net
net = RNNLM(embed_size, hidden_size, num_layers, vocab_size)
if torch.cuda.is_available():
	net.cuda(cuda_num)
net.load_state_dict(torch.load(rnnlm_model_file))

# initiate
state = (to_var(torch.zeros(num_layers, 1, hidden_size), cuda_num), to_var(torch.zeros(num_layers, 1, hidden_size), cuda_num))

# test
labels = [int(l) for l in all_labels]
np.save(label_npy, np.asarray(labels))

net.eval()
rnnlm_features = []
print("len(all_sents)", len(all_sents))
for step in range(len(all_sents)):
	print("step: %d" % step)
	# get mini-batch
	sents, lens = [all_sents[step]], [len(all_sents[step])]
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
	rnnlm_feature = torch.squeeze(h)
	rnnlm_features.append(rnnlm_feature.cpu().detach().numpy())

np.save(rnnlm_feature_npy, np.asarray(rnnlm_features))














