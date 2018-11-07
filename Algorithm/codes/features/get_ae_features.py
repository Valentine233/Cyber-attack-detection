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
from networks.autoencoder import SimpleAutoEncoder
from utils import *
from config import *


# data preprocessing
all_sentences, all_labels = getSent(anor_test, nor_test, nor_train, request_file, label_file)
data_processor = GetFeatures(all_sentences)
all_sents = data_processor.features

vocab_size = data_processor.voc_len()

# create net
rnnlm = RNNLM(embed_size, hidden_size, num_layers, vocab_size)
ae = SimpleAutoEncoder(hidden_size)
if torch.cuda.is_available():
	rnnlm.cuda(cuda_num)
	ae.cuda(cuda_num)
rnnlm.load_state_dict(torch.load(rnnlm_model_file))
ae.load_state_dict(torch.load(ae_model_file))

# initiate
state = (to_var(torch.zeros(num_layers, batch_size, hidden_size), cuda_num), to_var(torch.zeros(num_layers, batch_size, hidden_size), cuda_num))

# test
labels = [int(l) for l in all_labels]
np.save(label_npy, np.asarray(labels))

rnnlm.eval()
ae.eval()
ae_features = []
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
		
	padded_input_sents = pad_sequence(input_sents, batch_first=True, padding_value=0)
	padded_target_sents = pad_sequence(target_sents, batch_first=True, padding_value=0)

	if torch.cuda.is_available():
		padded_input_sents = to_var(padded_input_sents, cuda_num)
		padded_target_sents = to_var(padded_target_sents, cuda_num)

	# forward pass
	# outputs: (batch_size, seq_len, vocab_size)
	# h: (num_layers*num_directions, batch, hidden_size)
	outputs, h = rnnlm(padded_input_sents, [l+1 for l in lens], state) # len+1 for adding <sos>
	rnnlm_feature = torch.squeeze(h, 0)
	encode, decode = ae(rnnlm_feature)
	ae_feature = torch.squeeze(encode)
	ae_features.append(ae_feature.cpu().detach().numpy())


np.save(ae_feature_npy, np.asarray(ae_features))