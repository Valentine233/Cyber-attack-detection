import sys
sys.path.append("..")
from data_preprocessing.data_preprocessing import *
import copy
import numpy as np 
from tqdm import tqdm
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
# char
# train_sents, test_sents, train_labels, test_labels = getSent(train_fn, test_fn, train_label_fn, test_label_fn)
# data_processor = One_hot(train_sents, test_sents, train_labels, test_labels, batch_size)
# bpe
train_sents, test_sents, train_labels, test_labels = getSent(train_bpe_fn, test_bpe_fn, train_label_fn, test_label_fn)
data_processor = One_hot(train_sents, test_sents, train_labels, test_labels, batch_size, mode='bpe')

all_sents = data_processor.all_features
all_labels = data_processor.all_y
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
state = (to_var(torch.zeros(num_layers, 1, hidden_size), cuda_num), to_var(torch.zeros(num_layers, 1, hidden_size), cuda_num))

# test
labels = [int(l) for l in all_labels]
np.save(label_npy, np.asarray(labels))

rnnlm.eval()
ae.eval()
ae_features = []
for step in tqdm(range(len(all_sents))):
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
	outputs, h = rnnlm(padded_input_sents, lens, state) # len+1 for adding <sos>
	rnnlm_feature = torch.squeeze(h, 0)
	encode, decode = ae(rnnlm_feature)
	ae_feature = torch.squeeze(encode)
	ae_features.append(ae_feature.cpu().detach().numpy())


np.save(ae_feature_npy, np.asarray(ae_features))