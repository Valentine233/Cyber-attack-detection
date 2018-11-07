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
from networks.autoencoder import SimpleAutoEncoder
from utils import *
from config import *

# data preprocessing
# rnnlm_features = torch.load(rnnlm_feature_pt)
# labels = torch.load(label_pt)
# data_processor = MakeBatches(feature_pt, label_pt, batch_size)
# trainlen = data_processor.train_len()
# testlen = data_processor.test_len()
# step_num = int(trainlen / batch_size)
# test_step_num = int(testlen / batch_size)

sentences, labels = getSent(anor_test, nor_test, nor_train, request_file, label_file)
data_processor = One_hot(sentences, labels, batch_size)
trainlen = data_processor.train_len()
testlen = data_processor.test_len()
vocab_size = data_processor.voc_len()
step_num = int(trainlen / batch_size)
test_step_num = int(testlen / batch_size)


# create net
rnnlm = RNNLM(embed_size, hidden_size, num_layers, vocab_size)
ae = SimpleAutoEncoder(hidden_size)
if torch.cuda.is_available():
    rnnlm.cuda(cuda_num)
    ae.cuda(cuda_num)
rnnlm.load_state_dict(torch.load(rnnlm_model_file))

# create criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate)

# log file
log = open(ae_log_file,'w')

# initiate
state = (to_var(torch.zeros(num_layers, batch_size, hidden_size), cuda_num), to_var(torch.zeros(num_layers, batch_size, hidden_size), cuda_num))

# train
rnnlm.train()
ae.train()
for epoch in range(epoch_num):
    for step in range(step_num):
        step_loss = []

        # mini-batch
        sents, labels, lens = data_processor.getBatch(step, train=True)

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
        # features: (batch, hidden_size)
        features = torch.squeeze(h, 0)
        
        # forward pass
        encode, decode = ae(features.detach())
        loss = criterion(decode, features)
        step_loss.append(loss.data)
        log.write("[epoch%d, step%d]: loss %f\n" % (epoch+1, step+1, loss.data))

        # log
        if step == step_num-1:
            print("[epoch%d]: avg loss %f" % (epoch+1, np.mean(step_loss)))
            log.write("\n[epoch%d]: avg loss %f\n\n" % (epoch+1, np.mean(step_loss)))

        # backward and optimize
        ae.zero_grad()
        loss.backward()
        clip_grad_norm_(ae.parameters(), 0.5)
        optimizer.step()


# save model
torch.save(ae.state_dict(), ae_model_file)


# test
rnnlm.eval()
ae.eval()
for step in range(test_step_num):
    step_loss = []

    # mini-batch
    sents, labels, lens = data_processor.getBatch(step, train=True)

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
    # features: (batch, hidden_size)
    features = torch.squeeze(h, 0)
        
    # forward pass
    encode, decode = ae(features.detach())
    loss = criterion(decode, features)
    step_loss.append(loss.data)
    log.write("[TEST step%d]: loss %f\n" % (step+1, loss.data))

    # log
    if step % 100 == 0:
        print("[TEST step%d]: avg loss %f" % (step+1, np.mean(step_loss)))

    if step == test_step_num-1:
        print("[TEST]: avg loss %f" % (np.mean(step_loss)))
        log.write("\n[TEST]: avg loss %f\n\n" % (np.mean(step_loss)))

log.close()












