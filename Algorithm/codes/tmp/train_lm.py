# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:28:56 2018

@author: win 10
"""

from data import bi_gram,bpe_gram
from lm import RNNLM
from torch import optim
import torch
import numpy as np
from utils import to_var
import pickle
import copy
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_


data_fn = 'data/CSIC2010/request_train_bpe.txt'
batch_size = 100
learning_rate = 0.01
decay_rate = 0.2
epoches = 30
embed_size = 100
hidden_size = 100
num_layers = 2
clip = 50

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def getSent():
    sentences = []
    with open(data_fn) as f:
        for line in f:
            sentences.append(line.strip())
    return sentences

def getData(lm_name):
    sentences = getSent()
    if lm_name == "bigram":
        data = bi_gram(sentences)
    if lm_name == "bpe":
        data = bpe_gram(sentences)
    else:
        return None
    data.build()
    data.makeFeatures()    
    with open("data_"+lm_name+".pkl","wb") as f:
        pickle.dump(data, f)
    return data

def prepare(data, sentences):
    input_sentences2index = []
    target_sentences2index = []
    for sentence in sentences:
        input_sentence2index = []
        target_sentence2index = []
        for token in sentence.split():
            if token in data.word2index:
                input_sentence2index.append(data.word2index[token])
            else:
                input_sentence2index.append(data.word2index["<unk>"])
        target_sentence2index = copy.deepcopy(input_sentence2index)
        target_sentence2index.pop(0)
        target_sentence2index.append(data.word2index["<eos>"])
        input_sentences2index.append(input_sentence2index)
        target_sentences2index.append(target_sentence2index)
        
    input_sents = [torch.LongTensor(s) for s in input_sentences2index]
    target_sents = [torch.LongTensor(s) for s in target_sentences2index]
    padded_input_sents = to_var(pad_sequence(input_sents, batch_first=True, padding_value=0))
    padded_target_sents = to_var(pad_sequence(target_sents, batch_first=True, padding_value=0))
    sentences_lengths = [len(s) for s in input_sentences2index]
    
    m = []
    for i, seq in enumerate(padded_input_sents):
        m.append([])
        for index in seq:
            if index == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    mask = to_var(torch.ByteTensor(m))
    
    return padded_input_sents, padded_target_sents, sentences_lengths, mask
    
    

if __name__ == "__main__":
    
    data = getData("bpe")
    model = RNNLM(embed_size,hidden_size,num_layers,len(data.word2index))
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epoches):
        optimizer = lr_decay(optimizer, epoch, decay_rate, learning_rate)
        np.random.shuffle(data.sentences)
        batch_num = int(len(data.sentences)/batch_size)
        for i in range(batch_num):
            print_losses = []
            loss = 0
            n_totals = 0
            optimizer.zero_grad()
            sentences = copy.deepcopy(data.sentences[i*batch_size:(i+1)*batch_size])
            padded_input_sents, padded_target_sents, sentences_lengths, mask = prepare(data,sentences)
            outputs,h = model(padded_input_sents, sentences_lengths)
            for t in range(outputs.size(0)):
                mask_loss, nTotal = maskNLLLoss(outputs[t], padded_target_sents[t], mask[t])
                loss += mask_loss
                n_totals += nTotal
                print_losses.append(mask_loss.item() * nTotal)
            print("epoch:",epoch,"loss:",sum(print_losses) / n_totals)
            loss.backward()
            _ = clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
        if len(data.sentences) % batch_size != 0:
            print_losses = []
            loss = 0
            n_totals = 0
            optimizer.zero_grad()
            sentences = copy.deepcopy(data.sentences[(i+1)*batch_size:])
            padded_input_sents, padded_target_sents, sentences_lengths, mask = prepare(data,sentences)
            outputs,h = model(padded_input_sents, sentences_lengths)
            for t in range(outputs.size(0)):
                mask_loss, nTotal = maskNLLLoss(outputs[t], padded_target_sents[t], mask[t])
                loss += mask_loss
                n_totals += nTotal
                print_losses.append(mask_loss.item() * nTotal)
            print("epoch:",epoch,"loss:",sum(print_losses) / n_totals)
            loss.backward()
            _ = clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        print("epoch:",epoch,'done')
        torch.save(model.state_dict(), "model/model.params.lm")
    
    
    
    
    
    
    
    
