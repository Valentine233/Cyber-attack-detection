#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:31:08 2018

@author: longzhan
"""

#import pandas as pd
from data import bi_gram
from autoencoder import SimpleAutoEncoder
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch

SOS_TOKEN = 0
EOS_TOKEN = 1

raw_fn1 = 'data/CSIC2010/normalTrafficTraining.txt'
raw_fn2 = 'data/CSIC2010/normalTrafficTest.txt'
raw_fn3 = 'data/CSIC2010/anomalousTrafficTest.txt'

sent_fn = 'data/CSIC2010/log.txt'
#data = pd.read_excel("data/https_bp_hash.xlsx",header=None)
#统计用户访问次数的代码
#data[0].value_counts()[data[0].value_counts()>100]

input_dim = 128
hidden_dim = 16
learning_rate = 0.01
decay_rate = 0.05
epoches = 10
batch_size = 100

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def getSent(f1,f2,f3,f_out):
    sentences = []
    with open(f1,'r') as f:
        first_line = True
        for line in f:
            if first_line == True and line.strip() != "":
                sentences.append(line.strip())
                first_line = False
            if line.strip() == "":
                first_line = True
    with open(f2,'r') as f:
        first_line = True
        for line in f:
            if first_line == True and line.strip() != "":
                sentences.append(line.strip())
                first_line = False
            if line.strip() == "":
                first_line = True
    with open(f3,'r') as f:
        first_line = True
        for line in f:
            if first_line == True and line.strip() != "":
                sentences.append(line.strip())
                first_line = False
            if line.strip() == "":
                first_line = True
    with open(f_out,'w') as f:
        f.write('\n'.join(sentences))
    return sentences

def load(fn):
    sentences = []
    with open(fn,'r') as f:
        for line in f:
            sentences.append(line)
    return sentences

def prepare_seq(sentence,data):
    sentence2index = []
    for word in sentence.split():
        sentence2index.append(data.word2index[word])
    sentence2index.append(EOS_TOKEN)
    return Variable(torch.LongTensor(sentence2index))


def trainMLP():
    sentences = getSent(raw_fn1,raw_fn2,raw_fn3,sent_fn)
    data = bi_gram(sentences)
    data.build()
    data.makeFeatures()
    
    model = SimpleAutoEncoder(data.features.shape[1])
    model.zero_grad()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate/batch_size)
    loss_func = nn.MSELoss()
    loss = 0
    
    for epoch in range(epoches):
        for i in range(len(data.sentences_no_repeat)):
            if (i+1) % batch_size == 0:
                print("epoch:",epoch,"loss:",loss.data[0]/batch_size)
                loss.backward()
                optimizer.step()
                model.zero_grad()
                loss = 0
            else:
                input_ = Variable(torch.from_numpy(data.features[4]).type(torch.FloatTensor).view(1,-1))
                encoded, decoded = model(input_)
                loss += loss_func(decoded, input_)
                if i == len(data.sentences_no_repeat) - 1:
                    print("epoch:",epoch,"loss:",loss.data[0]/batch_size)
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    loss = 0
        print("epoch:",epoch,'done')
        torch.save(model.state_dict(), "model/model.params.MLP")
        
if __name__ == "__main__":
    trainMLP()    
