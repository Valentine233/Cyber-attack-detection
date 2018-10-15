#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:26:21 2018

@author: longzhan
"""

from data import supervised_bpe
from classifier import addMLP
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

data_fn = 'data/CSIC2010/output.txt'
batch_size = 100
word_dim = 32
learning_rate = 0.01
epochs = 30
 
def getSent(fn):
    sentences = []
    with open(fn,'r') as f:
        for line in f:
            if len(sentences) < 88000:
                sentences.append([line.strip(),0])
            else:
                sentences.append([line.strip(),1])
        return sentences

def prepare_seq(sentence,data):
    sentence2index = []
    for word in sentence.split():
        sentence2index.append(data.word2index[word])
    return Variable(torch.LongTensor(sentence2index))
   
if __name__ == '__main__':
    sentences = getSent(data_fn)
    data = supervised_bpe(sentences)
    data.build()
    data.makeBatch(batch_size)
    model = addMLP(data.word_count,word_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in data.batch_sentences:
            loss = 0
            model.zero_grad()
            for sentence,label in batch:
                input_ = prepare_seq(sentence,data)
                label = Variable(torch.LongTensor([int(label)]))
                out = model(input_)
                loss += loss_func(out,label)
            print("epoch:",epoch,"loss:",loss.data[0]/len(batch))
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,'done')
        torch.save(model.state_dict(), "model/model.params.addMLP")
    
    
    model.eval()
    #make gold tag
    gold = []
    pred = []
    for sentence, label in data.test_sentences:
        input_ = prepare_seq(sentence,data)
        out = model(input_)
        topv, topi = out.topk(1)
        pred_label = topi.squeeze().detach().data[0]
        pred.append(int(pred_label))
        gold.append(int(label))
    print (classification_report(gold, pred))
