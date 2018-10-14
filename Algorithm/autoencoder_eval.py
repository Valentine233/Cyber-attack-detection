#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:07:28 2018

@author: longzhan
"""

from autoencoder import SimpleAutoEncoder
from data import bi_gram
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np
from torch.autograd import Variable
rng = np.random.RandomState(42)


raw_fn1 = 'data/CSIC2010/normalTrafficTraining.txt'
raw_fn2 = 'data/CSIC2010/normalTrafficTest.txt'
raw_fn3 = 'data/CSIC2010/anomalousTrafficTest.txt'

sent_fn = 'data/CSIC2010/log.txt'
train_fn = 'data/CSIC2010/output.txt'

def getSent(f1,f2,f3,f_out):
    train_sentences = []
    test_sentences_p = []
    test_sentences_n = []
    with open(f1,'r') as f:
        first_line = True
        for line in f:
            if first_line == True and line.strip() != "":
                train_sentences.append(line.strip())
                first_line = False
            if line.strip() == "":
                first_line = True
    with open(f2,'r') as f:
        first_line = True
        for line in f:
            if first_line == True and line.strip() != "":
                test_sentences_p.append(line.strip())
                first_line = False
            if line.strip() == "":
                first_line = True
    with open(f3,'r') as f:
        first_line = True
        for line in f:
            if first_line == True and line.strip() != "":
                test_sentences_n.append(line.strip())
                first_line = False
            if line.strip() == "":
                first_line = True
    return train_sentences,test_sentences_p,test_sentences_n

def get_MLP_feature(data):
    model = SimpleAutoEncoder(data.features.shape[1])
    model.load_state_dict(torch.load('model/model.params.MLP'))
    model.eval()
    input_ = Variable(torch.from_numpy(data.features).type(torch.FloatTensor).view(data.features.shape[0],-1))
    X = model(input_)[0].data.numpy()
    return X
        
def evaluate(X,gold):
    clf = IsolationForest(max_samples=2000,n_estimators=500,
                      random_state=rng,contamination=0.2)
    clf.fit(X)
    pred=list(clf.predict(X))
    print (classification_report(gold, pred))
    
    
if __name__ == '__main__':
    #load data
    train_sentences,test_sentences_p,test_sentences_n = getSent(raw_fn1,raw_fn2,raw_fn3,sent_fn)
    data = bi_gram(train_sentences+test_sentences_p+test_sentences_n,shuffle=False)
    data.build()
    data.makeFeatures(test=True)
    
    #make gold tag
    gold = []
    for i in range(len(train_sentences+test_sentences_p)):
        gold.append(1)
    for i in range(len(test_sentences_n)):
        gold.append(-1)
    
    X = get_MLP_feature(data)
    evaluate(X,gold)
