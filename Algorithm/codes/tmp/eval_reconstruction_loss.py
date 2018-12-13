# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:36:34 2018

@author: win 10
"""

from utils import to_var
from autoencoder import SimpleAutoEncoder
import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt


test_fn = 'data/CSIC2010/request_test.txt'
test_label_fn = "data/CSIC2010/label_test.txt"

def getPrecision(pred,gold,label):
    total = 0
    correct = 0
    for i in range(len(pred)):
        if pred[i] == label:
            total += 1
            if gold[i] == pred[i]:
                correct += 1
    return correct/total

def getRecall(pred,gold,label):
    total = 0
    correct = 0
    for i in range(len(pred)):
        if gold[i] == label:
            total += 1
            if pred[i] == gold[i]:
                correct += 1
    return correct/total

def getAccuracy(pred,gold):
    total = len(pred)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gold[i]:
            correct += 1
    return correct/total
    
def mean(l):
    return sum(l)/len(l)
    
def getSent(data_fn):
    sentences = []
    with open(data_fn) as f:
        for line in f:
            sentences.append(line.strip())
    return sentences

if __name__ == "__main__":

    test_sentences = getSent(test_fn)
    test_labels = getSent(test_label_fn)
    test_labels = [int(i) for i in test_labels]
    
    with open("data.pkl",'rb') as f:
        data = pickle.load(f)
    data.makeFeatures(fit_obj=test_sentences)
    
    model = SimpleAutoEncoder(data.features.shape[1])
    model.load_state_dict(torch.load('model/model.params.MLP'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    loss_func = nn.MSELoss()
    
    nor_loss = []
    abr_loss = []
    
    for i in range(len(test_sentences)):
        input_ = to_var(torch.from_numpy(data.features[i]).type(torch.FloatTensor).view(1,-1))
        encoded, decoded = model(input_)
        loss = loss_func(decoded, input_)
        if test_labels[i] == 1:
            abr_loss.append(loss.item())
        else:
            nor_loss.append(loss.item())
    
    print(mean(nor_loss))
    print(mean(abr_loss))
    
    precision = []
    recall = []
    accuracy = []
    f1 = []
    seuils = np.linspace(8e-6, 1.18e-5, 100)
    loss_total = abr_loss + nor_loss
    for seuil in seuils:
        pred = []
        for l in loss_total:
            if l > seuil:
                pred.append(1)
            else:
                pred.append(0)
        precision.append(getPrecision(pred,test_labels,1))
        recall.append(getRecall(pred,test_labels,1))
        accuracy.append(getAccuracy(pred,test_labels))
        f1.append(2*precision[-1]*recall[-1]/(precision[-1]+recall[-1]))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(recall,precision)
    plt.figure()
    plt.xlabel('seuil')
    plt.ylabel('accuracy')
    plt.plot(seuils,accuracy)
    plt.figure()
    plt.xlabel('seuil')
    plt.ylabel('f1')
    plt.plot(seuils,f1)

    
    
    