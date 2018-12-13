#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:31:08 2018

@author: longzhan
"""

#import pandas as pd
from data import bi_gram,bpe_gram
from autoencoder import SimpleAutoEncoder,VAE
from torch import optim
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from utils import to_var
import pickle

bigram_data_fn = 'data/CSIC2010/request_train.txt'
bpe_data_fn = 'data/CSIC2010/request_train_bpe.txt'
batch_size = 100
learning_rate = 0.01/batch_size #parm for MLP
#learning_rate = 0.001 #parm for VAE
decay_rate = 0.2
epoches = 30


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def getSent(data_fn):
    sentences = []
    with open(data_fn) as f:
        for line in f:
            sentences.append(line.strip())
    return sentences

def VAE_loss(recon_x, x, mu, logvar, beta=1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta*KLD

def getData(lm): 
    if lm == "bigram":
        sentences = getSent(bigram_data_fn)
        data = bi_gram(sentences)
    if lm == "bpe":
        sentences = getSent(bpe_data_fn)
        data = bpe_gram(sentences)
    else:
        return None
    data.build()
    data.makeFeatures()    
    with open("data_"+lm+".pkl","wb") as f:
        pickle.dump(data, f)
    return data

def train(model_name,data):   
    if model_name == 'MLP':
        model = SimpleAutoEncoder(data.features.shape[1])
        loss_func = nn.MSELoss()
    elif model_name == 'VAE':
        model = VAE(data.features.shape[1])
    else:
        print("no such model")
        return
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epoches):
        optimizer = lr_decay(optimizer, epoch, decay_rate, learning_rate)
        np.random.shuffle(data.features)
        batch_num = int(len(data.sentences)/batch_size)
        for i in range(batch_num):
            optimizer.zero_grad()
            input_ = to_var(torch.from_numpy(data.features[i*batch_size:(i+1)*batch_size]).type(torch.FloatTensor))
            if model_name == 'MLP':
                encoded, decoded = model(input_)
                loss = loss_func(decoded, input_)
            else:
                recon_batch, mu, logvar = model(input_)
                loss = VAE_loss(recon_batch, input_, mu, logvar)
            print("epoch:",epoch,"loss:",loss.item()/batch_size)
            loss.backward()
            optimizer.step()
        #最后一个batch
        if len(data.sentences) % batch_size != 0:
            optimizer.zero_grad()
            input_ = to_var(torch.from_numpy(data.features[(i+1)*batch_size:]).type(torch.FloatTensor))
            if model_name == 'MLP':
                    encoded, decoded = model(input_)
                    loss = loss_func(decoded, input_)
            else:
                recon_batch, mu, logvar = model(input_)
                loss = VAE_loss(recon_batch, input_, mu, logvar)
            print("epoch:",epoch,"loss:",loss.item()/batch_size)
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,'done')
        torch.save(model.state_dict(), "model/model.params."+model_name)
        
if __name__ == "__main__":
    data = getData("bpe")
    #MLP or VAE
    train('MLP',data)    
    
               
                
                
            
    
        
                
    
    
    
