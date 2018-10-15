#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:22:47 2018

@author: longzhan
"""
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self,input_dim,word_size,hidden_size):
        super(Autoencoder,self).__init__()
        self.input_dim = input_dim
        self.word_embedding = nn.Embedding(word_size,self.input_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm_encode = nn.LSTM(self.input_dim, hidden_size)
        self.lstm_decode = nn.LSTM(self.input_dim, hidden_size)
        self.lin = nn.Linear(hidden_size,word_size)

    def encode(self,words):
        #输入的是单词的编号序列
        input_ = self.word_embedding(words).view(-1,1,self.input_dim)
        _, h = self.lstm_encode(input_)
        return h 
    
    def decode(self,word,h):
        #逐个单词输入解码
        input_ = self.word_embedding(word).view(1,1,self.input_dim)
        input_ = self.dropout(input_)
        _, h = self.lstm_decode(input_,h)
        out = F.log_softmax(self.lin(h[0].view(1,-1)), dim=1)
        return out
    
        
class SimpleAutoEncoder(nn.Module):
    def __init__(self,dim):
        super(SimpleAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim, int(dim/4)),
            nn.Tanh(),
            nn.Linear(int(dim/4), int(dim/16)),
            nn.Tanh(),
            nn.Linear(int(dim/16), int(dim/64)),
            nn.Tanh(),
            nn.Linear(int(dim/64), int(dim/128)),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(dim/128), int(dim/64)),
            nn.Tanh(),
            nn.Linear(int(dim/64), int(dim/16)),
            nn.Tanh(),
            nn.Linear(int(dim/16), int(dim/4)),
            nn.Tanh(),
            nn.Linear(int(dim/4), int(dim)),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
