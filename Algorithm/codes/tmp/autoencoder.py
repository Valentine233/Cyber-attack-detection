#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:22:47 2018

@author: longzhan
"""
import torch.nn as nn
import torch

class SimpleAutoEncoder(nn.Module):
    def __init__(self,dim):
        super(SimpleAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim, int(dim/4)),
            nn.Tanh(),
            nn.Linear(int(dim/4), int(dim/16)),
            #nn.Tanh(),
            #nn.Linear(int(dim/16), int(dim/64)),
            #nn.Tanh(),
            #nn.Linear(int(dim/64), int(dim/128)),
            #nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            #nn.Linear(int(dim/128), int(dim/64)),
            #nn.Tanh(),
            #nn.Linear(int(dim/64), int(dim/16)),
            #nn.Tanh(),
            nn.Linear(int(dim/16), int(dim/4)),
            nn.Tanh(),
            nn.Linear(int(dim/4), int(dim)),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

      
class VAE(nn.Module):
    def __init__(self,dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Sequential(
                nn.Linear(dim, int(dim/4)),
                nn.Tanh(),
                nn.Linear(int(dim/4), int(dim/16)),
                nn.Tanh()
                )                
        self.fc21 = nn.Linear(int(dim/16), 10)
        self.fc22 = nn.Linear(int(dim/16), 10)
        self.fc3 = nn.Sequential(
                nn.Linear(10, int(dim/16)),
                nn.Tanh(),
                nn.Linear(int(dim/16),int(dim/4)),
                nn.Tanh(),
                nn.Linear(int(dim/4),dim),
                nn.Sigmoid()
                )           
    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.fc3(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar        