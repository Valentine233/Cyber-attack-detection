# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:20:16 2018

@author: win 10
"""
import torch
import torch.nn as nn

class charCNN(nn.Module):
     def __init__(self,input_size,input_dim):
         super(charCNN,self).__init__()
         self.input_dim = input_dim
         self.embed = nn.Embedding(input_size,input_dim)
         self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 100, kernel_size=3, stride=1),
            nn.ReLU()
            )
         self.conv2 = nn.Sequential(
            nn.Conv1d(input_dim, 100, kernel_size=4, stride=1),
            nn.ReLU()
            )
         self.conv3 = nn.Sequential(
            nn.Conv1d(input_dim, 100, kernel_size=5, stride=1),
            nn.ReLU()
            )
         self.lin = nn.Sequential(
                 nn.Dropout(),
                 nn.Linear(300,2)
                 )
         
     def forward(self,input_seq):
         input_ = self.embed(input_seq).view(1,self.input_dim,-1)
         c1 = self.conv1(input_).view(100,-1) #out:(Cout,Lout)
         c2 = self.conv2(input_).view(100,-1)
         c3 = self.conv3(input_).view(100,-1)
         c1,_ = torch.max(c1,1)
         c2,_ = torch.max(c2,1)
         c3,_ = torch.max(c3,1)
         c = torch.cat([c1,c2,c3]).view(1,-1)
         out = self.lin(c)
         return out