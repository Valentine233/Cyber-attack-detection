#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:04:34 2018

@author: longzhan
"""
import random
import numpy as np
SOS_TOKEN = 0
EOS_TOKEN = 1

class bi_gram:
    def __init__(self,sentences,shuffle=False):
        self.sentences = sentences
        if shuffle:
            random.shuffle(self.sentences)
        self.sentences_no_repeat = list(set(self.sentences))
        self.unigram2index = {'<unk>':0}
        self.index2unigram = {0:'<unk>'}
        self.bigram_count = {}
        self.bigram2index = {'<unk><unk>':0}
        self.index2bigram = {0:'<unk><unk>'}
        self.features = np.zeros(1)
        self.batch_sentences = []
        
    def build(self,seuil=10,truncating=True):
        for sentence in self.sentences:
            for char in sentence:
                if char not in self.unigram2index:
                    self.unigram2index[char] = len(self.unigram2index)
            bisent = [[sentence[i],sentence[i+1]] for i in range(len(sentence)-1)]   
            for term in bisent:
                bichar = term[0] + term[1]
                if bichar in self.bigram_count:
                    self.bigram_count[bichar] += 1
                else:
                    self.bigram_count[bichar] = 1
        if truncating == True:
            for bichar in self.bigram_count:
                if self.bigram_count[bichar] >= seuil:
                    self.bigram2index[bichar] = len(self.bigram2index)
            print("Number of bichar truncated:",len(self.bigram_count)-len(self.bigram2index))
        self.index2unigram = {v: k for k, v in self.unigram2index.items()}
        self.index2bigram = {v: k for k, v in self.bigram2index.items()}
    
    def makeFeatures(self,fit_obj=None):
        if fit_obj != None:
            s = fit_obj
        else:
            s = self.sentences
        self.features = np.zeros((len(s),len(self.unigram2index)+len(self.bigram2index)))
        for i,sentence in enumerate(s):
            for char in sentence:
                if char not in self.unigram2index:
                    self.features[i][0] += 1/len(sentence)
                else:
                    self.features[i][self.unigram2index[char]] += 1/len(sentence)
            bisent = [[sentence[i],sentence[i+1]] for i in range(len(sentence)-1)]  
            for term in bisent:
                bichar = term[0] + term[1]
                if bichar not in self.bigram2index:
                    self.features[i][0+len(self.unigram2index)] += 1/len(sentence)
                else:
                    self.features[i][self.bigram2index[bichar]+len(self.unigram2index)] += 1/len(sentence)        



class bpe_gram:
    def __init__(self,sentences,shuffle=False):
        self.sentences = sentences
        if shuffle:
            random.shuffle(self.sentences)
        self.sentences_no_repeat = list(set(self.sentences))
        self.word2index = {'<pad>':0,'<unk>':1,'<eos>':2}
        self.word_count = {}
        self.features = np.zeros(1)
    
    def build(self,seuil=5,truncating=True):
        for sentence in self.sentences:
            words = sentence.split()
            for word in words:
                if word not in self.word_count:
                    self.word_count[word] = 1
                else:
                    self.word_count[word] += 1
        
        if truncating == True:
            for word in self.word_count:
                if self.word_count[word] > seuil:
                    self.word2index[word] = len(self.word2index)
            print("Number of subword truncated:",len(self.word_count)-len(self.word2index))
    
    def makeFeatures(self,fit_obj=None):
        if fit_obj != None:
            s = fit_obj
        else:
            s = self.sentences
        self.features = np.zeros((len(s),len(self.word2index)))
        for i,sentence in enumerate(s):
            for word in sentence.split():
                if word not in self.word2index:
                    self.features[i][1] += 1/len(sentence.split())
                else:
                    self.features[i][self.word2index[word]] += 1/len(sentence.split())
                    
           
class supervised_bpe:
    def __init__(self,sentences,shuffle=True):
        self.sentences = sentences
        if shuffle:
            random.shuffle(self.sentences)
        sentences_tmp1 = [sentence[0]+'||||'+str(sentence[1]) for sentence in sentences]
        sentences_tmp2 = list(set(sentences_tmp1))
        self.sentences_no_repeat = [sentence.split('||||') for sentence in sentences_tmp2]
        self.word2index = {"<unk>":0}
        self.index2word = {0:"<unk>"}
        self.word_count = 1
        self.batch_sentences = []
        self.train_sentences = self.sentences_no_repeat[:int(0.8*len(self.sentences_no_repeat))]
        self.test_sentences = self.sentences_no_repeat[int(0.8*len(self.sentences_no_repeat)):]
    
    def build(self):
        for sentence in self.sentences_no_repeat:
            sentence = sentence[0].split()
            for token in sentence:
                if token not in self.word2index:
                    self.word2index[token] = self.word_count
                    self.index2word[self.word_count] = token
                    self.word_count += 1
    
    def makeBatch(self,batch_size):
        self.batch_sentences = []
        batch = []
        for i,sentence in enumerate(self.train_sentences):
            if (i+1)%batch_size == 0:
                self.batch_sentences.append(batch)
                batch = []
            else:
                batch.append(sentence)
        if len(batch) > 0:
            self.batch_sentences.append(batch)    
    
class char: #用于训练有监督的char_CNN
    def __init__(self,sentences,shuffle=True):
        self.sentences = sentences
        if shuffle:
            random.shuffle(self.sentences)
        sentences_tmp1 = [sentence[0]+'||||'+str(sentence[1]) for sentence in sentences]
        sentences_tmp2 = list(set(sentences_tmp1))
        self.sentences_no_repeat = [sentence.split('||||') for sentence in sentences_tmp2]
        self.char2index = {}
        self.index2char = {}
        self.char_count = 0
        self.batch_sentences = []
        self.train_sentences = self.sentences_no_repeat[:int(0.8*len(self.sentences_no_repeat))]
        self.test_sentences = self.sentences_no_repeat[int(0.8*len(self.sentences_no_repeat)):]
    
    def build(self):
        for sentence in self.sentences_no_repeat:
            for char in sentence[0]:
                if char not in self.char2index:
                    self.char2index[char] = self.char_count
                    self.index2char[self.char_count] = char
                    self.char_count += 1
                    
    def makeBatch(self,batch_size):
        self.batch_sentences = []
        batch = []
        for i,sentence in enumerate(self.train_sentences):
            if (i+1)%batch_size == 0:
                self.batch_sentences.append(batch)
                batch = []
            else:
                batch.append(sentence)
        if len(batch) > 0:
            self.batch_sentences.append(batch)