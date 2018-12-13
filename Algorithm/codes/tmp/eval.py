#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:07:28 2018

@author: longzhan
"""

from autoencoder import SimpleAutoEncoder,VAE
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import copy
from sklearn.metrics import accuracy_score
from seq2seq.util.checkpoint import Checkpoint
import os

rng = np.random.RandomState(42)


test_fn = 'data/CSIC2010/request_test.txt'
test_bpe_fn = "data/CSIC2010/request_test_bpe.txt"
train_bpe_fn = "data/CSIC2010/request_train_bpe.txt"
test_label_fn = "data/CSIC2010/label_test.txt"

def getSent(data_fn):
    sentences = []
    with open(data_fn) as f:
        for line in f:
            sentences.append(line.strip())
    return sentences

def get_bigram_feature(test_sentences):
    with open("data_bigram.pkl",'rb') as f:
        data = pickle.load(f)
    train_X = copy.deepcopy(data.features)
    data.makeFeatures(fit_obj=test_sentences)
    test_X = copy.deepcopy(data.features)
    return train_X,test_X
    
def get_bpe_feature(test_sentences):
    with open("data_bpe.pkl",'rb') as f:
        data = pickle.load(f)
    train_X = copy.deepcopy(data.features)
    data.makeFeatures(fit_obj=test_sentences)
    test_X = copy.deepcopy(data.features)
    return train_X,test_X
   
def get_MLP_feature(X,model_fn):
    model = SimpleAutoEncoder(X.shape[1])
    model.load_state_dict(torch.load(model_fn))
    model.eval()
    input_ = Variable(torch.from_numpy(X).type(torch.FloatTensor).view(X.shape[0],-1))
    X = model(input_)[0].data.numpy()
    return X

def get_VAE_feature(X,model_fn):
    model = VAE(X.shape[1])
    model.load_state_dict(torch.load(model_fn))
    model.eval()
    input_ = Variable(torch.from_numpy(X).type(torch.FloatTensor).view(X.shape[0],-1))
    mu, logvar = model.encode(input_)
    X = model.reparameterize(mu,logvar).data.numpy()
    return X

def get_seq2seq_feature(sentences):
    def getEncoded(src_seq, model, src_vocab):
        src_id_seq = torch.LongTensor([src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        with torch.no_grad():
            encoder_outputs, encoder_hidden = model.encoder(src_id_seq, [len(src_seq)])
        return torch.sum(encoder_hidden.view(2,48),dim=0) #这里的维度要注意改动
                
    expt_dir = 'model/experiment'
    load_checkpoint = '2018_12_12_15_05_59'
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab

    
    Hidden = []
    for sentence in sentences:
        seq = sentence.strip().split()
        Hidden.append(getEncoded(seq, model, input_vocab).numpy())
    return np.array(Hidden)

   
def visualize(X,label):
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    X_embedded_pos = X_embedded[label==0]
    X_embedded_neg = X_embedded[label==1]
    plt.scatter(X_embedded_pos[:,0]*pow(10,6),X_embedded_pos[:,1]*pow(10,6),c='r')
    plt.show()
    plt.scatter(X_embedded_neg[:,0]*pow(10,6),X_embedded_neg[:,1]*pow(10,6),c='b')
    plt.show()
    return X_embedded

        
def evaluate_IF(X_train,X_test,gold,c):
    clf = IsolationForest(max_samples=500,n_estimators=200,behaviour="new",
                      random_state=rng,contamination=c)
    clf.fit(X_train)
    gold_for_IF = []
    for label in gold:
        if label == 1:
            gold_for_IF.append(-1)
        else:
            gold_for_IF.append(1)
    pred=list(clf.predict(X_test))
    print (classification_report(gold_for_IF, pred))
    print("accuracy:",accuracy_score(gold_for_IF, pred))
    
def evaluate_kmeans(X,test_labels):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    labels = [1-i for i in labels] #有时kmeans会把二者弄反，这句话看情况用
    print(classification_report(test_labels, labels))
    print("accuracy:",accuracy_score(test_labels, labels))
    return labels
    
if __name__ == '__main__':
    #load data
    test_sentences_bpe = getSent(test_bpe_fn)
    train_sentences_bpe = getSent(train_bpe_fn)
    test_sentences = getSent(test_fn)
    test_labels = getSent(test_label_fn)
    test_labels = [int(i) for i in test_labels]
    
    #feature of seq2seq
    X_test = get_seq2seq_feature(test_sentences_bpe)
    X_train = get_seq2seq_feature(train_sentences_bpe)
    
    #features of autoencoder
    #train_X,test_X = get_bpe_feature(test_sentences_bpe)
    #X_train = get_MLP_feature(X_train,'model/model.params.MLP')
    #X_test = get_MLP_feature(X_test,'model/model.params.MLP')
    
    labels = evaluate_kmeans(X_test,test_labels)
    evaluate_IF(X_train,X_test,test_labels,c=0.3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    