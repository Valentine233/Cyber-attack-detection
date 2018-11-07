# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:27:47 2018

@author: win 10
"""
import sys
sys.path.append("..")
from data_preprocessing.data_preprocessing import Glove
import re
import numpy as np
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data_fn = "../../data/last_year/data_lastyear.log"
embed_fn = "../../data/char_embedding/glove.840B.300d-char.txt"
dim = 300

request_fn = "../../data/HTTP_2010/request.txt"
labels_fn = "../../data/HTTP_2010/label.txt"

def get_features(sentence, emb):
    feature = np.zeros(dim)
    for char in sentence:
        feature += emb.get_embedding_matrix()[emb.get_word_index(char)]
    return feature

def load(dataset):
    sentences = []
    labels = []
    if dataset == "last_year":
        with open(data_fn) as f:
            pattern = re.compile(r"\"[^\"]*\"")
            for line in f:
                if line.startswith("[SAFE]"):
                    line = line.replace("[SAFE]","").strip()
                    request = pattern.findall(line)[0].strip()
                    sentences.append(request)
                    labels.append(0)
                else:
                    line = line.replace("[ATTACKED]","").strip()
                    request = pattern.findall(line)[0].strip()
                    sentences.append(request)
                    labels.append(1)
    else:
        with open(request_fn) as f:
            for line in f:
                sentences.append(line.strip())
        with open(labels_fn) as f:
            for line in f:
                labels.append(int(line.strip()))
    
    return sentences, labels

def visualization(features, labels):
    tsne = TSNE()
    features_low = tsne.fit_transform(features)
    pos_features = []
    neg_features = []
    for i in range(len(labels)):
        if labels[i] == 1:
            pos_features.append(features_low[i])
        else:
            neg_features.append(features_low[i])
    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    plt.scatter(pos_features[:,0],pos_features[:,1],c='r')
    plt.scatter(neg_features[:,0],neg_features[:,1],c='b')
    plt.show()
    
if __name__ == "__main__":
    emb = Glove(embed_fn, dim)
    # 2010 or last_year
    sentences, labels = load("2010") 

    # get the features       
    features = []
    for sentence in sentences:
        features.append(get_features(sentence, emb))
    features = np.array(features)
    
    # clustering
    kmeans = KMeans(n_clusters=2, random_state=1).fit(features)
    pred_labels = kmeans.labels_
    print(classification_report(labels, pred_labels))
    
    # visualization
    # visualization(features, labels)
    
    