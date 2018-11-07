# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:18:52 2018

@author: win 10
"""

import sys
sys.path.append("..")
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch
from utils import *
from collections import Counter
from config import *
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)

ae_features = np.load(ae_feature_npy)
labels = np.load(label_npy)
X_train, X_test, y_train, y_test = train_test_split(ae_features, labels, test_size=0.2, shuffle=True)

X_train_pos = []
for i in range(X_train.shape[0]):
    if y_train[i] == 0:
        X_train_pos.append(list(X_train[i]))
X_train_pos = np.array(X_train_pos)

clf = IsolationForest(max_samples=200,n_estimators=1000,
                      random_state=rng,contamination=0.2)
clf.fit(X_train_pos)
pred = list(clf.predict(X_test))
gold = list(y_test)
for i in range(len(gold)):
    if gold[i] == 1:
        gold[i] = -1
    if gold[i] == 0:
        gold[i] = 1
    
print (classification_report(gold, pred))