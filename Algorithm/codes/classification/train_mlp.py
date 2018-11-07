import sys
sys.path.append("..")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import torch
from utils import *
from collections import Counter
from config import *
import matplotlib.pyplot as plt

ae_features = np.load(ae_feature_npy)
labels = np.load(label_npy)
X_train, X_test, y_train, y_test = train_test_split(ae_features, labels, test_size=0.2, shuffle=True)

mlp = MLPClassifier(hidden_layer_sizes=(8, 4 ,2), learning_rate='invscaling')
mlp.fit(X_train, y_train)
pred_labels = mlp.predict(X_test)

equal = [1-x^y for (x, y) in zip(pred_labels, y_test)]
accuracy = sum(equal) / len(y_test)
print("Accuracy: %f" % accuracy)
precision = sum([x&y for (x, y) in zip(pred_labels, equal)]) / sum(pred_labels)
print("Precision: %f" % precision)
recall = sum([x&y for (x, y) in zip(pred_labels, equal)]) / sum(y_test)
print("Recall: %f" % recall)


# plot
pca = PCA(n_components=2)
pca.fit(X_test)
X_new = pca.transform(X_test)
plt.figure(1)
plt.subplot(121)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='.', c=y_test)
plt.title('real labels')
plt.subplot(122)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='.', c=pred_labels)
plt.title('predicted labels')
plt.show()