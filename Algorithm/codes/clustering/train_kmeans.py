import sys
sys.path.append("..")
from sklearn.cluster import KMeans
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

kmeans = KMeans(n_clusters=2).fit(ae_features)
pred_labels = kmeans.labels_
counter = Counter(pred_labels)
if counter[0] < counter[1]:
	pred_labels = 1 - pred_labels
equal = [1-x^y for (x, y) in zip(pred_labels, labels)]
accuracy = sum(equal) / len(labels)
print("Accuracy: %f" % accuracy)
precision = sum([x&y for (x, y) in zip(pred_labels, equal)]) / sum(pred_labels)
print("Precision: %f" % precision)
recall = sum([x&y for (x, y) in zip(pred_labels, equal)]) / sum(labels)
print("Recall: %f" % recall)


# plot
pca = PCA(n_components=2)
pca.fit(ae_features)
X_new = pca.transform(ae_features)
plt.figure(1)
plt.subplot(121)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='.', c=labels)
plt.title('real labels')
plt.subplot(122)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='.', c=pred_labels)
plt.title('predicted labels')
plt.show()