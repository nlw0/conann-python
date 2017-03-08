from pylab import *

import numpy as np
from itertools import product
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

set_printoptions(precision=3, threshold=5, linewidth=100, suppress=True)

# use all digits
mnist = fetch_mldata("MNIST original")

# Split the data into a training set and a test set

# X_train, X_test, y_train, y_test = train_test_split(
#     mnist.data / 255., mnist.target, train_size=6000, test_size=1000, random_state=0)

X_train = mnist.data[:60000]
X_test = mnist.data[60000:70000]
y_train = mnist.target[:60000]
y_test = mnist.target[60000:70000]

# pca = PCA(n_components=80, whiten=True, svd_solver='randomized')
# X_trans = pca.fit_transform(X_train)
# T_trans = pca.transform(X_test)
# print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))
# X_reconstructed = pca.inverse_transform(X_trans)

# svc = svm.SVC(kernel='rbf', C=2)
# svc = svm.SVC(kernel='rbf', C=100)
# svc = svm.SVC(kernel='rbf', C=4.0, gamma=0.01)
svc = svm.SVC(kernel='rbf', C=2.82842712475, gamma=0.00728932024638)
svc.fit(X_train, y_train)
y_train_pred = svc.predict(X_train)
y_pred = svc.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

accuracy = sum((y_pred == y_test)) / float(y_test.shape[0])

figure()
title('MNIST classifier using PCA+SVM, accuracy {:4.2f}%'.format(accuracy))
imshow(cnf_matrix / 200.0, cmap=cm.RdBu)

train_accuracy = sum((y_train_pred == y_train)) / float(y_train.shape[0])
accuracy = sum((y_pred == y_test)) / float(y_test.shape[0])
print("pca+svd error (train): {:.2f}%".format(100 * (1 - train_accuracy)))
print("pca+svd error (test): {:.2f}%".format(100 * (1 - accuracy)))
