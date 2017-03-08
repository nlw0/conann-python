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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


set_printoptions(precision=3, threshold=5, linewidth=100, suppress=True)

# use all digits
mnist = fetch_mldata("MNIST original")

# Split the data into a training set and a test set

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data / 255., mnist.target, train_size=4000, test_size=4000, random_state=0)

pca = PCA(n_components=80, whiten=True, svd_solver='randomized')
X_trans = pca.fit_transform(X_train)
T_trans = pca.transform(X_test)
print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))


C_range = np.logspace(1, 3, 5, base=2)
gamma_range = np.logspace(-3, 1, 5)
# C_range = np.linspace(1, 3, 5)
# gamma_range = np.linspace(1, 3, 5)
# kernel_range = ['linear', 'rbf']
kernel_range = ['rbf']
param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernel_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_trans, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)


for kk in ['linear', 'rbf']:
    svc = SVC(kernel=kk, C=4.0, gamma=0.01)  # 10, 0.1
    svc.fit(X_trans, y_train)
    y_train_pred = svc.predict(X_trans)
    y_pred = svc.predict(T_trans)
    train_accuracy = sum((y_train_pred == y_train)) / float(y_train.shape[0])
    accuracy = sum((y_pred == y_test)) / float(y_test.shape[0])
    print("pca+svd error (train): {:.2f}%".format(100 * (1 - train_accuracy)))
    print("pca+svd error (test): {:.2f}%".format(100 * (1 - accuracy)))


# PCA explained variance ratio:  0.894097873176
# The best parameters are {'kernel': 'rbf', 'C': 4.0, 'gamma': 0.01} with a score of 0.95
# pca+svd error (train): 0.72%
# pca+svd error (test): 12.28%
# pca+svd error (train): 0.02%
# pca+svd error (test): 5.25%
