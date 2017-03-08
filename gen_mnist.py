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

Nsmp = 1000
Nl = 100
Ndim = 90

pca = PCA(n_components=Ndim, whiten=True, svd_solver='randomized')
X_trans = pca.fit_transform(X_train)
T_trans = pca.transform(X_test)
print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))

p_digits = [2]
n_digits = [0]

xsel_positive = [x for p in p_digits for x in nonzero(y_train == p)[0][:Nsmp]]
xsel_negative = [x for p in n_digits for x in nonzero(y_train == p)[0][:Nsmp]]
x_positive = X_trans[xsel_positive].T
x_negative = X_trans[xsel_negative].T

tsel_positive = [x for p in p_digits for x in nonzero(y_test == p)[0][:Nsmp]]
tsel_negative = [x for p in n_digits for x in nonzero(y_test == p)[0][:Nsmp]]
test_positive = T_trans[tsel_positive].T
test_negative = T_trans[tsel_negative].T

# line_directions = np.random.randn(Ndim, Nl) / sqrt(Ndim)
# lines_homog = vstack((line_directions, 3 * (np.random.uniform(size=Nl) * 2 - 1.0)))
lines_homog = np.random.randn(Ndim + 1, Nl) / sqrt(Ndim)

savez('mnist_data',
      x_positive=x_positive,
      x_negative=x_negative,
      lines_homog=lines_homog,
      test_positive=test_positive,
      test_negative=test_negative)

savez('mnist_test',
      test_positive=test_positive,
      test_negative=test_negative)
