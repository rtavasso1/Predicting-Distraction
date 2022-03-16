# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:08:48 2022

@author: admin
"""

from matplotlib import pyplot as plt
import numpy as np
from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
    from_multi_index_to_3d_numpy,
    from_nested_to_3d_numpy,
)
from sklearn.model_selection import train_test_split
import sklearn
from pyts.classification import BOSSVS
from pyts.multivariate.classification import MultivariateClassifier
from pyts.multivariate.transformation import WEASELMUSE
from sklearn.linear_model import LogisticRegression
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
y= []
for i in range(15): #update range for num of files
    y.append(0)
    y.append(1)
    y.append(1)
y = np.array(y)
X = np.load('data.npy', allow_pickle=True)

# transformer = WEASELMUSE(word_size=5, n_bins=2, window_sizes=[12, 36],
#                           chi2_threshold=15, sparse=False, strategy='quantile')

# # X = transformer.fit_transform(X,y)
# # X_train, X_test, y_train, y_test = train_test_split(X,y)
# # clf = LogisticRegression(max_iter=300).fit(X_train, y_train)
# # print(clf.score(X_test,y_test))

# # N = len(X[0,0,:])
# # yf = fft(X[0,0,:])
# # xf = fftfreq(N, 0.1)[:N//2]
# # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# confusion = 0
# superscore = []
# for j in range(5):
#     X = np.load('data.npy', allow_pickle=True)
#     X = X.astype('float64')
#     X = transformer.fit_transform(X,y)
#     scores = []
#     for i in range(5):
#         X_train, X_test, y_train, y_test = train_test_split(X,y)
#         clf = LogisticRegression(solver='newton-cg', max_iter=100)
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         confusion += sklearn.metrics.confusion_matrix(y_test, y_pred)
#         scores.append(clf.score(X_test, y_test))
#     superscore.append(np.average(scores))
# print(superscore,np.average(superscore))
# disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=['Undistracted','Distracted'])
# disp.plot()
# plt.show()





# scores = []
# for i in range(30):
#     X_train, X_test, y_train, y_test = train_test_split(X,y)
#     clf = MultivariateClassifier(BOSSVS(strategy='uniform'))
#     clf.fit(X_train, y_train)
#     scores.append(clf.score(X_test, y_test))
# print(np.average(scores))

