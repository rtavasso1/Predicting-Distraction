# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:08:48 2022

@author: admin
"""

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from pyts.multivariate.transformation import WEASELMUSE
import matplotlib.pyplot as plt


X = np.load('data.npy', allow_pickle=True)
numSamples = X.shape[0]
y2, y3 = [0,1,1]*(numSamples//3), [0,1,2]*(numSamples//3)
y2, y3 = np.array(y2), np.array(y3)

skf = StratifiedKFold(n_splits=5,shuffle=True) #maintains class balance in folds for 2 class case
transformer = WEASELMUSE(word_size=5, n_bins=2, window_sizes=[12, 36],
                          chi2_threshold=15, sparse=False, strategy='uniform')

confusion2, confusion3 = 0,0
for n_iters in range(1):  #transforms data n different times
    X2 = transformer.fit_transform(X,y2)
    X3 = transformer.fit_transform(X,y3)
    scores2, scores3 = [],[]
    
    for train_index, test_index in skf.split(X2, y2):
        X_train, X_test = X2[train_index], X2[test_index]
        y_train, y_test = y2[train_index], y2[test_index]
        clf2 = LogisticRegression(solver='newton-cg').fit(X_train,y_train)
        y_pred = clf2.predict(X_test)
        confusion2 += confusion_matrix(y_test, y_pred)
        scores2.append(clf2.score(X_test, y_test))
        
    for train_index, test_index in skf.split(X3, y3):
        X_train, X_test = X3[train_index], X3[test_index]
        y_train, y_test = y3[train_index], y3[test_index]
        clf3 = LogisticRegression(solver='newton-cg').fit(X_train,y_train)
        y_pred = clf3.predict(X_test)
        confusion3 += confusion_matrix(y_test, y_pred)
        scores3.append(clf3.score(X_test, y_test))
        
print(f"2 Class Accuracy: {np.average(scores2):.2f}")
print(f"3 Class Accuracy: {np.average(scores3):.2f}")
disp2 = ConfusionMatrixDisplay(confusion_matrix=confusion2,display_labels=['Undistracted','Distracted'])
disp3 = ConfusionMatrixDisplay(confusion_matrix=confusion3,display_labels=['Undistracted','Distracted','Very Distracted'])
disp2.plot()
plt.show()
disp3.plot()
plt.show()