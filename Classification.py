# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:08:48 2022

@author: admin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from pyts.multivariate.transformation import WEASELMUSE, MultivariateTransformer
from pyts.transformation import WEASEL
from pyts.image import GramianAngularField
from pyts.multivariate.image import JointRecurrencePlot
import matplotlib.pyplot as plt
import seaborn as sns

def trainScore(X,y2,y3,col='all',graphs=True):
    skf = StratifiedKFold(n_splits=5,shuffle=True) #maintains class balance in folds for 2 class case
    if col=='all':
        transformer = WEASELMUSE(word_size=5, n_bins=2, window_sizes=[12, 36], chi2_threshold=15, sparse=False, strategy='uniform')
    else:
        transformer = WEASEL(sparse=False)
        
    confusion2, confusion3 = 0,0
    scores2, scores3 = [],[]
    for n_iters in range(3):  #transforms data n different times
        # X2 = transformer.fit_transform(X,y2)
        # X3 = transformer.fit_transform(X,y3)
        X2, X3 = X,X
        
        for train_index, test_index in skf.split(X2, y2):
            X_train, X_test = X2[train_index], X2[test_index]
            y_train, y_test = y2[train_index], y2[test_index]
            transformer.fit(X_train,y_train)
            X_train = transformer.transform(X_train)
            clf2 = LogisticRegression(solver='newton-cg').fit(X_train,y_train)
            X_test = transformer.transform(X_test)
            y_pred = clf2.predict(X_test)
            confusion2 += confusion_matrix(y_test, y_pred)
            scores2.append(clf2.score(X_test, y_test))
            
        for train_index, test_index in skf.split(X3, y3):
            X_train, X_test = X3[train_index], X3[test_index]
            y_train, y_test = y3[train_index], y3[test_index]
            transformer.fit(X_train,y_train)
            X_train = transformer.transform(X_train)
            clf3 = LogisticRegression(solver='newton-cg').fit(X_train,y_train)
            X_test = transformer.transform(X_test)
            y_pred = clf3.predict(X_test)
            confusion3 += confusion_matrix(y_test, y_pred)
            scores3.append(clf3.score(X_test, y_test))
    
    if col == 'all':
        print(f"2 Class Accuracy: {np.average(scores2):.2f}")
        print(f"3 Class Accuracy: {np.average(scores3):.2f}")
    else:
        print(f"2 Class Accuracy with only {col}: {np.average(scores2):.2f}")
        print(f"3 Class Accuracy with only {col}: {np.average(scores3):.2f}")
        
    if graphs==True:
        disp2 = ConfusionMatrixDisplay(confusion_matrix=confusion2,display_labels=['Undistracted','Distracted'])
        disp3 = ConfusionMatrixDisplay(confusion_matrix=confusion3,display_labels=['Undistracted','Distracted','Very Distracted'])
        disp2.plot()
        plt.title(col)
        plt.savefig('2by2Confusion'+col)
        plt.show()
        disp3.plot()
        plt.title(col)
        plt.savefig('3by3Confusion'+col)
        plt.show()
    return np.average(scores2), np.average(scores3)

X = np.load('data.npy', allow_pickle=True)
numSamples = X.shape[0]
y2, y3 = [0,1,1]*(numSamples//3), [0,1,2]*(numSamples//3)
y2, y3 = np.array(y2), np.array(y3)

# trainScore(X,y2,y3,graphs=False)
columns = ['Accel','Brake','Openness','PupilL','PupilR','Speed','Steering','Throttle','Center','Front','Back','Objects_numeric','gsr_phasic']
scores2,scores3 = [], []
for i, col in enumerate(columns):
    score2, score3 = trainScore(X[:,i,:],y2,y3,col=col,graphs=False)
    scores2.append(score2)
    scores3.append(score3)

# plt.hist(scores2)
# plt.hist(scores3)
ax = sns.barplot(x=columns,y=scores2, order=scores2.sort())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_ylabel('Accuracy')
plt.savefig('2 Class Feature Importance')
plt.show()
ax = sns.barplot(x=columns,y=scores3, order=scores3.sort())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_ylabel('Accuracy')
plt.savefig('3 Class Feature Importance')
plt.show()

# jrp = JointRecurrencePlot(threshold='point', percentage=50)
# X_jrp = jrp.fit_transform(X)
# plt.figure(figsize=(5, 5))
# plt.imshow(X_jrp[3], cmap='binary', origin='lower')
# plt.title('Joint Recurrence Plot', fontsize=18)
# plt.tight_layout()
# plt.show()

















