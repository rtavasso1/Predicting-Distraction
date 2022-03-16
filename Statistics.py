# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:19:50 2022

@author: admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X = np.load('data.npy', allow_pickle=True).T
y= []
for i in range(15): #update range for num of files
    y.append(0)
    y.append(1)
    y.append(2)
y = np.array(y).reshape(1,1,45)
y = np.repeat(y, 475, axis=0)
X = np.append(X,y, axis=1)

X_2d = np.empty((1,14))
for i in range(45):
    X_2d = np.concatenate((X_2d,X[:,:,i]))
X_2d = X_2d[1:,:]

columns = ["Accel","Brake","Openness","PupilL","PupilR","Speed","Steering","Throttle","Center","Front","Back","Objects_numeric","gsr_phasic","State"]
X = pd.DataFrame(X_2d, columns=columns)

state_grouped = X.groupby(['State']).corr()
corr = X.corr().abs().unstack().reset_index()
corr = corr[corr.level_0=='State']

looked_at = X.groupby('Objects_numeric')['Objects_numeric'].count()
#Generates Object Count Plot
Object_plot = sns.countplot(x='Objects_numeric', hue='State', data=X)
Object_plot.set_xticklabels(['Left','Front','Right','Arrows','Other','Combo','None'])
new_title = 'State'
Object_plot.legend_.set_title(new_title)
# replace labels
new_labels = ['Undistracted', 'Minorly Distracted', 'Majorly Distracted']
for t, l in zip(Object_plot.legend_.texts, new_labels):
    t.set_text(l)


X_new = X.groupby('State')['gsr_phasic']
gsr = sns.displot(x='gsr_phasic', hue='State', data=X);
new_labels = ['Undistracted', 'Minorly Distracted', 'Majorly Distracted']
for t, l in zip(gsr._legend.texts, new_labels):
    t.set_text(l)