# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:19:50 2022

@author: admin
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def graphGen(): #function that generates plots
    #Generates Object Count Plot
    Object_plot = sns.countplot(x='Objects_numeric', hue='State', data=X3)
    Object_plot.set_xticklabels(['Front','Arrows','Other','None'])
    Object_plot.legend_.set_title('State')

    #GSR plot
    gsr = sns.displot(x='gsr_phasic', hue='State', data=X3);

    #Accel plot
    Accel = sns.displot(x='Accel', hue='State', data=X3, kind='kde')

    #Speed
    Speed = sns.displot(x='Speed', hue='State', data=X3, kind='kde')

    #Openness
    Openness = sns.displot(x='Openness', hue='State', data=X3, kind='ecdf')
    
    #Front
    Front = sns.displot(x='Front', hue='State', data=X3, kind='kde')
    
    #Back
    Back = sns.displot(x='Back', hue='State', data=X3, kind='kde')
    
    #Center
    Center = sns.displot(x='Center', hue='State', data=X3, kind='kde')
    
    #Steering
    Steering = sns.displot(x='Steering', hue='State', data=X3, kind='kde')
    
    #PupilL
    PupilL = sns.displot(x='PupilL', hue='State', data=X3, kind='kde')
    
    #Versus
    Versus = sns.displot(x='Front', y='Speed', hue='State', data=X3)


X = np.load('data.npy', allow_pickle=True).T
numSamples,numSteps = X.shape[2],X.shape[0]
y2, y3 = [0,1,1]*(numSamples//3), [0,1,2]*(numSamples//3)
y2, y3 = np.array(y2), np.array(y3)
y2, y3 = y2.reshape(1,1,numSamples), y3.reshape(1,1,numSamples)
y2, y3 = np.repeat(y2, numSteps, axis=0), np.repeat(y3, numSteps, axis=0)
X2, X3 = np.append(X,y2, axis=1), np.append(X,y3, axis=1)

X2_2d = np.empty((1,14))
X3_2d = np.empty((1,14))
for i in range(numSamples):
    X2_2d = np.concatenate((X2_2d,X2[:,:,i]))
    X3_2d = np.concatenate((X3_2d,X3[:,:,i]))
X2_2d = X2_2d[1:,:]
X3_2d = X3_2d[1:,:]

columns = ["Accel","Brake","Openness","PupilL","PupilR","Speed","Steering","Throttle","Center","Front","Back","Objects_numeric","gsr_phasic","State"]
X2 = pd.DataFrame(X2_2d, columns=columns)
X3 = pd.DataFrame(X3_2d, columns=columns)
X2 = X2[X2<9000] #filters missing data (values>9000)
X3 = X3[X3<9000]
X3.State = X3.State.map({0:'Undistracted',1:'Distracted',2:'Very Distracted'}) # replace numeric state with categorical

corr2 = X2.corr().abs().unstack().reset_index()
corr2 = corr2[corr2.level_0=='State']
corr3 = X3.corr().abs().unstack().reset_index()
corr3 = corr3[corr3.level_0=='State']

if len(sys.argv) > 1:
    if sys.argv[1] == 'graphs':
        graphGen()
        
    if sys.argv[1] == 'single':
        file_num = int(input('File number: '))
        X2 = X2.iloc[3*numSteps*(file_num-1):3*numSteps*(file_num)] #select subsample from master list
        X3 = X3.iloc[3*numSteps*(file_num-1):3*numSteps*(file_num)]
        graphGen()
        