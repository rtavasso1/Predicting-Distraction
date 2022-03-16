# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:05:22 2022

@author: admin
"""

import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import copy
import glob
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
import tensorflow as tf
import os
import neurokit2 as nk
from scipy import signal
#Cite Pytorch Tabular in publication
# from pytorch_tabular import TabularModel
# from pytorch_tabular.models import CategoryEmbeddingModelConfig
# from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


def quantify(my_list):
    x = copy.deepcopy(my_list)
    for i in range(len(x)):
        if x[i] == ['BoundingBox', 'LeftGazePlane']:
            x[i] = 1.0
        if x[i] == ['BoundingBox', 'FrontGazePlane']:
            x[i] = 2.0
        if x[i] == ['BoundingBox', 'RightGazePlane']:
            x[i] = 3.0
        if x[i] == ['BoundingBox', 'Arrows']:
            x[i] = 4.0
        if x[i] == ['BoundingBox'] or x[i] == ['']:
            x[i] = 5.0
        if x[i] == ['BoundingBox', 'FrontGazePlane', 'LeftGazePlane']:
            x[i] = 6.0
        if x[i] == ['BoundingBox', 'LeftGazePlane', 'FrontGazePlane']:
            x[i] = 6.0
        if x[i] == ['BoundingBox', 'FrontGazePlane', 'RightGazePlane']:
            x[i] = 7.0
        if x[i] == ['BoundingBox', 'RightGazePlane', 'FrontGazePlane']:
            x[i] = 7.0
        else:
            x[i] = 5
    return x

def CountFrequency(my_list):
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def normalize(x):
    maximum = max(x)
    minimum = min(x)
    temp = [(x[i] - minimum)/(maximum-minimum) for i in range(len(x))]
    return temp

def sublister(x):
    temp = []
    sub = []
    for i in range(len(x)):
        try:
            if x[i] == 'BoundingBox' and x[i-1] == 'BoundingBox':
                temp.append(sub)
                sub = []
        except:
            pass
        sub.append(x[i])
        try:
            if x[i+1] in ['FrontGazePlane','LeftGazePlane','RightGazePlane','Arrows']:
                temp.append(sub)
                sub = []            
        except:
            pass
    return temp


filepath = r"D:\\Users\\admin\\Documents\\BitBrain\\LabStreamLayer\\XDF_Files\\Experiment1\\Test\\"
file_list = glob.glob(filepath + 'exp*.xdf')
#Combined = pd.DataFrame()
#Combined = np.empty([13,869,1])
#Combined = np
length = []
maxlen = 821 #update when adding new files by taking max(length)
for i, file in enumerate(file_list):
    data, header = pyxdf.load_xdf(file)
    
    for stream in data:
        y = stream['time_series']
        
        if stream['info']['name'] == ['Teleport Home']:
            Home = y[:,0].tolist()
            HomeIDX = []
            for idx, val in enumerate(Home):
                if val == 9999 and Home[idx+1] == 0:
                    HomeIDX.append(idx)
            if len(HomeIDX) == 2:
                dis1 = HomeIDX[0]
                dis2 = HomeIDX[1]
            else:
                print('More than 2 Home Teleports!!!')
            
    
        if stream['info']['name'] == ['UE4 ThrottleAndMore']:
            Throttle = np.array([[(y[i][0]+1)/2] for i in range(len(y))])[:,0].tolist()
            Brake = np.array([[-y[i][1]-1] for i in range(len(y))])[:,0].tolist()
            Steering = np.array([[y[i][2]] for i in range(len(y))])[:,0].tolist()
            #print('Controllers initialized')
            
        if stream['info']['name'] == ['BBT-BIO-AAB014_GEN_15']:
            BVP = y
            #print('BVP initialized')
            
        if stream['info']['name'] == ['BBT-BIO-AAB014_GEN_14']:
            gsr_signal = y[:,0].tolist()
            time = stream['time_stamps']
            time1 = time[1]
            timelast = time[-1]
            tot_time = timelast - time1
            sampling_rate = len(time)/tot_time
            gsr_sig, info = nk.eda_process(gsr_signal, sampling_rate = sampling_rate)
            gsr_phasic = gsr_sig.iloc[:,2].tolist()
            #print('GSR initialized')
            
        if stream['info']['name'] == ['BBT-BIO-AAB014_ExG_A_1']:
            ECG = y
            #print('ECG initialized')
            
        if stream['info']['name'] == ['UE4 Position']:
            Distance = y
            #print('Distance initialized')
            
        if stream['info']['name'] == ['UE4 Accel']:
            Accel = y[:,0].tolist()
            #Accel1 = [Accel[:dis1], Accel[dis1:dis2], Accel[dis2:]]
            #print('Acceleration initialized')
            
        if stream['info']['name'] == ['UE4 Speed']:
            Speed = y[:,0].tolist()
            #print('Speed initialized')
        
        if stream['info']['name'] == ['UE4 Lane']:
            Lane = y
            LaneL = [y[i][0] for i in range(len(Lane))]        
    
        if stream['info']['name'] == ['UE4 Pupil']:
            Pupil = y
            PupilL = np.array([y[i][0] for i in range(len(y))]).tolist()
            PupilR = np.array([y[i][1] for i in range(len(y))]).tolist()
            #print('Pupil initialized')
            
        if stream['info']['name'] == ['UE4 Openness']:
            Openness = y[:,0].tolist()
            #print(len(Openness))
            #print('Openness initialized')
            
        if stream['info']['name'] == ['UE4 Eye']:
            Objects = [y[i][0].split(',') for i in range(len(y))]   #[y[i][0] for i in range(len(y))]
            Objects_numeric = quantify(Objects)
            #print('Gaze initialized')
            
        if stream['info']['name'] == ['UE4 DistanceToCarsAndCenter']:
            Front = np.expand_dims(y[:,0], axis=1)[:,0].tolist()
            Back = np.expand_dims(y[:,1], axis=1)[:,0].tolist()
            Center = np.expand_dims(y[:,2], axis=1)[:,0].tolist()
            #print('Distance to cars and center initialized')
        
        if stream['info']['name'] == ['UE4 Road']:
            Road = y
            Road = [True if i == ['BP_Traffic_path_91'] else False for i in Road]
            
            
        if stream['info']['name'] == ['Teleport 1']:
            Teleport1 = y
            for idx, i in enumerate(Teleport1):
                if i == 9999:
                    TP1 = idx
                    break
                
        if stream['info']['name'] == ['Teleport 3']:
            Teleport2 = y
            for idx, i in enumerate(Teleport2):
                if i == 9999:
                    TP2 = idx
                    break
                
        if stream['info']['name'] == ['Teleport 4']:
            Teleport3 = y
            for idx, i in enumerate(Teleport3):
                if i == 9999:
                    TP3 = idx
                    break
        
        if stream['info']['name'] == ['ArrowSequence']:
            Arrow = y[:,0].tolist()
            IDX44 = []
            IDX55 = []
            IDX00 = []
            for idx, val in enumerate(Arrow):
                if val == 9999 and Arrow[idx+1] == 0:
                    IDX44.append(idx)
                if val == 10000 and Arrow[idx+1] == 0:
                    IDX55.append(idx)
                if val == 9998 and Arrow[idx+1] == 0:
                    IDX00.append(idx)
    if len(IDX44) < 7 or len(IDX55) < 7:
        print('DID NOT PRESS F OR G 7 TIMES')
    dis1 = IDX00[0] #end of no distraction
    dis2begin = IDX44[0]
    dis2 = IDX44[-1] #end of 4x4
    dis3begin = IDX55[0]
    dis3 = IDX55[-1] #end of 5x5
    gsr_phasic =  signal.resample(gsr_phasic, len(Accel)).tolist()
    # goose = Accel[:dis1]
    # goose += [9999.0] * (869 - len(goose))
    # pad = [[9999 for i in range(13)]]
    length.append(dis1)
    length.append(dis2-dis2begin)
    length.append(dis3-dis3begin)
    # mydict0 = [{'Accel': Accel[:dis1], 'Brake': Brake[:dis1], 'Openness': Openness[:dis1], 'PupilL': PupilL[:dis1], 'PupilR': PupilR[:dis1], 'Speed': Speed[:dis1], 'Steering': Steering[:dis1], 'Throttle': Throttle[:dis1], 'Center': Center[:dis1], 'Front': Front[:dis1], 'Back': Back[:dis1], 'Objects': Objects_numeric[:dis1], 'Arrow': Arrow[:dis1]}]
    # mydict1 = [{'Accel': Accel[dis1:dis2], 'Brake': Brake[dis1:dis2], 'Openness': Openness[dis1:dis2], 'PupilL': PupilL[dis1:dis2], 'PupilR': PupilR[dis1:dis2], 'Speed': Speed[dis1:dis2], 'Steering': Steering[dis1:dis2], 'Throttle': Throttle[dis1:dis2], 'Center': Center[dis1:dis2], 'Front': Front[dis1:dis2], 'Back': Back[dis1:dis2], 'Objects': Objects_numeric[dis1:dis2], 'Arrow': Arrow[dis1:dis2]}]
    # mydict2 = [{'Accel': Accel[dis2:], 'Brake': Brake[dis2:], 'Openness': Openness[dis2:], 'PupilL': PupilL[dis2:], 'PupilR': PupilR[dis2:], 'Speed': Speed[dis2:], 'Steering': Steering[dis2:], 'Throttle': Throttle[dis2:], 'Center': Center[dis2:], 'Front': Front[dis2:], 'Back': Back[dis2:], 'Objects': Objects_numeric[dis2:], 'Arrow': Arrow[dis2:]}]
    # Combined = Combined.append(mydict0, ignore_index = True)
    # Combined = Combined.append(mydict1, ignore_index = True)
    # Combined = Combined.append(mydict2, ignore_index = True)
    #mylist2 = tf.keras.preprocessing.sequence.pad_sequences(mylist2, padding = 'post', value=9999, maxlen=869, dtype=type(float))
    #mylist2 = np.expand_dims(mylist2,axis=2)
    
    
    mylist0 = [Accel[:dis1],Brake[:dis1],Openness[:dis1],PupilL[:dis1],PupilR[:dis1],Speed[:dis1],Steering[:dis1],Throttle[:dis1],Center[:dis1],Front[:dis1],Back[:dis1],Objects_numeric[:dis1],gsr_phasic[:dis1]]#,Arrow[:dis1]]
    mylist0 = [goose + [9999.0] * (maxlen - len(goose)) for goose in mylist0]
    mylist1 = [Accel[dis2begin:dis2],Brake[dis2begin:dis2],Openness[dis2begin:dis2],PupilL[dis2begin:dis2],PupilR[dis2begin:dis2],Speed[dis2begin:dis2],Steering[dis2begin:dis2],Throttle[dis2begin:dis2],Center[dis2begin:dis2],Front[dis2begin:dis2],Back[dis2begin:dis2],Objects_numeric[dis2begin:dis2],gsr_phasic[dis2begin:dis2]]#,Arrow[dis1:dis2]]
    mylist1 = [goose + [9999.0] * (maxlen - len(goose)) for goose in mylist1]
    mylist2 = [Accel[dis3begin:dis3],Brake[dis3begin:dis3],Openness[dis3begin:dis3],PupilL[dis3begin:dis3],PupilR[dis3begin:dis3],Speed[dis3begin:dis3],Steering[dis3begin:dis3],Throttle[dis3begin:dis3],Center[dis3begin:dis3],Front[dis3begin:dis3],Back[dis3begin:dis3],Objects_numeric[dis3begin:dis3],gsr_phasic[dis3begin:dis3]]#,Arrow[dis2:]]
    mylist2 = [goose + [9999.0] * (maxlen - len(goose)) for goose in mylist2]
    
    mylist0 = np.reshape(mylist0, (1,13,maxlen))
    mylist1 = np.reshape(mylist1, (1,13,maxlen))
    mylist2 = np.reshape(mylist2, (1,13,maxlen))
    
    try:
        Combined = np.concatenate((Combined,mylist0,mylist1,mylist2),axis=0)
    except:
        Combined = np.concatenate((mylist0,mylist1,mylist2),axis=0)
    
        
#print(Combined.shape)

y= []
for i in range(len(file_list)):
    y.append(0)
    y.append(1)
    y.append(2)

os.remove('data.npy')
np.save('data.npy', Combined, allow_pickle=True)

# index = np.arange(len(Front))
# # index = np.arange(HomeIDX[0])
# # index = np.arange(HomeIDX[0],HomeIDX[1])
# # index = np.arange(HomeIDX[1],TP1)
# # index = np.arange(TP1,TP2)
# # index = np.arange(TP2,TP3)
# # index = np.arange(TP3,len(Front))
# plt.plot(Front[index])
# plt.plot(-Back[index])
# plt.plot(Home[index])
# plt.ylim(-50,50)
# plt.title('Distance to cars')
# plt.show()
# plt.plot(Center[index])  
# plt.title('Distance to Center')
# plt.ylim(-300,300)
# plt.show()  
# plt.plot(Speed[index])
# plt.title('Speed')
# plt.show()
# plt.plot(Accel[index])
# plt.title('Accel')
# plt.show()
# plt.plot(Throttle[index])
# plt.plot(Brake[index])
# plt.title('Throttle and Brake')
# plt.show()
# plt.plot(Steering[index])
# plt.title('Steering')
# plt.show()
# plt.plot(Openness[index])
# plt.title('Eye Openness')
# plt.show()
