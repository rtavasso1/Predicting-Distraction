import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import copy
import glob
#from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
import os
import neurokit2 as nk
from scipy import signal

def quantify(my_list):
    x = copy.deepcopy(my_list)
    for i in range(len(x)):
        if x[i] == ['BoundingBox', 'LeftGazePlane']:
            x[i] = 2.0
        elif x[i] == ['BoundingBox', 'FrontGazePlane']:
            x[i] = 2.0
        elif x[i] == ['BoundingBox', 'RightGazePlane']:
            x[i] = 2.0
        elif x[i] == ['BoundingBox', 'Arrows']:
            x[i] = 4.0
        elif x[i] == ['BoundingBox']:
            x[i] = 5.0
        # elif x[i] == ['BoundingBox', 'FrontGazePlane', 'LeftGazePlane']:
        #     x[i] = 6.0
        # elif x[i] == ['BoundingBox', 'LeftGazePlane', 'FrontGazePlane']:
        #     x[i] = 6.0
        # elif x[i] == ['BoundingBox', 'FrontGazePlane', 'RightGazePlane']:
        #     x[i] = 7.0
        # elif x[i] == ['BoundingBox', 'RightGazePlane', 'FrontGazePlane']:
        #     x[i] = 7.0
        else:
            x[i] = 1000.0
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


filepath = r"C:\Users\Riley\Desktop\UTK\DataVis\Experiment1\\"
file_list = glob.glob(filepath + 'exp*.xdf')
#Combined = pd.DataFrame()
#Combined = np.empty([13,869,1])
#Combined = np
length = []
maxlen = 353 #update when adding new files by taking max(length)
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
            #Throttle = Brake = Steering = [5.0 for i in range(len(y))]
            #print('Controllers initialized')
            
        if stream['info']['name'] == ['BBT-BIO-AAB014_GEN_15']:
            BVP = y
            #print('BVP initialized')
            
        if stream['info']['name'] == ['BBT-BIO-AAB014_GEN_14']:
            gsr_phasic = y[:,0]
            gsr_phasic = pd.cut(gsr_phasic,bins=10,labels=list(range(10))).tolist()
            gsr_phasic = normalize(gsr_phasic)
            # time = stream['time_stamps']
            # time1 = time[1]
            # timelast = time[-1]
            # tot_time = timelast - time1
            # sampling_rate = len(time)/tot_time
            # gsr_sig, info = nk.eda_process(gsr_signal, sampling_rate = sampling_rate)
            # gsr_phasic = gsr_sig.iloc[:,2]
            #gsr_phasic = [5.0 for i in range(len(y))]
            #print('GSR initialized')
            
        if stream['info']['name'] == ['BBT-BIO-AAB014_ExG_A_1']:
            ECG = y
            #print('ECG initialized')
            
        if stream['info']['name'] == ['UE4 Position']:
            Distance = y
            #print('Distance initialized')
            
        if stream['info']['name'] == ['UE4 Accel']:
            Accel = y[:,0].tolist()
            #Accel = [5.0 for i in range(len(y))]
            #print('Acceleration initialized')
            
        if stream['info']['name'] == ['UE4 Speed']:
            Speed = y[:,0].tolist()
            #Speed = [5.0 for i in range(len(y))]
            #print('Speed initialized')
    
        if stream['info']['name'] == ['UE4 Pupil']:
            Pupil = y
            PupilL = np.array([y[i][0] for i in range(len(y))]).tolist()
            PupilR = np.array([y[i][1] for i in range(len(y))]).tolist()
            #PupilL = PupilR = [5.0 for i in range(len(y))]
            #print('Pupil initialized')
            
        if stream['info']['name'] == ['UE4 Openness']:
            Openness = y[:,0].tolist()
            #Openness = [5.0 for i in range(len(y))]
            #print('Openness initialized')
            
        if stream['info']['name'] == ['UE4 Eye']:
            Objects = [y[i][0].split(',') for i in range(len(y))]   #[y[i][0] for i in range(len(y))]
            Objects_numeric = quantify(Objects)
            #Objects_numeric = [5.0 for i in range(len(y))]
            #print('Gaze initialized')
            
        if stream['info']['name'] == ['UE4 DistanceToCarsAndCenter']:
            Front = np.expand_dims(y[:,0], axis=1)[:,0].tolist()
            Back = np.expand_dims(y[:,1], axis=1)[:,0].tolist()
            Center = np.expand_dims(y[:,2], axis=1)[:,0].tolist()
            #Front = Back = Center = [5.0 for i in range(len(y))]
            #print('Distance to cars and center initialized')
        
        if stream['info']['name'] == ['UE4 Road']:
            Road = y
            Road = [True if i == ['BP_Traffic_path_91'] else False for i in Road]
            
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
    try:
        dis1 = IDX00[0] #handles the exception for when I do not mark the end of undistracted driving by accident
    except:
        pass
    dis2begin = IDX44[0]
    dis2 = IDX44[-1] #end of 4x4
    dis3begin = IDX55[0]
    dis3 = IDX55[-1] #end of 5x5
    length.append(dis1)
    length.append(dis2-dis2begin)
    length.append(dis3-dis3begin)
    
    gsr_phasic = signal.resample(gsr_phasic, len(Accel)).tolist()
    
    x0 = np.linspace(1,dis1,dis1)
    x1 = np.linspace(1,dis2-dis2begin,dis2-dis2begin)
    x2 = np.linspace(1,dis3-dis3begin,dis3-dis3begin)
    xvals = np.linspace(1,maxlen, maxlen)
    
    mylist0 = [Accel[:dis1],Brake[:dis1],Openness[:dis1],PupilL[:dis1],PupilR[:dis1],Speed[:dis1],Steering[:dis1],Throttle[:dis1],Center[:dis1],Front[:dis1],Back[:dis1],Objects_numeric[:dis1],gsr_phasic[:dis1]]#,Arrow[:dis1]]
    #mylist0 = [np.interp(xvals, x0, sublist) for sublist in mylist0]
    mylist0 = [sublist + [9999.0] * (maxlen - len(sublist)) for sublist in mylist0]
    mylist1 = [Accel[dis2begin:dis2],Brake[dis2begin:dis2],Openness[dis2begin:dis2],PupilL[dis2begin:dis2],PupilR[dis2begin:dis2],Speed[dis2begin:dis2],Steering[dis2begin:dis2],Throttle[dis2begin:dis2],Center[dis2begin:dis2],Front[dis2begin:dis2],Back[dis2begin:dis2],Objects_numeric[dis2begin:dis2],gsr_phasic[dis2begin:dis2]]#,Arrow[dis1:dis2]]
    #mylist1 = [np.interp(xvals, x1, sublist) for sublist in mylist1]
    mylist1 = [sublist + [9999.0] * (maxlen - len(sublist)) for sublist in mylist1]
    mylist2 = [Accel[dis3begin:dis3],Brake[dis3begin:dis3],Openness[dis3begin:dis3],PupilL[dis3begin:dis3],PupilR[dis3begin:dis3],Speed[dis3begin:dis3],Steering[dis3begin:dis3],Throttle[dis3begin:dis3],Center[dis3begin:dis3],Front[dis3begin:dis3],Back[dis3begin:dis3],Objects_numeric[dis3begin:dis3],gsr_phasic[dis3begin:dis3]]#,Arrow[dis2:]]
    #mylist2 = [np.interp(xvals, x2, sublist) for sublist in mylist2]
    mylist2 = [sublist + [9999.0] * (maxlen - len(sublist)) for sublist in mylist2]
    #Interpolation cannot be used because it causes you to lose time synchronicity across samples
    
    mylist0 = np.reshape(mylist0, (1,13,maxlen))
    mylist1 = np.reshape(mylist1, (1,13,maxlen))
    mylist2 = np.reshape(mylist2, (1,13,maxlen))
    try:
        Combined = np.concatenate((Combined,mylist0,mylist1,mylist2),axis=0)
    except:
        Combined = np.concatenate((mylist0,mylist1,mylist2),axis=0)

os.remove('data.npy')
np.save('data.npy', Combined, allow_pickle=True)
