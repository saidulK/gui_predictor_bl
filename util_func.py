import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from qt import *
import time
import pandas as pd
from sklearn.decomposition import PCA
from numpy.fft import fft,rfft
import scipy.fftpack
from scipy.signal import stft,medfilt,firwin,convolve,butter,filtfilt
from scipy.signal import get_window
from scipy import interpolate
import random
import time



def freq_filter(data, f_size, cutoff, fs):
    lpf=firwin(f_size, cutoff/fs, window='hamming')    
    f_data=[convolve(d, lpf, mode='same') for d in data]
    return np.array(f_data)

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq=0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)    
    
    if len(data.shape)>1:
        y = [filtfilt(b, a, d) for d in data]
        return np.array(y)
    else:
        y = filtfilt(b, a, data)
        return y

def med_butter_filter(data, cutoff, fs, order, m_num):
    if len(data.shape)>2:
        medbutterdata = np.array([med_butter_filter(d, cutoff, fs, order, m_num) for d in data])
    elif len(data.shape)==2:
        fdata=[medfilt(d,m_num) for d in data.T]
        medbutterdata=np.array([butter_lowpass_filter(d, cutoff, fs, order) for d in fdata]).T
    else:        
        medbutterdata = butter_lowpass_filter(medfilt(data,m_num), cutoff, fs, order)
        
    return medbutterdata

def create_windows(data_series,window_size,overlap):
    win_result=[]
    l=len(data_series)
    ov=int(window_size*overlap)
    n = int((l - ov)//(window_size-ov))
    for i in range(n):
        win_result.append(data_series[i*(window_size-ov):i*(window_size-ov)+window_size])
    
    if (l - ov)%(window_size-ov):
        win_result.append(data_series[-window_size:])
    
    return np.array(win_result)


def gravity_vector(data_g):
    
    data=np.copy(data_g)
    gravity = butter_lowpass_filter(data.T[:3],0.3,50,1)
    return gravity    
    
def rotateData(data_g,act_g=[0,-1,0],gravity=None):    
    
    data=np.copy(data_g)
    act_g=act_g
    
    if gravity is None:
        gravity = butter_lowpass_filter(data.T[:3],0.1,50,1)   
    else:
        gravity = gravity.T
        
    bodyData = data.T[:3] - gravity
    
    for i,row in enumerate(data):
        
        
        quat = acc_to_quat(bodyData.T[i], row[:3], act_g, g= gravity.T[i])

        data[i][:3] = rotate_vector(quat, row[:3])
        if len(data[i])>3:
            data[i][-3:] = rotate_vector(quat, row[-3:])
            
    return data

def rotate_data_quat(data,quat):
    
    rotated_data = []
    
    for i,row in enumerate(data):
        if i<len(quat):
            
            rotated_data.append(rotate_vector(Quaternion(quat[i][0],-quat[i][1],-quat[i][2],-quat[i][3]), row[:3]))
    return np.array(rotated_data)


def rotateWindowedData(trainX,act_g=None):    
    #trainX shape 128,9
    
    d_list = list()
    coun2 = 0
    coun3 = 0

    if act_g is None:
        act_g=[0,1,0]

    d_prev = False
    d_next = False
    for i in range(trainX.shape[0]):

        if not i:
            d_prev = True
        elif i == len(trainX) - 1:
            d_next = True
        elif (trainX[i][:64] != trainX[i - 1][-64:]).any():
            d_prev = True
        elif (trainX[i][-64:] != trainX[i + 1][:64]).any():
            d_next = True
        for j in range(int(trainX.shape[1])):

            if not d_next and (j >= trainX.shape[1] / 2):
                break

            quat = acc_to_quat(trainX[i][j][:3], trainX[i][j][-3:], act_g)

            trainX[i][j][:3] = rotate_vector(quat, trainX[i][j][:3])
            trainX[i][j][3:6] = rotate_vector(quat, trainX[i][j][3:6])
            
            if trainX.shape[2]>6:
                trainX[i][j][-3:] = rotate_vector(quat, trainX[i][j][-3:])

            coun3 += 1

            if not d_prev and (j < trainX.shape[1] / 2):
                trainX[i - 1][int(trainX.shape[1] / 2) + j][:3] = trainX[i][j][:3]
                if trainX.shape[2]>6:
                    trainX[i - 1][int(trainX.shape[1] / 2) + j][-3:] = trainX[i][j][-3:]
                coun2 += 1
        d_list.append([coun2, coun3])

        

        print("Train:",i)
        coun2 = 0
        coun3 = 0

        d_prev = False
        d_next = False
        
    return trainX

def generate_filename(label):
    
    exp  = str(label[0])
    user = str(label[1])

    if int(exp)<10:
        exp="0"+exp
    if int(user)<10:
        user="0"+user    
    acc_file_name="acc_exp"+exp+"_user"+user+".txt"
    gyro_file_name="gyro_exp"+exp+"_user"+user+".txt"
    return acc_file_name,gyro_file_name



def equate_sampling_down(acc_data,gyro_data):
    
    if len(acc_data)>len(gyro_data):
        
        x=np.linspace(0,len(acc_data)-1,num=len(acc_data))
        f_list=[interpolate.interp1d(x, a, kind='quadratic') for a in acc_data.T]
        x_new=np.linspace(0,len(acc_data)-1,num=len(gyro_data))
        acc_new= [f(x_new) for f in f_list]
        acc_data=np.array(acc_new).T
    
    else:
        
        x=np.linspace(0,len(gyro_data)-1,num=len(gyro_data))
        f_list=[interpolate.interp1d(x, g, kind='quadratic') for g in gyro_data.T]
        x_new=np.linspace(0,len(gyro_data)-1,num=len(acc_data))
        gyro_new= [f(x_new) for f in f_list]
        gyro_data=np.array(gyro_new).T
        
    return acc_data,gyro_data


def equate_sampling(acc_data, gyro_data, acc_f, gyro_f, freq = 50):
    
       
    x=np.linspace(0,len(acc_data)-1,num=len(acc_data))
    f_list=[interpolate.interp1d(x, a, kind='quadratic') for a in acc_data.T]
    x_new=np.linspace(0,len(acc_data)-1,num= int(len(acc_data)*freq/acc_f))
    acc_new= [f(x_new) for f in f_list]
    acc_data=np.array(acc_new).T
       
    x=np.linspace(0,len(gyro_data)-1,num=len(gyro_data))
    f_list=[interpolate.interp1d(x, g, kind='quadratic') for g in gyro_data.T]
    x_new=np.linspace(0,len(gyro_data)-1,num = int(len(gyro_data)*freq/gyro_f))
    gyro_new= [f(x_new) for f in f_list]
    gyro_data=np.array(gyro_new).T
    
    acc_data,gyro_data = acc_data[:min(len(acc_data),len(gyro_data))],gyro_data[:min(len(acc_data),len(gyro_data))]
    
    return acc_data,gyro_data


def invalid_data(user,act,label_data):
    for i in range(len(label_data)//8):  
        if user in label_data[i*8]:    
            for j in range(7):           
                if (act in label_data[i*8+j]) and len(label_data[i*8+j].split(" "))>1:
                        range_data     = label_data[i*8+j].split(" ")[1:]
                        range_data[-1] = range_data[-1].split("\n")[0]
                        index=list()
                        for r in range_data:
                            if len(r.split("-"))>1:
                                a,b=int(r.split("-")[0]),int(r.split("-")[1])
                                index.append(np.arange(a,b))

                        return np.hstack(index)    
def time_to_sec(time):
    start,end = time.split("-")[0],time.split("-")[1]
    start = start.split(":")
    end   = end.split(":")
    start = (int(start[0])*60)+int(start[1])
    end = (int(end[0])*60)+int(end[1])
    return start,end


def act_data_index(user,act,frq,label_data):
    for i in range(len(label_data)//7):
            if user in label_data[i*7]: 
                for j in range(1,7):           
                    if (act in label_data[i*7+j]):
                        act_time=label_data[i*7+j].split(" ")[1:]
                        act_time[-1]=act_time[-1].split("\n")[0]
                        index_list=list()
                        for time in act_time:
                            if len(time)>1:
                                t=time_to_sec(time)
                                if int(t[1])-int(t[0])>4:
                                    index_list.append(np.arange(int(t[0]*frq),int(t[1]*frq)))
                        return np.hstack(index_list)
                    
                    
def load_act_data(act,sensors,location="G:/RealWorld Dataset/",user="User_1",freq=50,remove_data=True):
    
    if user=="User_8" and act=="climbingup":
        return None
    
    data=list()
    
    location= location + user +"/"
    
    final_url_acc=location+"/acc_"+act+"_csv/readMe"
    final_url_gyr=location+"/gyr_"+act+"_csv/readMe"
    f_acc = open(final_url_acc, "r")
    f_gyr = open(final_url_gyr, "r")
    f_acc=f_acc.readlines()
    f_gyr=f_gyr.readlines()
    label_file=open("G:/RealWorld Dataset/errors.txt","r")
    label_data=label_file.readlines()
    
    if len(sensors[0])==1:
        sensors=list(sensors)
        
    for sensor in sensors:
        acc_f,gyro_f = None, None      
        for i in range(6,len(f_acc),5):  
            if (sensor in f_acc[i]) and (sensor in f_gyr[i]):                               
                acc_entries,gyro_entries=f_acc[i+1].split(" ")[2],f_gyr[i+1].split(" ")[2] 
                acc_entries,gyro_entries=int(acc_entries),int(gyro_entries)

                acc_f,gyro_f=f_acc[i+2].split(" ")[2],f_gyr[i+2].split(" ")[2]        
                acc_f,gyro_f=float(acc_f),float(gyro_f)
       
                
        if acc_f!=None and gyro_f!=None:            
            """if abs((acc_entries/acc_f)-(gyro_entries/gyro_f))>0.1:
                
                if ((acc_entries/acc_f)>(gyro_entries/gyro_f)):
                    acc_entries= int((gyro_entries/gyro_f)*acc_f)
                else:
                    gyro_entries= int((acc_entries/acc_f)*gyro_f)
            """               
            #read from files
            
            acc_data=pd.read_csv(location+"acc_"+act+"_csv/"+"acc_"+act+"_"+sensor+".csv",delimiter=",")
            gyro_data=pd.read_csv(location+"gyr_"+act+"_csv/"+"Gyroscope_"+act+"_"+sensor+".csv",delimiter=",")     
            
            #acc_values=acc_data.iloc[:acc_entries,-3:].values
            acc_values=acc_data.iloc[:,-3:].values
            acc_values=acc_values/9.81998
            #gyro_values=gyro_data.iloc[:gyro_entries,-3:].values
            gyro_values=gyro_data.iloc[:,-3:].values
            
            acc_values,gyro_values=equate_sampling(acc_values,gyro_values,acc_f,gyro_f,freq=freq)
            
            act_index=act_data_index(user,act,freq,label_data)
            total_data=np.hstack((acc_values,gyro_values))
            #if user=="User_8" and act=="climbingup" and sensor== "waist":
            #    total_data=np.delete(total_data,np.arange(25000,45000),axis=0)
            
            if (act_index is not None) and remove_data:
                act_index = act_index[act_index<len(total_data)]
                total_data=np.take(total_data,act_index,axis=0)
            else:
                total_data=total_data[1500:len(total_data)-1500]
                
            data.append(total_data)
            #print(acc_data.iloc[:,-3:].values.T.shape,gyro_data.iloc[:,-3:].values.T.shape)
    return data
