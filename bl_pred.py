import threading
import numpy as np
import pandas as pd
from collections import Counter
import scipy.fftpack
from scipy.signal import stft,medfilt,firwin,convolve,butter,filtfilt
from scipy.signal import get_window
from scipy import interpolate
import pickle
from util_func import *
from qt import *
import itertools
import time
import keras
from tensorflow.keras.models import model_from_json

import tensorflow as tf
import numpy as np
from keras import backend as K

from models import *
from bl_funcs import *
from config import *



class predictor:

    def __init__(self):
        self.min_time = 3
        self.window_size  = 128
        self.window_overlap = 0.8
        self.model = None
        self.mean = None
        self.std  = None
        self.activities =None
        self.acc_wrp_gravity = True

    def setup_prediction(self):
        model_config = model_settings_3
        '''with open("G:/bluetooth/models/" + model_config['model_name'] + ".json") as model_file:
            model_json    = model_file.read()'''
        model_name = model_config['model_name']
        in_shape,out_shape = model_config['in_out_shape'][0],model_config['in_out_shape'][1]
        self.model        = model_names[model_name](in_shape,out_shape)
        self.model.load_weights("G:/bluetooth/models/"+model_config['weights'])
        mean_std_json     = json.load(open("G:/bluetooth/dataset_info/"+model_config['mean_std']+".json"))

        mean_json         = mean_std_json['mean']
        self.mean         = self.load_array_from_json(mean_json)
        self.mean         = np.array(self.mean)

        std_json          = mean_std_json['std']
        self.std          = self.load_array_from_json(std_json)
        self.std          = np.array(self.std)

        activities_json   = json.load(open("G:/bluetooth/dataset_info/"+model_config['activities']+".json"))
        self.activities   = self.load_array_from_json(activities_json)

        print("Model setup successful")



    def load_array_from_json(self,data_json):

        data_list = []
        for key in data_json.keys():
            data_list.append(data_json[key])
        return data_list


    def resize_data(self,data,freq=50):

        x = np.linspace(0, len(data) - 1, num=len(data))
        t = self.ns_to_s(data[-1, 0] - data[0, 0])
        f_list = [interpolate.interp1d(x, a, kind='quadratic') for a in data.T[1:]]
        x_new = np.linspace(0, len(data) - 1, num=int(t * freq))
        new_data = [f(x_new) for f in f_list]
        new_data = np.array(new_data).T
        return new_data

    def normalize_data(self,data, mean=None, std=None):

        x_reshaped = data.reshape(-1, data.shape[-1])
        if mean is None or std is None:
            mean, std = x_reshaped.mean(axis=0), x_reshaped.std(axis=0)
        x_reshaped = (x_reshaped - mean) / std
        return x_reshaped.reshape(data.shape)

    def create_array(self,acc_data,gyro_data):
        body_data = acc_data - np.array([butter_lowpass_filter(data, 0.3, 50, 1) for data in acc_data.T]).T
        total_data = np.hstack((body_data, gyro_data, acc_data))
        return total_data

    def ns_to_s(self,value):
        return value/1000000000

    def has_min_time(self,data):
        return True if self.ns_to_s(data[-1][0] - data[0][0])>= self.min_time else False

    def take_votes(self,prediction, n=3):

        pred_array = prediction.copy()

        if len(prediction)>n:
            pred_array = prediction[-n:].copy()

        pred = np.argmax(pred_array, axis=1)
        pred_counter = Counter(pred)


        n_max = pred_counter.most_common()[0][1]

        if len(pred_counter.most_common())!=1 and pred_counter.most_common()[1][1] == n_max:
            return pred_array[-1],pred_counter.most_common()[0][0]
        else:
            act_no = pred_counter.most_common()[0][0]
            print(pred[:])
            pred_new = pred_array[pred[:] == act_no]
            pred_new = np.mean(pred_new,axis=0)
            print(pred_new)
            #return pred_new,act_no
            return pred_array[-1],pred_counter.most_common()[0][0]


    def process_bluetooth_data(self,buffer,freq=50):

        try:
            acc_data  = np.array(buffer['Acc'])
            gyro_data = np.array(buffer['Gyro'])

        except Exception as e:
            print(e,"np array")
            return None

        if self.has_min_time(acc_data) and self.has_min_time(gyro_data):

            acc_data  = self.resize_data(acc_data,freq)
            gyro_data = self.resize_data(gyro_data,freq)


            min_len = min(len(acc_data), len(gyro_data))
            acc_data, gyro_data = acc_data[-min_len:], gyro_data[-min_len:]

            if self.acc_wrp_gravity:
                acc_data = acc_data/9.81

            total_data = self.create_array(acc_data,gyro_data)
            total_data = create_windows(total_data, self.window_size, self.window_overlap)
            return total_data

        else:
            raise Exception("Not Enough Time")
            return None


    def predict(self,bl_data):

        try:

            processed_data = self.process_bluetooth_data(bl_data)

            if processed_data is not None:

                final_data = processed_data.copy()
                # final_data = np.array([rotate_data(d,rotTotal=False) for d in final_data])


                final_data = self.normalize_data(final_data, self.mean, self.std)

                if len(final_data.shape) < 3:
                    final_data = final_data.reshape((1, final_data.shape[0], final_data.shape[1]))

                prediction = self.model.predict(final_data)

                confidence,act_no = self.take_votes(prediction)

                print(self.activities[act_no],act_no,confidence)
                return confidence,act_no


            else:
                return None
                #print("pdata is none")
        except Exception as e:

            print(e)


if __name__ == '__main__':

    data_server = bl_receiver()
    data_server.start()
    p  = predictor()
    p.setup_prediction()
    start = time.time()

    while True:

        if time.time() - start >1:

            print("predict")
            data = data_server.get_buffer_data()
            t =p.predict(data)
            print(t)
            start = time.time()








