import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Input,InputLayer
from tensorflow.keras.layers import LSTM,TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import models,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D,GlobalAveragePooling2D

from tensorflow.keras.models import model_from_json

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint



def LSTM_CNN_DROPOUT(input_shape=None, n_outputs=None):
    K.set_image_data_format('channels_last')

    model = Sequential()
    model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=input_shape))  # 16
    model.add(Activation("relu"))
    model.add(Bidirectional(LSTM(16, return_sequences=True)))  # 16
    model.add(Activation("relu", name="LSTM_2"))
    layer = model.get_layer('LSTM_2')

    # model.add(BatchNormalization())
    model.add(keras.layers.Reshape((layer.output_shape[1], layer.output_shape[2], 1)))
    # model.add(BatchNormalization(name='normal_'))
    model.add(Dropout(0.3))

    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='relu', name='conv0'))  # 32
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pooling0'))
    model.add(Dropout(0.3))  # 0.3
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', name='conv1'))  # 64
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pooling1'))
    model.add(Dropout(0.3))  # 0.3
    model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu', name='conv2'))  # 128
    model.add(Dropout(0.3))  # 0.3
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization(name='normal'))

    model.add(Dense(n_outputs, kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Activation("softmax"))
    # model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.0005)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def LSTM_CNN(input_shape=None, n_outputs=None):
    K.set_image_data_format('channels_last')


    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Activation("relu", name="LSTM_2"))
    layer = model.get_layer('LSTM_2')
    model.add(keras.layers.Reshape((layer.output_shape[1], layer.output_shape[2], 1)))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', name='conv1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pooling1'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', name='conv2'))
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization(name='normal'))

    model.add(Dense(n_outputs, kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Activation("softmax"))
    # model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def CNN_LSTM(input_shape=None, n_outputs=None):
    K.set_image_data_format('channels_last')

    model = Sequential()
    model.add(keras.layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
    model.add(Conv2D(128, (5, 1), strides=(2, 2), padding='same', activation='relu', name='conv1'))
    model.add(MaxPooling2D((2, 1), strides=(2, 1), name='pooling0'))
    model.add(Conv2D(64, (3, 1), strides=(1, 1), padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D((2, 1), strides=(2, 1), name='pooling1'))
    model.add(Conv2D(32, (3, 1), strides=(1, 1), padding='same', activation='relu', name='conv3'))
    model.add(BatchNormalization(name='normal'))

    model.add(GlobalAveragePooling2D())
    layer = model.get_layer('normal')
    model.add(keras.layers.Reshape((layer.output_shape[3], -1)))

    model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Bidirectional(LSTM(16)))
    model.add(Activation("relu", name="LSTM_2"))

    model.add(Dense(n_outputs, kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Activation("softmax"))
    # model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    return model