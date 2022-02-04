import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input,InputLayer
from keras.layers import LSTM,TimeDistributed
from keras.layers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import models,Model
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ConvLSTM2D,Concatenate,Reshape,GRU
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D,GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score


model_settings_1  = {'model_name':'LSTM_CNN','mean_std':'UCI_mean_std','activities':'UCI_activities','in_out_shape':[(128,9),12],'weights':'weights-model_LSTM_CNN_norm_UCI_non_aug.h5'}

model_settings_2  = {'model_name':'LSTM_CNN','mean_std':'UCI_AUG_mean_std','activities':'UCI_activities','in_out_shape':[(128,9),12],'weights':'weights-model_LSTM_CNN_norm_UCI_ALL_AUG_NEW.h5'}

model_settings_3  = {'model_name':'LSTM_CNN_DROPOUT','mean_std':'UCI_mean_std','activities':'UCI_activities','in_out_shape':[(128,9),12],'weights':'weights-model_LSTM_CNN_DROPOUT_norm_UCI_non_aug.h5'}

model_settings_4  = {'model_name':'LSTM_CNN_DROPOUT','mean_std':'UCI_AUG_mean_std','activities':'UCI_activities','in_out_shape':[(128,9),12],'weights':'weights-model_LSTM_CNN_DROPOUT_norm_UCI_ALL_AUG_NEW.h5'}

model_settings_5  = {'model_name':'CNN_LSTM','mean_std':'UCI_mean_std','activities':'UCI_activities','in_out_shape':[(128,9),12],'weights':'weights-model_CNN_LSTM_norm_UCI_non_aug.h5'}

model_settings_6  = {'model_name':'CNN_LSTM','mean_std':'UCI_AUG_mean_std','activities':'UCI_activities','in_out_shape':[(128,9),12],'weights':'weights-model_CNN_LSTM_norm_UCI_ALL_AUG_NEW.h5'}

model_settings_LSTM_CNN = {'model_name':'LSTM_CNN',
                           'mean_std':'hardware_mean_std',
                           'activities':'hardware_activities',
                           'in_out_shape':[(128,6),7],
                           'weights':'weights-LSTM_CNN damko_hardware.h5'}

model_settings_LSTM_CNN_PARALLEL = {'model_name':'LSTM_CNN_PARALLEL',
                           'mean_std':'hardware_mean_std',
                           'activities':'hardware_activities',
                           'in_out_shape':[(128,6),7],
                           'weights':'weights-LSTM_CNN_PARALLEL damko_hardware.h5'}


'''model_settings_FUSION = {'model_name':'FUSION',
                           'mean_std':'hardware_mean_std',
                           'activities':'hardware_activities',
                           'in_out_shape':[(128,9),8],
                           'weights':''}'''

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


def LSTM_CNN(input_shape=None,n_outputs=None):
    
    K.set_image_data_format('channels_last')
    n_outputs = n_outputs
    input_shape = input_shape
    input           = Input(shape=input_shape)

    lstm_1          = Bidirectional(LSTM(32,return_sequences=True))(input)
    activation_1    = Activation("relu")(lstm_1)
    lstm_2          = Bidirectional(LSTM(32,return_sequences=True))(activation_1)
    activation_2    = Activation("relu")(lstm_2)

    reshape_layer_1 = Reshape((lstm_2.shape[1],lstm_2.shape[2],1))(activation_2)
    cnn_1           = Conv2D(64, (5,5), strides=(2,2), activation='relu')(reshape_layer_1)
    max_pool_1      = MaxPooling2D((2,2), strides=(2,2))(cnn_1)
    cnn_2           = Conv2D(128, (3,3), strides=(1,1), activation='relu')(max_pool_1)
    global_avg      = GlobalAveragePooling2D()(cnn_2)
    dense           = Dense(n_outputs,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.005))(global_avg)

    model = Model(inputs = input, outputs= dense)
    #model.summary()

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

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.0005)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def LSTM_CNN_PARALLEL(input_shape=None,n_outputs=None):
    
    K.set_image_data_format('channels_last')
    input_shape = input_shape
    input           = Input(shape=input_shape)

    lstm_1          = Bidirectional(LSTM(32,return_sequences=True))(input)
    activation_1    = Activation("relu")(lstm_1)
    lstm_2          = Bidirectional(LSTM(32,return_sequences=True))(activation_1)
    activation_2    = Activation("relu")(lstm_2)
    lstm_3          = Bidirectional(LSTM(32))(activation_2)
    activation_3    = Activation("relu")(lstm_3)
    activation_3    = Dropout(0.5)(activation_3)
    flatten         = Flatten()(activation_3)


    reshape_2         = Reshape((input.shape[1],input.shape[2],1))(input)
    cnn_1_2           = Conv2D(64, (5,1), strides=(2,2), activation='relu')(reshape_2)
    #max_pool_1_2     = MaxPooling2D((2,2), strides=(2,2))(cnn_1_2)
    cnn_2_2           = Conv2D(128, (3,3), strides=(1,1), activation='relu')(cnn_1_2)
    global_avg_2      = GlobalAveragePooling2D()(cnn_2_2)
    global_avg_2      = Dropout(0.5)(global_avg_2)
    flatten_2         = Flatten()(global_avg_2)

    concatenate       = Concatenate()([flatten,flatten_2])
   
    dense             = Dense(n_outputs,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.005))(concatenate)



    model = Model(inputs = input, outputs= dense)
    #model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model

def fusion_model(input_shape=None,n_outputs=None,m1_weights=None,m2_weights=None):

    K.set_image_data_format('channels_last')

    model1 = LSTM_CNN_PARALLEL(input_shape,n_outputs)
    model2 = LSTM_CNN(input_shape,n_outputs)

    if m1_weights is not None:
        model1.load_weights(m1_weights)
    if m2_weights is not None:
        model2.load_weights(m2_weights)
    
    
    concatenate = Concatenate()([model1.layers[-2].output,model2.layers[-2].output])
    concatenate = Dropout(0.3)(concatenate)
    concatenate = Reshape((-1,concatenate.shape[-1]))(concatenate)
    gru_1       = GRU(32, return_sequences = True, activation = 'relu')(concatenate)
    gru_1       = Dropout(0.3)(gru_1)
    gru_2       = GRU(32, activation = 'relu')(gru_1)
    gru_2       = Dropout(0.3)(gru_2)
    bn          = BatchNormalization()(gru_2)
    bn          = Dropout(0.5)(bn)
    dense       = Dense(n_outputs,activation='softmax')(bn)
   
    model       = Model(inputs=[model1.input,model2.input],outputs= dense)

    if (m1_weights is not None) and (m2_weights is not None):
        for layer in model.layers[:-6]:
            layer.trainable = False
    
    for layer in model.layers[-6:]:
            print(layer)

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()

    return model


model_names = {'LSTM_CNN':LSTM_CNN, 
               'LSTM_CNN_DROPOUT':LSTM_CNN_DROPOUT, 
               'CNN_LSTM':CNN_LSTM,
               'LSTM_CNN_PARALLEL':LSTM_CNN_PARALLEL,
               'FUSION':fusion_model
               }

