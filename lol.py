
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import lol

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

# %%
disp = False
# base temperature : 5 celsius for timothy
tbase = 5


# %% #####
def sos(dft):
    # returns start of the season
    # dft = pandas timeseries

    mv_avg = [1,1,1]
    dft5C = (dft >= tbase) *1

    dft5C_conv = np.convolve(dft5C.loc[:], mv_avg,'same')
    dft5C_conv = (dft5C_conv == sum(mv_avg)) * 1
    dft5C_conv = (dft5C_conv.cumsum() >= 1) *1
    out0 = 0
    if dft5C_conv.sum() > 0:
        out0 = np.where(np.diff(dft5C_conv) == 1 )[0][0]+2
    out = dft.index[out0]
    return out

def gdd(ts):
    # returns accumulated growth degree days
    # ts = pandas timeseries

    out = pad_ts(ts).copy()
    out[out<tbase]=0

    for yy in out.index.year.unique():
        start  = date(yy, 1, 1)
        end = date(yy, 12, 31)
        ind_yy = out.index[(out.index.date >= start) &  (out.index.date <= end)]
        outtmp = out.loc[ind_yy]
        outtmp.loc[:sos(outtmp)] = 0
        outtmp = outtmp.cumsum()
        out.loc[ind_yy] = outtmp
    
    return out

def pad_ts(ts):
    # ts = pandas timeseries
    # removes one day for bisextile years

    out = ts.copy()
    if out.index[0].month == 1 & out.index[0].day == 1:
        yys = out.index[0]
    else:
        yys = datetime(out.index[0].year,1,1)
        out.loc[yys] = out.iloc[0]

    if out.index[-1].month == 12 & out.index[-1].day == 31:
        yye = out.index[-1]
    else:
        yye = datetime(out.index[0].year,12,31)
        out.loc[yye] = out.iloc[-1]

    outi= out.resample('D')
    out = outi.interpolate(methond='polynomial',order=3)
    #if out.shape[0] == 365:
    #    out = out.drop(out.index[-1])
    return out


# %% #####

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

class weather_station:
    def __init__(self, df_t, dd_historic, dd_horizon, year_train, model_type):
        # df_t is a pd.Series

        self.data = df_t
        self.dd_historic, self.dd_horizon, self.year_train = dd_historic, dd_horizon, year_train

        if type(model_type) == str: model_type = (model_type)
        if 'lstm' in model_type:
            self.n_features = 1
            self.lstm_data = self.prep_data_4lstm()
            self.lstm_scaler = self.lstm_data[4]
            self.lstm_model = self.build_model_4lstm()
        #if 'prophet' in model_type: 
        #if 'creme' in model_type: 
        #if 'naive' in model_type: 

    def prep_data_4lstm(self):
        # to change later: historic period for training,
        #   for now, for training set, it is year_train years of data since the first day in the weather station
        #   for validation set, it is dhist + dhoriz after the training set

        T = self.data.index
        Tselect = self.data.index[self.data.index.year<=self.data.index[0].year + self.year_train]

        train = self.data.loc[Tselect]
        validate = self.data[(self.data.index > Tselect[-1]) & (self.data.index <= Tselect[-1] + timedelta(self.dd_historic + self.dd_horizon) )]
        #train.shape, validate.shape

        scaler = StandardScaler()
        scaler.fit(train.values.reshape(-1,1))

        train_sc = scaler.transform(train.values.reshape(-1,1))
        validate_sc = scaler.transform(validate.values.reshape(-1,1))
        #scaler.scale_, scaler.mean_, scaler.var_

        Xtrain, Ytrain = split_sequence(train_sc, self.dd_historic, self.dd_horizon)
        Xval, Yval = split_sequence(validate_sc, self.dd_historic, self.dd_horizon)
        #Xtrain.shape, Ytrain.shape
        #Xval.shape, Yval.shape

        return Xtrain, Ytrain, Xval, Yval, scaler

    def build_model_4lstm(self):
        n_neurons = 100
        n_epochs, n_batch_size = 25, 32
        Xtrain, Ytrain, Xval, Yval, sc = self.lstm_data

        encoder_inputs = tf.keras.layers.Input(shape=(self.dd_historic, self.n_features))
        encoder_l1 = tf.keras.layers.LSTM(n_neurons, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]

        decoder_inputs = tf.keras.layers.RepeatVector(self.dd_horizon)(encoder_outputs1[0])
        decoder_l1 = tf.keras.layers.LSTM(n_neurons, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
        decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features))(decoder_l1)

        model_out = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
        model_out.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())

        model_out.fit(Xtrain,Ytrain,epochs=n_epochs,validation_data=(Xval,Yval),batch_size=n_batch_size,verbose=0,callbacks=[reduce_lr])

        return model_out

    def lstm_predict(self,x_input):
        # x_input = pd.Series of length dd_historic

        dd_last = x_input.index[-1]

        x_input = x_input.values.reshape(-1,1)
        x_input = self.lstm_scaler.transform(x_input)
        x_input = x_input.reshape(1, self.dd_historic, self.n_features)

        ind_horizon= pd.date_range(dd_last+timedelta(1),periods=self.dd_horizon)

        yhat = self.lstm_model.predict(x_input)
        yhat = self.lstm_scaler.inverse_transform(yhat)
        yhat = pd.Series(yhat.flatten(),index=ind_horizon)

        return yhat
