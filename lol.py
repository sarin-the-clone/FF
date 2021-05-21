
import pickle
import os

from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from boto3.dynamodb.conditions import Key

from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

import fbprophet
from fbprophet import Prophet
import itertools
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

import calendar
from river import compose
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing
from river import time_series
from river import neural_net as nn
from river import reco
from river import linear_model as lm

disp = True
# base temperature : 5 celsius for timothy
tbase = 5

# %% misc fun for temperature timeseries
def sos(dft):
    # returns start of the season
    # dft = pandas timeseries

    mv_avg = [1,1,1]
    #out=pd.Series()
    #out = pd.DataFrame(columns=dft.columns)
    dft5C = (dft >= tbase) *1

    dft5C_conv = np.convolve(dft5C.loc[:], mv_avg,'same')
    dft5C_conv = (dft5C_conv == sum(mv_avg)) * 1
    dft5C_conv = (dft5C_conv.cumsum() >= 1) *1
    out0 = 0
    if dft5C_conv.sum() > 0:
        out0 = np.where(np.diff(dft5C_conv) == 1 )[0][0]+2
    out = dft.index[out0]
    #out.loc['sos'] = out0

    #for k in dft.columns:
    #    dft5C_conv = np.convolve(dft5C.loc[:,k], mv_avg,'same')
    #    dft5C_conv = (dft5C_conv == sum(mv_avg)) * 1
    #    dft5C_conv = (dft5C_conv.cumsum() >= 1) *1
    #    out0 = 0
    #    if dft5C_conv.sum() > 0:
    #        out0 = np.where(np.diff(dft5C_conv) == 1 )[0][0]+2
    #    out0 = dft.index[out0]
    #    out.loc['sos',k] = out0
        #print(k,out0)
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

    #out.loc[:sos(out).loc['sos']] = 0
    #for k in out.columns:
    #    out.loc[:sos(out).loc['sos',k],k] = 0
    #    out[k] = out[k].cumsum()

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
    out = outi.interpolate(method='polynomial',order=3)
    #if out.shape[0] == 365:
    #    out = out.drop(out.index[-1])
    return out


def frost2ts(ws,table):
    #queries dynamodb table and returns the timeseries for the weather station ws as a pd.series

    fe = Key("SensorID").eq(ws)
    response = table.query(KeyConditionExpression=fe)

    out = []
    out_year = []
    out_day = []
    for i in response["Items"]:
        for ii in i.keys():
            if (ii != "SensorID") and (ii != "Year") and (ii != "Unnamed: 0"):
                out = np.append(out,float(i[ii]))
                out_day = np.append(out_day,int(ii[4:]))
                out_year = np.append(out_year,int(i["Year"]))

    ts_out = pd.DataFrame(out)
    ts_out['t']=list(zip(out_year,out_day))
    ts_out.index = ts_out['t'].apply(lambda x: datetime.strptime(str(int(x[0]))+ "-" +str(int(x[1])), "%Y-%j"))#.strftime("%Y-%m-%d") )
    ts_out.index.name = None
    ts_out = ts_out.rename({0:'value'},axis=1)['value']
    ts_out = pad_ts(ts_out).copy()
    ts_out = pd.Series(ts_out)
    return ts_out

def get_harvest_date(ts,target_gdd):
    ind_curryear = ts[ts.index.year == ts.index[-1].year].index
    if (ts[ind_curryear] >= target_gdd).max():
        HThatind = ts[ind_curryear][ts[ind_curryear] <= target_gdd].index[-1]
    else :
        HThatind = ts[ind_curryear].index[0]
    #HThat = HThatind.date()
    return HThatind

# %% misc fun for river

def get_ordinal_date(x):
    return {'ordinal_date': x['day'].toordinal()}

def get_month_distances(x):
    return {
        calendar.month_name[month]: math.exp(-(x['day'].month - month) ** 2)
        for month in range(1, 13)
    }

# %% misc fun for data preparation

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

def stan_init(m):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

def get_ordinal_date(x):
    return {'ordinal_date': x['day'].toordinal()}

#########################################################################################################
# %%
#########################################################################################################

class weather_station:
    def __init__(self,ws_name, df_t, dd_historic, dd_horizon, year_train, model_type):
        # df_t is a pd.Series
        self.ws_name = ws_name
        self.data = df_t
        self.dd_historic, self.dd_horizon, self.year_train = dd_historic, dd_horizon, year_train

        if not os.path.exists('./models_bck/'): os.makedirs('./models_bck/')
        self.pck_filename = './models_bck/' + str(ws_name)+'_snarimax.pickle'

        if type(model_type) == str: model_type = (model_type)

        if 'lstm' in model_type:
            self.n_features = 1
            self.lstm_data = self.prep_data_4lstm() # = Xtrain, Ytrain, Xval, Yval, scaler
            self.lstm_scaler = self.lstm_data[4]
            self.lstm_model = self.build_model_4lstm()

        if 'prophet' in model_type:
            self.ph_bestparameters = {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 3}
            self.ph_train = self.prep_data_4ph()
            self.ph_model = self.build_model_4ph()

        if 'snarimax' in model_type:
            #p=365,d=0,q=90, m=365, sp=90,sd=0, sq=90,
            #p : Order of the autoregressive part, number of past target values that will be included as features. (horizon for Xt)
            #d : Differencing order.
            #q : Order of the moving average part, number of past error terms that will be included as features. (horizon for epsilon)
            #m : Season length used for extracting seasonal features.
            #sp: Seasonal order, number of past target values that will be included as features.
            #sd: Seasonal differencing order
            #sq: Seasonal order of the moving average, number of past error terms that will be included as features.
            #self.snarimax_para = [365, 0, 90, 365, 90, 0, 90]
            self.snarimax_para = [0, 0, 730, 365, 0, 0, 730]
            #self.snarimax_para = [730, 0, 90, 365, 90, 0, 365]
            #self.snarimax_para = [0, 0, 365, 365, 90, 0, 365]
            self.snarimax_data = self.prep_data_4snarimax()
            self.build_model_4snarimax()

        #if 'naive' in model_type:


### lstm seq2seq ###
    def prep_data_4lstm(self): #return Xtrain, Ytrain, Xval, Yval, scaler

        T = self.data.index
        #Ttrain = T[0:self.year_train*365]
        #Tval = T[self.year_train*365:]
        Ttrain = T[0:-self.dd_historic-self.dd_horizon]
        Tval = T[-self.dd_historic-self.dd_horizon:]

        train = self.data.loc[Ttrain]
        validate = self.data[Tval]
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

    def build_model_4lstm(self): #return model_out
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

    def lstm_learn_new(self, X, Y): #does not work
        n_epochs, n_batch_size = 25, 32

        X = X.values.reshape(-1,1)
        X = self.lstm_scaler.transform(X)
        X = X.reshape(1, self.dd_historic, self.n_features)

        Y = Y.values.reshape(-1,1)
        Y = self.lstm_scaler.transform(Y)
        Y = Y.reshape(1, self.dd_historic, self.n_features)

        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
        self.lstm_model.fit(X,Y,epochs=n_epochs,batch_size=n_batch_size,verbose=0,callbacks=[reduce_lr])

    def lstm_predict(self,x_input): #return yhat
        # x_input = pd.Series of length dd_historic
        #x_input = self.data[-self.dd_historic:]
        dd_last = x_input.index[-1]

        x_input = x_input.values.reshape(-1,1)
        x_input = self.lstm_scaler.transform(x_input)
        x_input = x_input.reshape(1, self.dd_historic, self.n_features)

        ind_horizon= pd.date_range(dd_last+timedelta(1),periods=self.dd_horizon)

        yhat = self.lstm_model.predict(x_input)
        yhat = self.lstm_scaler.inverse_transform(yhat)
        yhat = pd.Series(yhat.flatten(),index=ind_horizon)

        return yhat

### FB prophet ###
    def getparameters_4ph(self): #return best_params and overide default parameters
        param_grid = { 'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.3, 0.5, 1],
                       'seasonality_prior_scale': [0.01, 0.1, 1, 3, 6, 10, 20],
                       }
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []
        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params,daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, growth='linear')
            m.add_seasonality(name='yearly', period=365.25, fourier_order=20)
            m = m.fit(self.ph_train)
            df_cv = cross_validation(m, initial='730 days', period='90 days', horizon = '180 days')
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        if disp: print(tuning_results)
        best_params = all_params[np.argmin(rmses)]
        if disp: print(best_params)
        self.ph_bestparameters = best_params
        return best_params

    def prep_data_4ph(self): #return df_ph_train
        df_ph_train = pd.DataFrame(self.data.values , columns=['y'])
        #df_ph_train.reset_index(inplace=True)
        #df_ph_train = df_ph_train.rename({'index':'ds'},axis=1)
        df_ph_train['ds'] = self.data.index
        return df_ph_train

    def build_model_4ph(self): #return mprophet
        mprophet = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, growth='linear',
                     seasonality_prior_scale=self.ph_bestparameters ['seasonality_prior_scale'],changepoint_prior_scale=self.ph_bestparameters ['changepoint_prior_scale'])
        #mprophet.add_seasonality(name='yearly', period=365.25, fourier_order=20)
        mprophet.fit(self.ph_train)
        return mprophet

    def ph_predict(self): #return yhat
        future = self.ph_model.make_future_dataframe(periods = self.dd_horizon)
        forecast = self.ph_model.predict(future)
        yhat = pd.Series(forecast['yhat'].values,index=forecast['ds'])
        yhat.loc[self.data.index] = self.data
        return yhat

### River SNARIMAX ###

    def prep_data_4snarimax(self): #converts into river dataset format
        #self.data = self.data.loc[self.data.index[:-self.data.index[-1].weekday()]].copy()
        svg_data = self.data.values
        #svg_data = savgol_filter(self.data, 7, 1, mode='nearest')
        dataset = pd.DataFrame(svg_data, columns=['temp'])
        dataset['ds']= self.data.index
        dataset.index = dataset['ds']
        dataset['ds_ordinal']= dataset['ds'].apply(lambda x:{'ordinal_date': x.toordinal()})
        dataset['ds']= dataset['ds'].apply(lambda x:{'day':x})
        return dataset #saved in self.snarimax_data

    def build_model_4snarimax(self):
        if os.path.exists(self.pck_filename): #if model backup exists then load it and update model from start1 to start2
            src_bck = pickle.load(open(self.pck_filename,'rb'))
            model = src_bck.snarimax_model
            metric = src_bck.snarimax_metric
            self.snarimax_para = src_bck.snarimax_para
            self.snarimax_model = model
            self.snarimax_metric = metric

            start1 = src_bck.data.index[-1]
            start2 = self.data.index[-1]#self.data.index[-self.data.index[-1].weekday()]

        else: #if model backup does not exist then rebuild model from the start
            p, d, q, m, sp, sd, sq = self.snarimax_para
            extract_features = compose.TransformerUnion(get_ordinal_date)
            model = (
                    extract_features |

                    time_series.SNARIMAX(
                        p=p, d=d, q=q, m=m, sp=sp, sd=sd, sq=sq,
                        regressor=(
                            #preprocessing.Normalizer() |
                            preprocessing.AdaptiveStandardScaler(alpha=0.1)|
                            preprocessing.StandardScaler() |

                            #preprocessing.RobustScaler(with_scaling=True) |
                            linear_model.LinearRegression(
                                intercept_init=0,
                                optimizer=optim.SGD(0.0001), #important parameter
                                #optimizer=optim.AdaDelta(0.8,0.00001), #important parameter
                                #optimizer=optim.AMSGrad(lr=0.01,beta_1=0.8,beta_2=0.1),
                                intercept_lr=0.001
                                )
                            )
                        )
                    )



            metric = metrics.Rolling(metrics.MSE(), self.dd_historic)
            #metric = metrics.MSE()

            start1 = self.data.index[0]
            start2 = self.data.index[-1]#self.data.index[-self.data.index[-1].weekday()]

        if start1 < start2:
            for t in pd.date_range(start1,start2,freq='D'):
                x, y = self.snarimax_data.loc[t][['ds','temp']].values
                y_pred = model.forecast(horizon=1, xs=[x])
                #print(x,y,y_pred[0],y-y_pred[0])
                model = model.learn_one(x, y)
                metric = metric.update(y, y_pred[0])


            self.snarimax_model = model
            self.snarimax_metric = metric
            with open(self.pck_filename, 'wb') as fh: pickle.dump(self,fh)

            #for t in pd.date_range(start1, start2):
            #    x = self.snarimax_data.loc[pd.date_range(t-timedelta(self.dd_historic),t)][['ds']].values
            #    y = self.snarimax_data.loc[pd.date_range(t-timedelta(self.dd_historic),t)][['temp']].values
            #    x = np.hstack(x)
            #    y = np.hstack(y)
            #    y_pred = model.forecast(horizon=self.dd_historic+1, xs=x)
            #    for i in range(0,self.dd_historic):
            #        model = model.learn_one(x[i], y[i])
            #        metric = metric.update(y[i], y_pred[i])




        return

    def snarimax_predict(self):
        t = self.data.index[-1]
        horizon = pd.date_range(t+timedelta(1),periods=self.dd_horizon,freq='D')

        x, y = self.snarimax_data.loc[t][['ds','temp']].values
        y_pred = self.snarimax_model.forecast(horizon=1, xs=[x])

        if True: #t.weekday()==1:
            self.snarimax_model = self.snarimax_model.learn_one(x, y)
            self.snarimax_metric = self.snarimax_metric.update(y, y_pred[0])

            with open(self.pck_filename, 'wb') as fh: pickle.dump(self,fh)

        future = [ {'day': t + timedelta(dd)}
                    for dd in range(1, self.dd_horizon + 1) ]

        forecast = self.snarimax_model.forecast(horizon=self.dd_horizon, xs=future)

        yhat = pd.Series(forecast,index=horizon)
        yhat = self.data.append(yhat)

        #print('predicting ', t.date())

        return yhat
