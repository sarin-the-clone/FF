
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# base temperature : 5 celsius for timothy
tbase = 5

def sos(dft):
    # returns start of the season
    # dft = pandas timeseries

    mv_avg = [1,1,1]
    out = pd.DataFrame(columns=dft.columns)
    dft5C = (dft >= tbase) *1
    for k in dft.columns:
        dft5C_conv = np.convolve(dft5C.loc[:,k], mv_avg,'same')
        dft5C_conv = (dft5C_conv == sum(mv_avg)) * 1
        dft5C_conv = (dft5C_conv.cumsum() >= 1) *1
        out0 = 0
        if dft5C_conv.sum() > 0:
            out0 = np.where(np.diff(dft5C_conv) == 1 )[0][0]+2
        out0 = dft.index[out0]
        out.loc['sos',k] = out0
        #print(k,out0)
    return out

def gdd(ts):
    # returns accumulated growth degree days
    # ts = pandas timeseries

    out = pad_ts(ts).copy()
    out[out<tbase]=0
    for k in out.columns:
        out.loc[:sos(out).loc['sos',k],k] = 0
        out[k] = out[k].cumsum()
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
