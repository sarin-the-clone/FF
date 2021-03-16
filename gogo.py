import pandas as pd
import matplotlib as plt
import lol


disp = False

# load temperature timeseries from 'dftempall.pickle'
# sourceID = weather station name
df_temp_all = pd.read_pickle('dftempall.pickle')
if disp: df_temp_all.head()
if disp: df_temp_all['sourceID'].unique()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                   omly select a sourceID, i.e. doing one model per weather station, just to test for now
# df_temp_s = temperature for weather station 
s = df_temp_all['sourceID'].sample().values[0] #selects a random weather station
df_temp_s = df_temp_all[df_temp_all['sourceID']== s][['value','dd','yy']]
df_temp_s.index=df_temp_s['dd']
if disp: df_temp_s['mmdd'] = df_temp_s['dd'].apply(lambda x: str(x.month).zfill(2) + '-' + str(x.day).zfill(2))
if disp: df_temp_s[['value',	'yy','mmdd']].pivot(columns = 'yy',index='mmdd').plot()

# calculate accumulated gdd for station s and interpolate missing values :
df_aGDD_s = pd.Series([],dtype='float64')
df_temp_s_pad = pd.Series([],dtype='float64')
for y in df_temp_s['yy'].unique():
    tmp00 = df_temp_s[df_temp_s['yy']==y][['value']]
    tmp01 = lol.gdd(pd.Series(tmp00.values.flatten(),index=tmp00.index))
    df_aGDD_s = df_aGDD_s.append(tmp01)
for y in df_temp_s['yy'].unique():
    tmp00 = df_temp_s[df_temp_s['yy']==y][['value']]
    tmp02 = lol.pad_ts(pd.Series(tmp00.values.flatten(),index=tmp00.index))
    df_temp_s_pad = df_temp_s_pad.append(tmp02)

df_t = df_temp_s_pad
if disp: df_t.plot()
if disp: df_aGDD_s.plot()


# dhist = #days of temperature input
# dhoriz = #days in the future to predict
# n_years_2train = #years for the training set
dhist, dhoriz, n_years_2train = int(365/2), int(365/4), 3

# generates lstm model for s (takes about 1-2 minutes) :
src = lol.weather_station(df_t, dhist, dhoriz, n_years_2train, 'lstm')


##### plot input temperatures and forecast #####
start = src.data.index[src.year_train*365+src.dd_historic+src.dd_horizon] #first day after validation data
end =  df_t.index[-src.dd_horizon-1]
for today in pd.date_range(start,end):
    plt.cla()
    ind_curr = pd.date_range(today-timedelta(dhist-1),today)
    x_input = df_t.loc[ind_curr]
    yhat = yhat = src.lstm_predict(x_input)

    x_input.plot(style='b-')
    df_t.loc[yhat.index].plot(style='g-')
    yhat.plot(style='r--')
    plt.pause(0.0001)


##### plot accumulated temperatures and forecast #####
Harvest, Harvest_hat = start, start
for today in pd.date_range(start,end):
    plt.cla()
    ind_curr = pd.date_range(today-timedelta(dhist-1),today)
    x_input = df_t.loc[ind_curr]
    yhat = yhat = src.lstm_predict(x_input)
    ind_pred = yhat.index
    yhat = lol.gdd(df_t.loc[pd.date_range(today-timedelta(365),today)].append(yhat))

    ind_curr_year = df_aGDD_s[df_aGDD_s.index. year == today.year].index
    Harvest = df_aGDD_s[ind_curr_year][df_aGDD_s[ind_curr_year] <= 500].index[-1].date()
    ind_curr_year = yhat[yhat.index. year == today.year].index
    Harvest_hat = yhat[ind_curr_year][yhat[ind_curr_year] <= 500].index[-1].date()

    plt.title(['today: '+str(today.date()), 'HT: ' + str(Harvest), 'HT_hat: ' + str(Harvest_hat), 
               '(today - HT,HT_hat - HT): (' + str((Harvest - today.date()).days) + ',' + str((Harvest_hat - Harvest).days) + ')' ])
    df_aGDD_s[ind_curr].plot(style='b-')
    df_aGDD_s.loc[ind_pred].plot(style='g-')
    yhat[ind_pred].plot(style='r--')
    plt.legend('past temperatures', 'future temperatures', 'forecast')
    plt.pause(0.0001)

