import pandas as pd
import matplotlib as plt
import lol


disp = True

# load temperature timeseries from 'dftempall.pickle'
# sourceID = weather station name
df_temp_all = pd.read_pickle('dftempall.pickle')
if disp: df_temp_all.head()
if disp: df_temp_all['sourceID'].unique()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
#  ran    d                                                                                                                                                                                                                                                                                                    omly select a sourceID, i.e. doing one model per weather station, just to test for now
# df_temp_s = temperature for weather station s
s = df_temp_all['sourceID'].sample().values[0]
df_temp_s = df_temp_all[df_temp_all['sourceID']== s][['value','dd','yy']]
df_temp_s.index=df_temp_s['dd']
df_temp_s['mmdd'] = df_temp_s['dd'].apply(lambda x: str(x.month).zfill(2) + '-' + str(x.day).zfill(2))
if disp: df_temp_s[['value',	'yy','mmdd']].pivot(columns = 'yy',index='mmdd').plot()

# calculate accumulated gdd for station s
df_aGDD_s = pd.DataFrame()
for y in df_temp_s['yy'].unique():
    tmp01 = df_temp_s[df_temp_s['yy']==y][['value']]
    tmp01 = lol.gdd(tmp01)
    df_aGDD_s = df_aGDD_s.append(tmp01)

df_temp_s = df_temp_s[['value']]

if disp: df_temp_s.plot()
if disp: df_aGDD_s.plot()




