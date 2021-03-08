import pandas as pd
import lol

df_temp_all = pd.read_pickle('dftempall.pickle')

df_temp_all.head()
df_temp_all['sourceId'].unique()

s = df_temp_all['sourceID'].sample().values[0]

df_temp_s = df_temp_all[df_temp_all['sourceID'] == s]


