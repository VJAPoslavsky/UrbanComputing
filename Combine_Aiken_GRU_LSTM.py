#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from scipy.stats import wilcoxon


# In[14]:


states = [
    'AK',
    'AL',
    'AR',
    'AZ',
    'DE',
    'GA',
    'ID',
    'KS',
    'KY',
    'LA',
    'MA',
    'MD',
    'ME',
    'MI',
    'MN',
    'NC',
    'ND',
    'NE',
    'NH',
    'NJ',
    'NM',
    'NV',
    'NY',
    'OH',
    'OR',
    'PA',
    'RI',
    'SC',
    'SD',
    'TN',
    'TX',
    'UT',
    'VA',
    'VT',
    'WA',
    'WI',
    'WV'
]

lags = ['1', '2']


# In[15]:


for state in states:
    for lag in lags:
        #Load DataFrames
        df_Aiken = pd.read_csv('Baselines_Aiken/' + state + '_' + lag + '.csv')
        df_8 = pd.read_csv('GRU_LSTM/' + state + '_LSTMandGRU_Lookback8_Lag' + lag + '.csv')
        df_52 = pd.read_csv('GRU_LSTM/' + state + '_LSTMandGRU_Lookback52_Lag' + lag + '.csv')

        #Remove NaN rows
        df_Aiken = df_Aiken.dropna()
        df_8 = df_8.dropna()
        df_52 = df_52.dropna()
        min_length = min([len(df_Aiken), len(df_8), len(df_52)])

        #Make all DataFrames same length
        df_Aiken = df_Aiken.iloc[-min_length:,:]
        df_8 = df_8.iloc[-min_length:,:]
        df_52 = df_52.iloc[-min_length:,:]

        #Load dates
        dates = pd.read_csv('Raw/' + state + '/' + state + 'reports' + '.csv')
        dates = dates.iloc[-min_length:,:]
        dates = dates['date']

        #Create combined DataFrame
        combined_df = df_Aiken.copy()
        combined_df.index = dates
        combined_df['GRU_8'] = df_8['GRU'].to_numpy()
        combined_df['GRU_52'] = df_52['GRU'].to_numpy()
        combined_df['LSTM_8'] = df_8['LSTM'].to_numpy()
        combined_df['LSTM_52'] = df_52['LSTM'].to_numpy()
        combined_df['GRU_8_Spatial'] = df_8['spatialGRU'].to_numpy()
        combined_df['GRU_52_Spatial'] = df_52['spatialGRU'].to_numpy()
        combined_df['LSTM_8_Spatial'] = df_8['spatialLSTM'].to_numpy()
        combined_df['LSTM_52_Spatial'] = df_52['spatialLSTM'].to_numpy()

        combined_df.to_csv('Combined/' + state + '_' + lag + '_combined.csv')


# In[ ]:


df = pd.read_csv('Combined/' + state + '_2_combined.csv')
cols = df.columns[1:]


for state in states:
    df = pd.read_csv('Combined/' + state + '_2_combined.csv')
    cols = df.columns[1:]
    


# In[50]:


df = pd.read_csv('Combined/AK_2_combined.csv', index_col=0)
cols = df.columns[1:]
df_cols = []
for col in cols:
    df_cols.append(col + '_MSE')
    df_cols.append(col + '_RMSE')
    df_cols.append(col + '_CORR')


# In[83]:


df = pd.read_csv('Combined/AK_1_combined.csv', index_col=0)
cols = df.columns[1:]
df_cols = []
for col in cols:
    df_cols.append(col + '_MSE')
    df_cols.append(col + '_RMSE')
    df_cols.append(col + '_CORR')
    df_cols.append(col + '_WIL')

df_rows = []

for state in states:
    df_row = []
    df = pd.read_csv('Combined/' + state + '_1_combined.csv', index_col=0)
    for col in cols:
        y_pred = df[col].to_numpy()
        y_test = df['groundtruth'].to_numpy()
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        CORR = pearsonr(y_test, y_pred)[0]
        if col != 'AR':
            WIL = wilcoxon(y_pred, df['AR'].to_numpy())[0]
        else:
            WIL = np.nan
        df_row.append(MSE)
        df_row.append(RMSE)
        df_row.append(CORR)
        df_row.append(WIL)
    df_rows.append(df_row)

print(df_rows)


# In[84]:


tmp_df = pd.DataFrame(df_rows, columns=df_cols, index=states)
tmp_df
tmp_df.to_csv('Metrics_Combined_1.csv')

