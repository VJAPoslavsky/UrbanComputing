#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import datetime
from keras import backend as K


# In[11]:


def load_data(state, path='data/US_FULL/'):
    filename_reports = state + '/' + state + 'reports.csv'
    filename_predictors = state + '/' + state + 'predictors.csv'

    df_X = pd.read_csv(path + filename_predictors)
    df_y = pd.read_csv(path + filename_reports)

    df_X['groundtruth'] = df_y['groundtruth']

    return df_X


# In[4]:


def feature_selection(df):
    highest_corr = df.corr()['groundtruth'].sort_values(ascending=False).head(20)
    corr_df = pd.DataFrame()
    corr_df['date'] = df['date']
    for i in range(1, 11):
        corr_df[highest_corr.index[i]] = df[highest_corr.index[i]]
    corr_df['groundtruth'] = df['groundtruth']
    corr_df.index = corr_df['date']
    corr_df = corr_df.drop(columns=['date'])
    return corr_df


# In[5]:


def create_supervised_data(df, lookback, lag):
    index = 0
    arr = []
    num_cols = len(df.columns)
    for row in df.itertuples():
        if index < lookback:
            index = index + 1
            continue
        new_row = np.array([])
        for i in range(1, lookback+1):
            for feature in range(len(df.columns)):
                new_row = np.append(new_row, df.iloc[index-i,feature])
        new_row = np.append(new_row, row.groundtruth)
        arr.append(new_row)
        index = index + 1
    final_df = pd.DataFrame(arr, index=df.iloc[lookback:,:].index)
    final_df = final_df.iloc[:, num_cols*lag:]

    return pd.DataFrame(arr, index=df.iloc[lookback:,:].index)


# In[6]:


def split_train_test(df):
    len_df = int(len(df)/2)
    X_train = df.iloc[:len_df,:-1]
    X_test = df.iloc[len_df:,:-1]
    y_train = df.iloc[:len_df,-1:]
    y_test = df.iloc[len_df:,-1:]

    return X_train, y_train, X_test, y_test


# In[7]:


def add_predictors(df, state, path = 'data/SpatialPredictors/'):
    years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    df['year'] = df.index
    df['year'] = df['year']
    df['year'] = df['year'].str[-4:]
    dfpred = pd.read_excel(path + years[0] + 'data.xlsx', index_col=0, engine="openpyxl")
    for column in dfpred.index:
        df[column] = 1
    for year in years:
        tmp = pd.read_excel(path + year + 'data.xlsx', index_col=0, engine="openpyxl")
        tmp = tmp[state]
        index = 0
        if year == '2010':
            for idx in tmp.index:
                df.loc[df['year'] == '2009', idx] = tmp[idx]
        for idx in tmp.index:
            df.loc[df['year'] == year, idx] = tmp[idx]
    df = df.drop(columns=['year'])
    return df


# In[13]:


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


lookbacks = [8,52]
lags = [1,2]

for state in states:
    print('==========Starting running State ', state, '==========')
    for lookback in lookbacks:
        print('==========Using lookback ', str(lookback), '==========')
        for lag in lags:
            print('==========Using lag ', str(lag), '==========')
            for spatial in ['', 'spatial']:
                if spatial == '':
                    final_df = pd.DataFrame()
                df = load_data(state)
                dates = df['date']
                df = feature_selection(df)
                df_X = df.iloc[:,:-1]
                df_y = df.iloc[:,-1:]
                if spatial == 'spatial':
                    df_X = add_predictors(df_X, state)
                cols = df_X.columns
                cols = np.append(cols, 'groundtruth')
                df_X.columns
                scalerX = MinMaxScaler(feature_range=(-1, 1))
                scalerY = MinMaxScaler(feature_range=(-1, 1))
                scaledX = scalerX.fit_transform(df_X.values)
                scaledY = scalerY.fit_transform(df_y.values)
                scaled = pd.DataFrame(scaledX)
                scaled['groundtruth'] = scaledY
                scaled.columns = cols
                scaled.index = df.index
                new_df = create_supervised_data(scaled,lookback, lag)
                X_train, y_train, X_test, y_test = split_train_test(new_df)

                X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

                regressorGRU = Sequential()
                # First GRU layer with Dropout regularisation
                regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
                regressorGRU.add(Dropout(0.2))
                # Second GRU layer
                regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
                regressorGRU.add(Dropout(0.2))
                # Third GRU layer
                regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
                regressorGRU.add(Dropout(0.2))
                # Fourth GRU layer
                regressorGRU.add(GRU(units=50, activation='tanh'))
                regressorGRU.add(Dropout(0.2))
                # The output layer
                regressorGRU.add(Dense(units=1))
                # Compiling the RNN
                regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
                # Fitting to the training set
                regressorGRU.fit(X_train,y_train,epochs=1000,batch_size=40)

                y_pred_GRU = regressorGRU.predict(X_test)
                y_pred_GRU = scalerY.inverse_transform(y_pred_GRU)
                y_test = scalerY.inverse_transform(y_test)

                if spatial == '':
                    final_df['groundtruth'] = np.ravel(y_test)

                K.clear_session()

                # LSTM
                regressorLSTM = Sequential()
                # firt LSTM layer
                regressorLSTM.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
                regressorLSTM.add(Dropout(0.2))
                # second LSTM layer
                regressorLSTM.add(LSTM(units=50, return_sequences=True))
                regressorLSTM.add(Dropout(0.2))
                # third LSTM layer
                regressorLSTM.add(LSTM(units=50, return_sequences=True))
                regressorLSTM.add(Dropout(0.2))
                # fourth LSTM layer
                regressorLSTM.add(LSTM(units=50))
                regressorLSTM.add(Dropout(0.2))
                # output layer
                regressorLSTM.add(Dense(units=1))
                # compile the RNN
                regressorLSTM.compile(optimizer='rmsprop', loss='mean_squared_error')
                # fit to the training set
                regressorLSTM.fit(X_train, y_train, epochs=1000, batch_size=40)

                y_pred_LSTM = regressorLSTM.predict(X_test)
                y_pred_LSTM = scalerY.inverse_transform(y_pred_LSTM)

                K.clear_session()

                final_df[spatial + 'GRU'] = y_pred_GRU
                final_df[spatial + 'LSTM'] = y_pred_LSTM
            final_df.to_csv('results/' + state + '_LSTMandGRU_Lookback' + str(lookback) + '_Lag' + str(lag) + '.csv', index=False)
