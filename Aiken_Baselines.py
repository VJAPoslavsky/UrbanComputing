#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
import random
import warnings
import argparse
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16


# ### I. Reusable functions for loading data and running models

# In[2]:


'''
Loads data for a single outbreak.
Arguments: Two filenames for CSV files, the first for the reports CSV and the second for the trends CSV.
Outputs: Two dataframes, one containing the reports and one containing the trends
'''
def load_data(reports_fname, trends_fname):
    reports = pd.read_csv(reports_fname, index_col=0)
    reports.index = pd.to_datetime(reports.index)
    if 'Epi Week' in reports.columns:
        reports = reports.drop('Epi Week', axis=1)
    trends = pd.read_csv(trends_fname, index_col=0)
    trends.index = pd.to_datetime(trends.index)
    return reports, trends


# In[3]:


'''
Converts a time-series into a supervised learning problem.
Inputs: Epidemiological timeseries (list/array/series), trends dataframe, list of autoregressive lags to be used as
predictors
Outputs: X, a matrix of predictors, and Y, a vector of targets
'''
# Convert timeseries to supervised learning problem
def to_supervised(y, trends, lags, max_train):
    if lags is not None:
        xcols = []
        for lag in lags:
            xcols.append(y.shift(lag))
        x = pd.concat(xcols, axis=1)
        x.columns = ['lag_' + str(lag) for lag in lags]
        nullrows = x[(pd.isnull(x).any(axis=1))].index
        if trends is not None:
            x = pd.merge(x, trends, on='date')
        x = x.drop(nullrows, axis=0)
        y = y.drop(nullrows, axis=0)
    elif trends is not None:
        x = trends
    return x, y


# In[4]:


'''
Performs train/test split on X and Y matrices.
Inputs: X, a matrix of predictors, Y, a vector of targets, a list of dates in the training period, and a list of dates
in the test period
Outputs: X and Y matrices for each train and test periods
'''
def split(x, y, dates_train, dates_test):
    min_train, max_train = dates_train
    min_test, max_test = dates_test
    x_train = x[(x.index >= min_train) & (x.index <= max_train)]
    x_test = x[(x.index >= min_test) & (x.index <= max_test)]
    y_train = y[(y.index >= min_train) & (y.index <= max_train)]
    y_test = y[(y.index >= min_test) & (y.index <= max_test)]
    return x_train, y_train, x_test, y_test


# In[5]:


'''
Given input data (epi timeseries and trends dataframe), a time horizon of prediction, and a set of prediction dates,
use earlier dates to train model and then uses regression to predict for each of the prediction dates.
'''
def run(epi, trends, th, predict_dates, use_lags=True, use_trends=True, weekly=True, random_seed=0):
    min_predict_date, max_predict_date = predict_dates[0], predict_dates[1]
    predictions = []
    coefs = []
    days_before = th*7 if weekly else th
    if use_lags and use_trends:
        x, y = to_supervised(epi, trends, [th], min_predict_date - pd.Timedelta(days=days_before))
    elif use_trends and not use_lags:
        x, y = to_supervised(epi, trends, None, min_predict_date - pd.Timedelta(days=days_before))
    elif use_lags and not use_trends:
        x, y = to_supervised(epi, None, [th], min_predict_date - pd.Timedelta(days=days_before))
    x_train, y_train, x_test, y_test = split(x, y, (epi.index[0], min_predict_date - pd.Timedelta(days=days_before)),
                                            (min_predict_date, max_predict_date))
    lr = LassoCV(max_iter=100000, cv=3, random_state=random_seed)
    lr.fit(x_train, y_train)
    return predict_dates, lr.predict(x_test), lr.coef_


# In[6]:


'''
Give input data (reports dataframe and trends dataframe) and a time horizon of prediction, runs theoretical linear
regression model using trends and ground truth time series from data frame.
'''
def run_model_theoretical(reports, trends, th, use_lags=True, use_trends=True, weekly=True, random_seed=0):
    epi = reports['groundtruth']
    dates, yhats, coefs = [], [], []
    predict_dates = epi.index[2*(th+1):] if weekly else epi.index[2*(7*(th+1)):]
    for predict_date in predict_dates:
        date, preds, coef = run(epi, trends, th, (predict_date, predict_date), use_lags, use_trends, weekly, random_seed=random_seed)
        dates.append(date[0])
        yhats.append(preds[0])
        coefs.append(coef)
    return predict_dates, yhats, coefs


# In[7]:


'''
Given input data (reports dataframe and trends dataframe) and a reporting delay (in weeks), runs linear regression
model for each report, in each case predicting until the end of the next report
'''
def run_model_practical(reports, trends, delay, use_lags=True, use_trends=True, weekly=True):
    reportids = reports.columns[:-1]
    results = {reportid:{'dates':None, 'yhats':None} for reportid in reportids}
    for r, reportid in enumerate(reportids):
        next_reportid = reportids[r+1] if r < len(reportids) - 1 else None
        epi = reports[reportid]
        min_predict_date = epi.index[len(epi.dropna()) - delay]
        max_predict_date = pd.to_datetime(next_reportid) if next_reportid is not None else epi.index[-1]
        predict_dates = epi[(epi.index >= min_predict_date) & (epi.index <= max_predict_date)].index
        dates, yhats = [], []
        for i, predict_date in enumerate(predict_dates):
            date, preds, _ = run(epi, trends, i+1, (predict_date, predict_date), use_lags, use_trends, weekly)
            dates.append(date[0])
            yhats.append(preds[0])
        results[reportid]['dates'] = dates
        results[reportid]['yhats'] = yhats
        results[reportid]['dates_report'] = epi.index[:len(epi.dropna())]
        results[reportid]['report'] = epi[:len(epi.dropna())]
    return results


# ### II. Run Models - Theoretical Version

# In[8]:


def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))


# In[9]:


def rrmse(x, y):
    return rmse(x, y)/np.mean(x)


# In[10]:


colors = {'AR':'#658E9C', 'GT':'#8CBA80', 'ARGO':'#CD5555'}
models = ['AR', 'GT', 'ARGO']
countries = ['AK',
    'AL',
    'AR',
    'AZ',
    'DE',
    'GA',
    'ID',
    'KS',
    'KY']
#countries = ['AK', 'AL', 'AR', 'AZ', 'DE', 'GA', 'ID', 'KS', 'KY', 'LA', 'MA', 'ME', 'MI', 'MN', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV']
#titles = ['Influenza in AK', 'Influenza in AL', 'Influenza in AR', 'Influenza in AZ', 'Influenza in DE', 'Influenza in GA', 'Influenza in ID', 'Influenza in KS', 'Influenza in KY', 'Influenza in LA', 'Influenza in MA', 'Influenza in MD', 'Influenza in ME', 'Influenza in MI', 'Influenza in MN', 'Influenza in NC', 'Influenza in ND', 'Influenza in NE', 'Influenza in NH', 'Influenza in NJ', 'Influenza in NM', 'Influenza in NV', 'Influenza in NY', 'Influenza in OH', 'Influenza in OR', 'Influenza in PA', 'Influenza in RI', 'Influenza in SC', 'Influenza in SD', 'Influenza in TN', 'Influenza in TX', 'Influenza in UT', 'Influenza in VA', 'Influenza in VT', 'Influenza in WA', 'Influenza in WI', 'Influenza in WV']
#countries = ['angola', 'colombia', 'drc', 'madagascar', 'yemen']
#titles = ['Yellow Fever in Angola', 'Zika in Colombia', 'Ebola in the DRC', 'Pneumonic Plague in Madagascar', 'Cholera in Yemen']


# In[11]:


'''
Given country, plotting axes (optional), and time horizon, run theoretical models for time horizon and plot
'''
def run_all_theoretical(country, line_plot_ax, heatmap_ax, th, weekly=True, models = ['AR', 'GT', 'ARGO']):
    random.seed(0)
    reports, trends = load_data('data/'  + country + '/' + country + 'reports.csv',
                                         'data/' + country + '/' + country + 'predictors.csv')
    results = {model:{} for model in models}
    results_series = reports[['groundtruth']]
    argo_coefs = []
    # Get results for each model
    for model in results.keys():
        print('Current model: ', model)
        use_lags = False if model == 'GT' else True
        use_trends = False if model == 'AR' else True
        for i in range(10):
            results[model]['dates'], yhats, coefs = run_model_theoretical(reports, trends, th,
                                                                                            use_lags, use_trends, weekly, random.randrange(0, 1000))
            results[model]['coefs'] = coefs
            if model == 'ARGO':
                argo_coefs.append(np.vstack(coefs))
        results[model]['corr'] = np.corrcoef(list(reports['groundtruth'].values)[-len(yhats):], yhats)[0][1]
        results[model]['rmse'] = rmse(list(reports['groundtruth'].values)[-len(yhats):], yhats)
        results[model]['rrmse'] = rrmse(list(reports['groundtruth'].values)[-len(yhats):], yhats)
        results[model]['yhats'] = yhats
        results_series = results_series.merge(pd.DataFrame(results[model], index=results[model]['dates'])[['yhats']],
                                            how='outer', left_index=True, right_index=True)
        results_series = results_series.rename({'yhats':model}, axis='columns')
        results_series.to_csv('results/' + country + '_' + str(th) + '.csv', index=False)
    # Plot of predictions
    if line_plot_ax is not None:
        line_plot_ax.fill_between(reports.index, 0, reports['groundtruth'], color='lightgrey', label='Ground truth')
        for model in results.keys():
            line_plot_ax.plot(results[model]['dates'], results[model]['yhats'], color=colors[model],
                              label = model + ' corr = %.2f ' % results[model]['corr'], linewidth=3)
        line_plot_ax.spines['right'].set_visible(False)
        line_plot_ax.spines['top'].set_visible(False)
        #line_plot_ax.legend(loc='best')
        line_plot_ax.set_xlim(reports.index[0], reports.index[-1])
        line_plot_ax.set_ylim(0, max(reports['groundtruth'])*1.3)
    if heatmap_ax is not None:
        for j in range(len(argo_coefs)):
            for i in range(len(reports.index) - len(yhats)):
                to_add = np.zeros(len(argo_coefs[j][0]))
                argo_coefs[j] = np.vstack([to_add, argo_coefs[j]])
        coefs = np.mean(np.stack(argo_coefs), axis=0)
        coefs = np.array(coefs).T
        #mask = pd.DataFrame(coefs).isnull()
        heatmap_ax.set_xlim(min(reports.index), max(reports.index))
        sns.heatmap(coefs, cmap='RdBu_r', vmin=-1, vmax=1, ax=heatmap_ax, cbar=False)
        heatmap_ax.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
        heatmap_ax.set_yticklabels(['Autoregressive term'] + list(trends.columns))
        for tick in heatmap_ax.get_yticklabels():
            tick.set_rotation(0)
    return results


# In[ ]:


'''
Run theoretical models for each outbreak for time horizons 1 and 2. Save evaluations for heatmaps.
'''
ths = [2, 1]
evaluations = {country:{} for country in countries}
for c, country in enumerate(countries):
    print(country)
    if True:
        df_corr = pd.DataFrame(index=models, columns=ths)
        df_rmse = pd.DataFrame(index=models, columns=ths)
        df_rrmse = pd.DataFrame(index=models, columns=ths)
        #fig, ax = plt.subplots(2, figsize=(20, 9))
        for th in [2, 1]:
            #a = ax[0] if th == 1 else ax[1]
            fig, a = plt.subplots(1, figsize=(20, 5))
            weekly = False if country == 'madagascar' else True
            results = run_all_theoretical(country, a, None, th, weekly)
            df_corr[th] = [results[model]['corr'] for model in models]
            df_rmse[th] = [results[model]['rmse'] for model in models]
            df_rrmse[th] = [results[model]['rrmse'] for model in models]
            a.set_ylabel('Cases')
            #a.set_title('Assuming Reporting Delay of ' + str(th) + ' Weeks')
            a.set_title('Digital Epidemiological Modeling of ' + 'Influenza in ' + countries[c], fontsize='x-large')
            a.legend(loc='best')
            plt.tight_layout()
            #plt.subplots_adjust(top=0.88)
            #plt.suptitle('Digital Epidemiological Modeling of ' + titles[c], fontsize='x-large')
            plt.savefig(country + '_' + str(th) + '.png')
        #plt.savefig('newnewfigures/theoretical/' + country + '_both' + '.png')
        evaluations[country]['corr'] = df_corr
        evaluations[country]['rmse'] = df_rmse
        evaluations[country]['rrmse'] = df_rrmse


# In[ ]:


'''
Figure 1 in paper
'''
ths = [2, 1]
evaluations = {country:{} for country in countries}
fig, ax = plt.subplots(5, 2, figsize=(20, 12), sharex=False, sharey=False)
for c, country in enumerate(countries):
    print(country)
    df_corr = pd.DataFrame(index=models, columns=ths)
    df_rmse = pd.DataFrame(index=models, columns=ths)
    for th in [2, 1]:
        a = ax[c][th-1]
        weekly = False if country == 'madagascar' else True
        results = run_all_theoretical(country, a, None, th, weekly)
        df_corr[th] = [results[model]['corr'] for model in models]
        df_rmse[th] = [results[model]['rmse'] for model in models]
        months = mdates.MonthLocator()
        months_format = mdates.DateFormatter('%b %y')
        a.xaxis.set_major_locator(months)
        a.xaxis.set_major_formatter(months_format)
    evaluations[country]['corr'] = df_corr
    evaluations[country]['rmse'] = df_rmse
ax[0, 1].legend(loc='upper right', prop={'size': 10})
ax[0, 0].set_ylabel('New Cases (Weekly)', size='small')
ax[1, 0].set_ylabel('New Cases (Weekly)', size='small')
ax[2, 0].set_ylabel('New Cases (Weekly)', size='small')
ax[3, 0].set_ylabel('New Cases (Daily)', size='small')
ax[4, 0].set_ylabel('New Cases (Weekly)', size='small')
for i in range(5):
    ax[i, 1].get_yaxis().set_visible(False)
plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.97])
plt.figtext(0.5, 0.962, 'Yellow Fever in Angola', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.77, 'Zika in Colombia', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.572, 'Ebola in the DRC', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.38, 'Plague in Madagascar', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.19, 'Cholera in Yemen', ha='center', va='center', size='medium')
plt.figtext(0.25, 0.98, 'Assuming reporting delay of 1 week', ha='center', va='center', size='large')
plt.figtext(0.75, 0.98, 'Assuming reporting delay of 2 weeks', ha='center', va='center', size='large')
#plt.figtext(0.01, 0.99, 'a', ha='center', va='center', size='large', weight='bold')
plt.savefig('results1_part1.png')
