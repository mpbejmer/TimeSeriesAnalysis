# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:42:21 2021

Not my code.
"""


#Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters



# load and pre-process data

dataset = pd.read_csv("AAPL.csv")
dataset['Mean'] = (dataset['Low'] + dataset['High'])/2

# preparing the dataset by shifting open , close, Low, High  by 1
steps = -1
dataset_for_prediction = dataset.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift(steps)

dataset_for_prediction=dataset_for_prediction.dropna()


from pandas.tseries.offsets import BDay
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction = dataset_for_prediction.set_index('Date')
#dataset_test.index.freq='B'



dataset_for_prediction['Mean'].plot(color='green', figsize=(15,2))
plt.legend([  'Mean'])
plt.title(" Apple Stock Value")


# normalizing input features
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High','Open', 'Close', 'Volume', 'Adj Close', 'Mean']])
scaled_input = pd.DataFrame(scaled_input)
X = scaled_input + 0.0001

sc_out = MinMaxScaler(feature_range=(0, 1))
scaled_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaled_output = pd.DataFrame(scaled_output)
Y = scaled_output

X = X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Adj Close', 6:'Mean'})
X.index = dataset_for_prediction.index

Y = Y.rename(columns={0:'Stock Price next day'})
Y.index=dataset_for_prediction.index

# decompose time series into: Y[t] = Trend[t] + Seasonal[t] + residuals[t]
import statsmodels.api as sm
seas_d = sm.tsa.seasonal_decompose(X['Mean'], model='additive', period=365)
#seas_d = sm.tsa.seasonal_decompose(X['Mean'], model='multiplicative') # need to define period also
fig = seas_d.plot()
fig.set_figheight(10)
plt.show()

train_size = int(len(dataset) * 0.80)
test_size = int(len(dataset)) - train_size

train_X, train_Y = X[:train_size].dropna(), Y[:train_size].dropna()
test_X, test_Y = X[train_size:].dropna(), Y[train_size:].dropna()

test_X.columns

Y_test=Y['Stock Price next day'][train_size:].dropna()


#Correlograms
fig,ax = plt.subplots(2,1, figsize=(10,5))
fig = sm.tsa.graphics.plot_acf(Y_test, lags=50, ax=ax[0])
fig = sm.tsa.graphics.plot_pacf(Y_test, lags=50, ax=ax[1])
plt.show()


# How to find the optimal model:
# from pmdarima.arima import auto_arima
# step_wise=auto_arima(train_Y, 
#                      start_p=1,  start_q=1, 
#                      max_p=7,  max_q=7, 
#                      d=1, max_d=7,
#                      trace=True, 
#                      error_action='ignore', 
#                      suppress_warnings=True, 
#                      stepwise=True)


# TRAIN MODEL Using SARIMAX(p, d, q) = (0, 1, 1)
from statsmodels.tsa.statespace.sarimax import SARIMAX

model= SARIMAX(train_Y, 
               exog=train_X,            
              order=(0,1,1),
              enforce_invertibility=False, enforce_stationarity=False)

results = model.fit()
predictions = results.predict(start=train_size, end=train_size+test_size+(steps)-1, exog=test_X)
predictions.index = test_Y.index

forecast_1 = results.forecast(steps=test_size-1, exog=test_X)
forecast_1.index = test_Y.index

act = Y_test.to_frame()

# forecast_apple= pd.DataFrame(forecast_1)
# forecast_apple.reset_index(drop=True, inplace=True)
# forecast_apple.index=test_X.index
# forecast_apple['Actual'] = scaled_output.iloc[train_size:, 0].values
# forecast_apple.rename(columns={0:'Forecast'}, inplace=True)

# forecast_apple['Forecast'].plot(legend=True)
# forecast_apple['Actual'].plot(legend=True)

predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Stock Price next day'].values
predictions.rename(columns={0:'Pred'}, inplace=True)


predictions['Actual'].plot(figsize=(20,8), legend=True, color='blue')
predictions['Pred'].plot(legend=True, color='red', figsize=(20,8))




