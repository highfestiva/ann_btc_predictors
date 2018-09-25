#!/usr/bin/env python3

'''NOTE: I took the liberty of fixing the off-by-one bug that made this code look useful.

   The file data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27 must be downloaded separately.
   https://www.kaggle.com/mczielinski/bitcoin-historical-data/data
   
   This script is stolen. There are some nice explanations in his original tut here:
   https://github.com/activewizardslab/time-series/blob/master/Bitcoin_price_forecasting_with_LSTM_and_GRU-v.2.ipynb
   (Even though it does not work.)
   '''

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
# from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go


epochs = 1000
patience = 200
batch_size = 65536


########################################
# load and extract train and test data

data = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')

data['Timestamp'] = pd.to_datetime(data['Timestamp'],unit='s')
data.set_index('Timestamp', inplace=True)
agg_price = data['Weighted_Price'].resample('15min').mean()
agg_pct = agg_price.pct_change()
agg_pct = agg_pct[agg_pct.index >= '2014-01-01']
print(agg_price.tail())

l = int(len(agg_pct)*0.8)
df_train = agg_pct[:l]
df_test = agg_pct[l:]

print('training and testing sizes:', len(df_train), len(df_test))


########################################
# exploratory analysis

# convert to DataFrame
working_data = pd.concat([df_train, df_test])
working_data = working_data.reset_index()
working_data = working_data.set_index('Timestamp')

# seasonal decomposition
s = sm.tsa.seasonal_decompose(working_data.Weighted_Price.values, freq=60)

trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
    line = dict(color = ('rgb(244, 146, 65)'), width = 4))
trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
    line = dict(color = ('rgb(66, 244, 155)'), width = 2))

trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
    line = dict(color = ('rgb(209, 244, 66)'), width = 2))

trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
    line = dict(color = ('rgb(66, 134, 244)'), width = 2))

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
# py.plot(fig, filename='.seasonal_decomposition.html')

# autocorrelation
# plt.figure(figsize=(15,7))
# ax = plt.subplot(211)
# sm.graphics.tsa.plot_acf(working_data.Weighted_Price.values.squeeze(), lags=48, ax=ax)
# ax = plt.subplot(212)
# sm.graphics.tsa.plot_pacf(working_data.Weighted_Price.values.squeeze(), lags=48, ax=ax)
# plt.tight_layout()
# plt.show()

# pick data again (with new indexing this time)
df_train = working_data[:len(df_train)]
df_test = working_data[len(df_train):]


########################################
# data preparation

def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# reshape, scale and look-back
from sklearn.preprocessing import MinMaxScaler

training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
test_set = df_test.values
test_set = np.reshape(test_set, (len(test_set), 1))

# scale datasets
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)
test_set = scaler.transform(test_set)

# create datasets which are suitable for time series forecasting
look_back = 1
X_train, Y_train = create_lookback(training_set, look_back)
X_test, Y_test = create_lookback(test_set, look_back)

# reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))


########################################
# setup and train LSTM

# initialize sequential model, add 2 stacked LSTM layers and densely connected output neuron
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))

# compile and fit the model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False, 
                    validation_data=(X_test, Y_test), 
                    callbacks = [EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=patience, verbose=1)])

########################################
# plot training accuracy

trace1 = go.Scatter(
    x = np.arange(0, len(history.history['loss']), 1),
    y = history.history['loss'],
    mode = 'lines',
    name = 'Train loss',
    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
)
trace2 = go.Scatter(
    x = np.arange(0, len(history.history['val_loss']), 1),
    y = history.history['val_loss'],
    mode = 'lines',
    name = 'Test loss',
    line = dict(color=('rgb(244, 146, 65)'), width=2)
)

data = [trace1, trace2]
layout = dict(title = 'Train and Test Loss during training', 
              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))
fig = dict(data=data, layout=layout)
py.plot(fig, filename='.training_process.html')
print('LSTM training complete')
print()

# add one additional data point to align shapes of the predictions and true labels
X_test = np.append(X_test, scaler.transform(working_data.iloc[-1][0]))
X_test = np.reshape(X_test, (len(X_test), 1, 1))

# get predictions and then make some transformations to be able to calculate RMSE properly in USD
prediction = model.predict(X_test)
prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
prediction2_inverse = np.array(prediction_inverse[:,0][:-1])
Y_test2_inverse = np.array(Y_test_inverse[:,0])

RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
print('test RMSE: %.3f' % RMSE)


def price_scatter(x, y, price, **kwargs):
    x = x[-len(y):]
    price = price.values[-len(y)-1:-1]
    assert len(x) == len(y) == len(price)
    # y = price * (y+1)
    return go.Scatter(x=x, y=y, **kwargs)


# plot in USD with dates
test_dates = df_test.index
trace1 = price_scatter(test_dates, Y_test2_inverse, agg_price, name= 'Actual Price', 
                   line=dict(color=('rgb(66, 244, 155)'), width=2))
trace2 = price_scatter(test_dates, prediction2_inverse, agg_price, name= 'Predicted Price',
                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))
data = [trace1, trace2]
layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.plot(fig, filename='.results_demonstrating1.html')


########################################
# improve accuracy using GRU model

# This function prepares random train/test split, 
# scales data with MinMaxScaler, create time series labels (Y)
def get_split(working_data, n_train, n_test, look_back = 1):
    # get a point from which we start to take train dataset and after it - test dataset
    start_point = randint(0, (len(working_data)-n_test-n_train))
    df_train = working_data[start_point:start_point+n_train]
    df_test = working_data[start_point+n_train:start_point+n_train+n_test]

    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    test_set = df_test.values
    test_set = np.reshape(test_set, (len(test_set), 1))
    
    # scale datasets
    scaler_cv = MinMaxScaler()
    training_set = scaler_cv.fit_transform(training_set)
    test_set = scaler_cv.transform(test_set)

    # create datasets which are suitable for time series forecasting
    X_train, Y_train = create_lookback(training_set, look_back)
    X_test, Y_test = create_lookback(test_set, look_back)

    # reshape datasets so that they will be ok for the requirements of the models in Keras
    X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
    
    return X_train, Y_train, X_test, Y_test, scaler_cv, start_point

# This function takes datasets from the previous function as input and train model using these datasets
def train_model(X_train, Y_train, X_test, Y_test):
    # initialize sequential model, add bidirectional LSTM layer and densely connected output neuron
    model = Sequential()
    model.add(GRU(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    
    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle = False, 
                    validation_data=(X_test, Y_test), verbose=0,
                    callbacks = [EarlyStopping(monitor='val_loss',min_delta=5e-5,patience=patience,verbose=0)])
    return model

# This function uses trained model and test dataset to calculate RMSE
def get_rmse(model, X_test, Y_test, scaler, start_point, working_data, n_train):
    # add one additional data point to align shapes of the predictions and true labels
    X_test = np.append(X_test, scaler.transform(working_data.iloc[start_point+n_train+len(X_test)][0]))
    X_test = np.reshape(X_test, (len(X_test), 1, 1))
    
    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_test)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
    prediction2_inverse = np.array(prediction_inverse[:,0][:-1])
    Y_test2_inverse = np.array(Y_test_inverse[:,0])
    
    #calculate RMSE
    RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
    return RMSE, prediction2_inverse

def workflow(working_data, get_split, train_model, get_rmse,n_train = 250,n_test = 50,look_back = 1):
    X_train, Y_train, X_test, Y_test, scaler, start_point = get_split(working_data, n_train, n_test)
    model = train_model(X_train, Y_train, X_test, Y_test)
    RMSE, predictions = get_rmse(model, X_test, Y_test, scaler, start_point, working_data, n_train)
    return RMSE, predictions

RMSE, predictions = workflow(working_data, get_split, train_model, get_rmse, n_train=len(df_train), n_test=len(df_test))
print('Test GRU model RMSE: %.3f' % RMSE)

# This function is used to repeat the workflow ten times and to calculate average RMSE
def cross_validate(working_data,get_split,train_model,get_rmse,workflow,n_train = 250,n_test = 50,look_back = 1):
    rmse_list = []
    for i in range(10):
        print('Iteration:', i+1)
        RMSE, _ = workflow(working_data, get_split, train_model, get_rmse, n_train, n_test, look_back)
        rmse_list.append(RMSE)
        print('Test RMSE: %.3f' % RMSE)
    mean_rmse = np.mean(rmse_list)
    return mean_rmse, rmse_list

mean_rmse, rmse_list = cross_validate(working_data, get_split, train_model, get_rmse, workflow)
print('Average RMSE: ', mean_rmse)


predictions_new = predictions - mean_rmse
RMSE_new = sqrt(mean_squared_error(Y_test2_inverse, predictions_new))
print('Test GRU model RMSE_new: %.3f' % RMSE_new)


trace1 = price_scatter(x=test_dates, y=Y_test2_inverse, price=agg_price, name= 'Actual Price', 
                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))
trace2 = price_scatter(x=test_dates, y=predictions_new, price=agg_price, name= 'Predicted Price',
                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))
data = [trace1, trace2]
layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.plot(fig, filename='.results_demonstrating2.html')


def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon = 1e-8): 
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2 + epsilon)) * 100
SMAPE = symmetric_mean_absolute_percentage_error(Y_test2_inverse, predictions_new)
print('Test SMAPE (percentage): %.3f' % SMAPE)
