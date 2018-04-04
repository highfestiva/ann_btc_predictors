#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Import data
coin_fname = 'data/bitcoin_usdt_1m.json'
data = pd.read_json(coin_fname, orient='split')
model_fname = './.ff_checkpoints/bitcoin_model.ckpt'

# Drop unrelated variables
data = data.drop(['time','time_close','ignore'], axis=1)
close_column_index = data.columns.get_loc('close')

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
train_end = int(0.8*n)
test_start = train_end + 1
test_end = n
data_train = data[train_start:train_end, :]
data_test = data[test_start:test_end, :]

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
def inverse_price_transform(scaler, colidx, arr):
    factor = 0.5 * scaler.data_range_[colidx] # [-1;+1] needs halving
    center = 0.5 * scaler.data_range_[colidx] # get back up to the middle of the range
    min = scaler.data_min_[colidx] # add the minima to inverse
    return arr * factor + (center + min)

# Build X and y
X_train = data_train[:-1, :] # previous state
y_train = data_train[+1:, close_column_index] # predict this next state
X_test = data_test[:-1, :] # previous state
y_test = data_test[+1:, close_column_index] # predict this next state
X_test_future = data_test[:, :] # complete range, including last element, for predicting into the future
full_plot = np.concatenate([data_train[:, close_column_index], data_test[:, close_column_index]])
full_train_plot = np.copy(data_train[:,close_column_index]) # include first and last point of training data
full_train_plot[:] = np.nan # empty, so we just draw the last part (test+prediction)
full_train_plot = np.append(full_train_plot, [X_test[0,close_column_index]]) # include the first point of the test input data used for predicting the first point of the test output data

# Number of columns (used to be stocks) in training data
n_columns = p#X_train.shape[1]

# who knows
tf.reset_default_graph()

# Neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
sess = tf.Session()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_columns])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_columns, n_neurons_1]), name='W_hidden_1')
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]), name='bias_hidden_1')
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]), name='W_hidden_2')
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]), name='bias_hidden_2')
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]), name='W_hidden_3')
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]), name='bias_hidden_3')
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]), name='W_hidden_4')
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]), name='bias_hidden_4')

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]), name='W_out')
bias_out = tf.Variable(bias_initializer([1]), name='bias_out')

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)
#opt = tf.train.GradientDescentOptimizer(0.2).minimize(mse)

# load/save sessions
saver = tf.train.Saver()

# load or init
try:
    saver.restore(sess, model_fname)
    print('model restored')
except:
    sess.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(inverse_price_transform(scaler, close_column_index, full_plot))
line2, = ax1.plot(inverse_price_transform(scaler, close_column_index, np.concatenate([full_train_plot, y_test * 0.5, [y_test[-1]]]))) # include first X of test as anchor and last y as we'll place our prediction there
plt.show()
plt.pause(0.01)

# Fit neural net
batch_size = 256
mse_train = []
mse_test = []

# Run
epochs = 100
done_test_err = 6e-6
for e in range(epochs):

    X_train_epoch = X_train
    y_train_epoch = y_train
    if True:
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train_epoch = X_train[shuffle_indices]
        y_train_epoch = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):

        # Show progress
        if np.mod(i, 200) == 0:
            # MSE train and test
            mse_train.append(sess.run(mse, feed_dict={X: X_train_epoch, Y: y_train_epoch}))
            mse_test.append(sess.run(mse, feed_dict={X: X_test, Y: y_test}))
            test_err = mse_test[-1]
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', test_err)
            if len(mse_test) >= 2 and test_err == min(mse_test):
                saver.save(sess, model_fname)
                print('model saved')
            # Prediction
            pred = sess.run(out, feed_dict={X: X_test_future})
            line2.set_ydata(inverse_price_transform(scaler, close_column_index, np.concatenate([full_train_plot, pred[0]])))
            plt.title('Epoch %s, batch %s, test error %s' % (e, i, test_err))
            if test_err < done_test_err:
                break
            plt.pause(0.01)
            continue

        start = i * batch_size
        batch_x = X_train_epoch[start:start + batch_size]
        batch_y = y_train_epoch[start:start + batch_size]
        # Run optimizer with batch
        sess.run(opt, feed_dict={X: batch_x, Y: batch_y})

    if test_err < done_test_err:
        break

pred = pred[0][1:-1]
actual = full_plot[-len(pred):]
assert len(actual) == len(pred)
print('avg diff:', sum(abs(pred - actual)) / len(pred))
pred[1:] = actual[:-1]
print('using previous point:', sum(abs(pred - actual)) / len(pred))

print('done')

plt.ioff()
plt.show()
