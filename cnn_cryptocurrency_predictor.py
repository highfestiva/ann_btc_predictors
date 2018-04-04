#!/usr/bin/env python3

import functools
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets


def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:
    def __init__(self, image, label, dropout=0.5, filter_num=64):
        self.image = image
        self.label = label
        self.dropout = dropout
        self.filter_num = filter_num
        # define tf options only once; here
        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):
        with tf.variable_scope('model') as scope:
            layers = []

            # conv_1 [batch, ngf, 9] => [batch, 64, ngf]
            with tf.variable_scope('conv_1'):
                output = relu(conv1d(self.image, self.filter_num, name='conv_1'))
                layers.append(output)

            # conv_2 - conv_6
            layer_specs = [
                (self.filter_num * 2, 0.5),    # conv_2: [batch, 64, ngf] => [batch, 32, ngf * 2]
                (self.filter_num * 4, 0.5),    # conv_3: [batch, 32, ngf * 2] => [batch, 16, ngf * 4]
                (self.filter_num * 8, 0.5),    # conv_4: [batch, 16, ngf * 4] => [batch, 8, ngf * 8]
                (self.filter_num * 8, 0.5),    # conv_5: [batch, 8, ngf * 8] => [batch, 4, ngf * 8]
                (self.filter_num * 8, 0.5),    # conv_6: [batch, 4, ngf * 8] => [batch, 2, ngf * 8]
            ]

            # adding layers
            for out_channels,dropout in layer_specs:
                with tf.variable_scope('conv_%d' % (len(layers) + 1)):
                    rectified = lrelu(layers[-1], 0.2)
                    # [batch, in_width, in_channels] => [batch, in_width/2, out_channels]
                    convolved = conv1d(rectified, out_channels)
                    # batchnormalize convolved
                    output = batchnorm(convolved, is_2d=False)
                    # dropout
                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)
                    layers.append(output)

            # fc1
            h_fc1 = relu(fully_connected(layers[-1], 256, name='fc1'))
            # dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)
            # fc2
            result = tf.sigmoid(fully_connected(h_fc1_drop, 2, name='fc2'))
            return result

    @define_scope
    def optimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.prediction))
        return tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



class DataSet(object):
    def __init__(self, images, labels, dtype=dtypes.float32, seed=None):
        assert images.shape[0] == labels.shape[0]
        seed1, seed2 = random_seed.get_seed(seed)
        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.total_batches = images.shape[0]

    def next_batch(self, batch_size, shuffle=True):
        start = self.index_in_epoch
        # first epoch shuffle
        if self.epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.total_batches)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # next epoch
        if start + batch_size <= self.total_batches:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self._images[start:end], self._labels[start:end]
        else: # epoch ending
            self.epochs_completed += 1
            # store what is left of this epoch
            batches_left = self.total_batches - start
            images_left = self._images[start:self.total_batches]
            labels_left = self._labels[start:self.total_batches]
            # shuffle for new epoch
            if shuffle:
                perm = np.arange(self.total_batches)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - batches_left
            end = self.index_in_epoch
            images_new = self._images[start:end]
            labels_new = self._labels[start:end]
            return np.concatenate((images_left, images_new), axis=0), np.concatenate((labels_left, labels_new), axis=0)


def conv1d(input, output_dim,
           conv_w=9, conv_s=2,
           padding='SAME', name='conv1d',
           stddev=0.02, bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [conv_w, input.get_shape().as_list()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        c = tf.nn.conv1d(input, w, conv_s, padding=padding)
        if bias:
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            return c + b
        return c


def batchnorm(input, name='batchnorm', is_2d=False):
    with tf.variable_scope(name):
        input = tf.identity(input)
        channels = input.get_shape()[-1]
        offset = tf.get_variable('offset', [channels],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        if is_2d:
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        else:
            mean, variance = tf.nn.moments(input, axes=[0, 1], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                                                                     variance_epsilon=variance_epsilon)
        return normalized


def fully_connected(input, output_dim, name='fc', stddev=0.02):
    with tf.variable_scope(name):
        unfolded_dim = functools.reduce(lambda x, y: x*y, input.get_shape().as_list()[1:])
        w = tf.get_variable('w',
            [unfolded_dim, output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        input_flat = tf.reshape(input, [-1, unfolded_dim])
        return tf.matmul(input_flat, w) + b


def lrelu(x, a):
    with tf.name_scope('lrelu'):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def relu(x, name='relu'):
    return tf.nn.relu(x)



def xform_seq3d_array(df, moving_window=64, steps_ahead_to_predict=5, train_test_ratio=4.0):
    '''Returns values and labels. The values are 3D[timestep][moving_window][columns]. The labels
       are [1,0] for close price is up the next N timestemp or [0,1] for close price is down
       the next N timestep, where N is steps_ahead_to_predict.'''
    def process_data(df):
        dfs = []
        for i in range(moving_window):
            dfs += [df]
            df = df.shift()
        df_seq = pd.concat(dfs, axis=1)
        df_seq = df_seq.iloc[moving_window:-steps_ahead_to_predict]
        values = df_seq.values.reshape(-1, moving_window, len(df.columns))
        labels = []
        df_labels = df.iloc[moving_window:-steps_ahead_to_predict]
        for i,close_price in zip(df_labels.index, df_labels['close']):
            predict_close_price = df.loc[i+steps_ahead_to_predict, 'close']
            labels.append([1.0,0.0] if predict_close_price > close_price else [0.0,1.0])
        labels = np.array(labels)
        return values, labels

    # transform DataFrame to our liking
    df = df.drop(['time','time_close','ignore'], axis=1)
    values,labels = process_data(df)
    
    # normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    for i in range(len(values)):
        values[i] = scaler.fit_transform(values[i])

    # shuffling the data
    perm = np.arange(labels.shape[0])
    np.random.shuffle(perm)
    values = values[perm]
    labels = labels[perm]

    # selecting 1/5 for testing, and 4/5 for training
    train_test_idx = int((1.0 / (train_test_ratio + 1.0)) * labels.shape[0])
    train_values = values[train_test_idx:,:,:]
    train_labels = labels[train_test_idx:]
    test_values = values[:train_test_idx,:,:]
    test_labels = labels[:train_test_idx]

    train = DataSet(train_values, train_labels)
    test = DataSet(test_values, test_labels)

    return Datasets(train=train, validation=None, test=test)



print('+' + '-'*143 + '+')
print('| Predicting if bitcoin price goes up or down 5 minutes ahead. The accurracy should come up to above 95% after 5 minutes of GPU training or so. |')
print('+' + '-'*143 + '+')
# load data
coin_fname = 'data/bitcoin_usdt_1m.json'
df = pd.read_json(coin_fname, orient='split')
minutes_ahead_to_predict = 5
dataset = xform_seq3d_array(df, steps_ahead_to_predict=minutes_ahead_to_predict)
print('bitcoin 1-minute data loaded')

# setup graph
image = tf.placeholder(tf.float32, [None, 64, 9])
label = tf.placeholder(tf.float32, [None, 2])
dropout = tf.placeholder(tf.float32)
model = Model(image, label, dropout=dropout)

# model saving/restoring
saver = tf.train.Saver()
checkpoint_fname = '.cnn_checkpoints/bitcoin_model.ckpt'

# session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    try:
            saver.restore(sess, checkpoint_fname)
            print('model restored')
    except:
            sess.run(tf.global_variables_initializer())
            print('first run, no previous model to load')
    for i in range(10000):
        images, labels = dataset.train.next_batch(400)
        if i % 10 == 0:
            images_eval, labels_eval = dataset.test.next_batch(1000)
            accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})
            print('\rstep %d, accuracy %.1f%%   ' % (i, accuracy*100), end='')
        sess.run(model.optimize, {image: images, label: labels, dropout: 0.5})

        if (i > 0) and (i % 500 == 0):
            saver.save(sess, checkpoint_fname)
            print('\nmodel saved')

    images_eval, labels_eval = dataset.test.next_batch(1000)
    accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})
    print('final accuracy on testing set: %.1f%%' % (accuracy*100))
