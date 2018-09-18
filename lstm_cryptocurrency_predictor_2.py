#!/usr/bin/env python3

# Shamelessly stolen from https://pythonprogramming.net/balancing-rnn-data-deep-learning-python-tensorflow-keras/?completed=/normalizing-sequences-deep-learning-python-tensorflow-keras/
# and modified to improve the accuracy a little bit (albeit with more training required)

import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
import indicators


SEQ_LEN = 120  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "neo_usdt"
EPOCHS = 60  # how many passes through our data
BATCH_SIZE = 1024  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def preprocess_df(df):
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            if 'volume' in col:
                # kill empty volume
                low = 0
                e = 1e-4
                while low == 0:
                    low = df[col].quantile(e)
                    e *= 2
                df.loc[(df[col]<low),col] = low
            # normalize, but don't scale
            df[col] = np.log(df[col])
            df[col] = df[col].diff()

    df.dropna(inplace=True)

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        row = [n for n in i[:-1]]  # store all but the target
        prev_days.append(row)
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            target = i[-1]
            a = np.array(prev_days)
            # scale per window (not per whole feature)
            a = preprocessing.scale(a, axis=0) # scale allows outliers to continue being just that
            if target == 0:
                sells.append([a, target])
            else:
                buys.append([a, target])

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


main_df = pd.DataFrame() # begin empty

ratios = ["neo_usdt", "neo_btc", "neo_eth", "ethereum_usdt", "bitcoin_usdt"]  # the ratios we want to consider
for ratio in ratios:  # begin iteration
    print(ratio)
    dataset = f'data/{ratio}_5m.csv'  # get the full path to the file.
    df = pd.read_csv(dataset)  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    rename_cols = [(c, f'{ratio}_'+c) for c in 'open close hi lo volume'.split()]
    df.rename(columns={c:v for c,v in rename_cols}, inplace=True)

    df = df[['time']+[v for c,v in rename_cols]]  # pick only our columns
    if ratio == RATIO_TO_PREDICT:
        df['srsi'] = indicators.calc_stoch_rsi(df[f'{ratio}_close']) + 0.5 # just don't let close to zero for ln
        df['crsi'] = indicators.calc_connors_rsi(df[f'{ratio}_close']) + 0.5 # just don't let close to zero for ln
        df['hv'] = indicators.calc_historical_volatility(df[f'{ratio}_close']) + 0.5 # just don't let close to zero for ln
    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)

###########################################
# mdf = main_df.reset_index()
# import finplot as fplt
# fplt.plot(x=mdf['time']/1e3, y=mdf['srsi'])
# fplt.plot(x=mdf['time']/1e3, y=mdf['crsi'])
# fplt.plot(x=mdf['time']/1e3, y=mdf['hv'])
# fplt.show()
###########################################

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)

## here, split away some slice of the future data from the main main_df.
times = main_df.index.values
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"predicting steps ahead: {FUTURE_PERIOD_PREDICT}")
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"validation don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir=".logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint(".lstm_checkpoints/{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save(".lstm_checkpoints/{}".format(NAME))
