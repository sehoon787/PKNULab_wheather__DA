from __init__ import colnames

import pandas as pd
import tensorflow as tf
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten

def load_data(filename, train_type="lst") :
    df = pd.read_csv(filename)[colnames]

    #colnames 0 : lst,  1 : ta,  2 ~ : train column
    if train_type == "lst" :
        Y = df.loc[:, colnames[0]].values
    elif train_type == "ta" :
        Y = df.loc[:, colnames[1]].values
    else :
        return False

    X = df.loc[:, colnames[2:]].values

    return X, Y

def my_train(train_data="../train/train_data.csv",
               train_type="lst",
               model_path="../model",
               model_name="train_model",
               epoch=100,
               batch_size=1024) :
    '''
    :param train_data: load validation file path (defalut : "../train/train_data.csv")
    :param train_type: lst / ta (defalut : "lst")
    :param model_path: train model path (defalut : "../model")
    :param model_name: train model name (defalut : "train_model")
    :param epoch: train epoch (defalut : 100)
    :param batch_size: train batch size (defalut : 1024)
    '''

    X, Y = load_data(train_data, train_type)

    print("X shape : ", X.shape)
    print("Y shape : ", Y.shape)

    my_model = model_path + "/" + model_name + ".h5"


    Y.min(), Y.max()

    model = Sequential()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(my_model, monitor="val_loss", verbose=1 , save_best_only=True, mode="auto")

    Adam = tf.keras.optimizers.Adam(clipnorm=1.)

    model = Sequential()

    model.add(LSTM(512, input_shape=(X.shape[1], 1), return_sequences=True)) #input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원
    model.add(Dropout(0.5)) #과적합 방지를 위한 드랍아웃 비율은 0.5
    model.add(LSTM(256, return_sequences=True)) #LSTM 층  256차원출력
    model.add(Dropout(0.5)) #과적합 방지를 위한 드랍아웃 비율은 0.5
    model.add(LSTM(128, return_sequences=True)) #LSTM 층  256차원출력+
    model.add(Dropout(0.5)) #드랍아웃 층
    model.add(LSTM(64, return_sequences=True)) #LSTM 층  256차원출력+
    model.add(Dropout(0.5)) #드랍아웃 층

    model.add(LSTM(32)) #LSTM층 128차원 출력
    model.add(Dropout(0.5)) #과적합 방지를 위한 드랍아웃 비율은 0.5

    model.add(Dense(100)) #활성화 함수
    model.add(Flatten())
    model.add(Dense(1, activation='relu')) #활성화 함수

    model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mse'])
    model.summary()

    model.fit(X, Y, epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[early_stopping, checkpoint])

    # TA model
    # y.min(), y.max()
    #
    # model = Sequential()
    #
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint("lstm01.h5", monitor="val_loss", verbose=1, save_best_only=True,
    #                                                 mode="auto")
    #
    # Adam = tf.keras.optimizers.Adam(clipnorm=1.)
    #
    # model = Sequential()
    # model.add(LSTM(512, input_shape=(X.shape[1], 1), return_sequences=True))
    # input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원 출력
    # model.add(Dropout(0.5))  # 과적합 방지를 위한 드랍아웃 비율은 0.5
    # model.add(LSTM(256, return_sequences=True))  # LSTM 층  256차원출력+
    # model.add(Dropout(0.5))  # 드랍아웃 층
    # model.add(LSTM(128, return_sequences=True))  # LSTM 층  256차원출력+
    # model.add(Dropout(0.5))  # 드랍아웃 층
    # model.add(LSTM(64, return_sequences=True))  # LSTM층 128차원 출력
    # model.add(Dropout(0.5))  # 과적합 방지를 위한 드랍아웃 비율은 0.5
    # # model.add(LSTM(32, return_sequences=True)) #LSTM층 128차원 출력
    # # model.add(Dropout(0.5)) #과적합 방지를 위한 드랍아웃 비율은 0.5
    # model.add(Dense(100))  # 활성화 함수
    # model.add(Flatten())
    # model.add(Dense(1, activation='relu'))  # 활성화 함수
    #
    # # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    # model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mse'])
    # model.summary()
    #
    # model.fit(X, y, epochs=50, batch_size=4096, verbose=1, callbacks=[early_stopping, checkpoint])

    model.save(my_model)

    return my_model