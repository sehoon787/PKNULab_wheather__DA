import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# check_col = ['isitu-LST','insitu-TG','GK2A-LST','insitu-TED0.05']     # isitu-LST
# colnames = ['Band1','Band2','Band3','Band4','Band5','Band6','Band7','Band8','Band9','Band10','Band11','Band12','Band13','Band14','Band15','Band16','30daysBand3','30daysBand13',
#             ,'SolarZA','SateZA','ESR','Height','LandType','insitu-HM','insitu-TD','insitu-TG','insitu-TED0.05','insitu-TED0.1','insitu-TED0.2','insitu-TED0.3','insitu-PA','insitu-PS', "month"]
colnames = [
            'Band1','Band2','Band3','Band4','Band6','Band7','Band8','Band9',
            'Band10','Band11','Band12','Band13','Band14','Band15','Band16',
            '30daysBand3','30daysBand13',
            'SolarZA','SateZA','ESR','Height','LandType',
            'insitu-HM','insitu-TD', 'month'
            # 'insitu-TED0.05', 'insitu-TED0.1','insitu-TED0.2','insitu-TED0.3'
            ]

def load_data(filename, manual=True):  # 필요에 따라 레이블을 사용할수도 안할수도 있음
    global check_col

    df = pd.read_csv(filename)

    if manual:
        check_col = list(df.keys())

    X = df.loc[:, check_col[1:]].values
    y = df.loc[:, check_col[0]].values - 1  # 레이블 값을 지난번과 다르게 0~8까지로 사용하도록 변경

    return X, y



# X, y = load_data("../과제1-2_결측제거_0729_1/merge/202107_merge.csv")
X, y = load_data("../../과제2 결측제거 extract_col/과제2 결측치 제거.csv")

y.min(), y.max()
model = Sequential()
print(X.shape, y.shape)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
checkpoint = tf.keras.callbacks.ModelCheckpoint("lstm01.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto")

Adam = tf.keras.optimizers.Adam(clipnorm=1.)

model = Sequential()

# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(64, int(len(check_col)-1))))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(RepeatVector(64))

model.add(LSTM(256, input_shape=(int(len(check_col)-1), 1), return_sequences=True)) #input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원 출력
# model.add(LSTM(9, input_shape=(int(len(check_col)-1), 1), return_sequences=True)) #input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원 출력
model.add(Dropout(0.3)) #드랍아웃 층
model.add(LSTM(512, input_shape=(int(len(check_col)-1), 1), return_sequences=True)) #input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원 출력
model.add(Dropout(0.3)) #과적합 방지를 위한 드랍아웃 비율은 0.3
model.add(LSTM(256, return_sequences=True)) #LSTM 층  256차원출력+
model.add(Dropout(0.3)) #드랍아웃 층
model.add(LSTM(128)) #LSTM층 128차원 출력
model.add(Dense(100)) #활성화 함수
model.add(Dense(1, activation='relu')) #활성화 함수

# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mse'])
model.summary()

model.fit(X, y, epochs=200, batch_size=512, verbose=1, callbacks=[early_stopping, checkpoint])
