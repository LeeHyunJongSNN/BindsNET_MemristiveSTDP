import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizer_v2 import adam
from keras_preprocessing.sequence import pad_sequences

wave_data = []
wave_label = []

max_len = 320

fname = " "
for fname in ["C:/Pycharm BindsNET/Wave_classifiers/Wi-Fi_Preambles"
              "/WIFI_10MHz_IQvector_(minus)3dB_20000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
            continue

        linedata = [complex(x) for x in line.split()]
        if len(linedata) == 0:
            continue

        # linedata_real = [x.real for x in linedata[0:len(linedata) - 1]]
        # linedata_imag = [x.imag for x in linedata[0:len(linedata) - 1]]
        # linedata_all = linedata_real + linedata_imag

        linedata_abs = [abs(x) for x in linedata[0:len(linedata) - 1]]

        cl = linedata[-1].real

        wave_label.append(cl)
        # wave_data.append(linedata_all)
        wave_data.append(linedata_abs)

    f.close()

train_data = np.array(wave_data)
train_label = np.array(wave_label)
train_data = pad_sequences(train_data, maxlen=max_len)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(20000, 320)))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(optimizer=adam, lose='mse', metrics=['mae'])
model.fit(train_data, train_label, epochs=100, batch_size=1)

x_input = train_data[random.randint(1, 20000)]
x_input - x_input.reshape((1, 320))

yhat = model.predict(x_input)
print(yhat)