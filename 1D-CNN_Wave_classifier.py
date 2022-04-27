import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from keras.optimizer_v2 import adam
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

max_len = 320
embedding_dim = 320
dropout_ratio = 0.3
num_filters = 160
kernel_size_1 = 4
kernel_size_2 = 8
kernel_size_3 = 16
hidden_units_1 = 32
hidden_units_2 = 16

test_valid_ratio = 0.3
test_ratio = 0.5

train_data = []
test_data = []

wave_data = []
wave_label = []

fname = " "
for fname in ["WIFI_10MHz_IQvector_(minus)3dB_60000.txt"]:
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

train_data, test_valid_data, train_label, test_valid_label = train_test_split(
    wave_data, wave_label, test_size=test_valid_ratio)
valid_data, test_data, valid_label, test_label = train_test_split(
    test_valid_data, test_valid_label, test_size=test_ratio)

train_data = pad_sequences(train_data, maxlen=max_len)
valid_data = pad_sequences(valid_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)
train_label = np.array(train_label)
valid_label = np.array(valid_label)
test_label = np.array(test_label)

print('train size :', train_data.shape)
print('valid size :', valid_data.shape)
print('test size :', test_data.shape)

adam = adam.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model = Sequential()
model.add(Embedding(1, embedding_dim))
model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters, kernel_size_1, padding='valid', activation='relu'))
model.add(Conv1D(num_filters, kernel_size_2, padding='valid', activation='relu'))
model.add(Conv1D(num_filters, kernel_size_3, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(dropout_ratio))
model.add(Dense(hidden_units_1, activation='relu'))
model.add(Dense(hidden_units_2, activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_data, train_label, validation_data=(valid_data, valid_label), epochs=400,
                    batch_size=64, use_multiprocessing=True, callbacks=[es, mc])

print("\n test accuracy: %.4f" % (model.evaluate(test_data, test_label, batch_size=64)[1]))

prediction = model.predict(test_data)
bin_prediction = tf.round(prediction)
print(classification_report(test_label, bin_prediction))
cm = confusion_matrix(test_label, bin_prediction)
print(cm)
print("Probability of Detection: %.4f" % (cm[0][0] / (cm[0][0] + cm[1][0])))
print("False Negative Probability: %.4f" % (cm[1][0] / (cm[0][0] + cm[1][0])))
print("False Positive Probability: %.4f" % (cm[0][1] / (cm[0][1] + cm[1][1])))
