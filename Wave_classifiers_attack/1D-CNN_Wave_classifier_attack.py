import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from keras.optimizers.optimizer_v2 import adam
from keras_preprocessing.sequence import pad_sequences
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
wave_classes = []

attack_type = "Gaussian"
attack_stddev = 1
attack_mean = 0
attack_intensity = 1
noise_intensity = 1

abs_traindata = []
abs_validdata = []
attacked_testdata = []

fname = " "
for fname in ["D:/SNN_dataset/Wi-Fi_Preambles/"
              "WIFI_10MHz_IQvector_18dB_20000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
                continue

        linedata = [complex(x) for x in line.split()]
        if len(linedata) == 0:
                continue

        cl = int(linedata[-1].real)
        wave_classes.append(cl)
        wave_data.append(linedata)

    f.close()

train_data, test_valid_data, train_label, test_valid_label = train_test_split(
    wave_data, wave_classes, test_size=test_valid_ratio)

valid_data, test_data, valid_label, test_label = train_test_split(
    test_valid_data, test_valid_label, test_size=test_ratio)

for i in range(len(train_data)):

    linedata_labelremoved = [x for x in train_data[i][0:len(train_data[i]) - 1]]
    linedata_abs = [abs(x) for x in linedata_labelremoved[0:len(linedata_labelremoved)]]

    abs_traindata.append(linedata_abs)

for j in range(len(valid_data)):
    linedata_labelremoved = [x for x in valid_data[j][0:len(valid_data[j]) - 1]]
    linedata_abs = [abs(x) for x in linedata_labelremoved[0:len(linedata_labelremoved)]]

    abs_validdata.append(linedata_abs)

if attack_type == "Gaussian":
    for k in range(len(test_data)):
        linedata_labelremoved = [x for x in test_data[k][0:len(test_data[k]) - 1]]
        attack = (attack_stddev * np.random.randn(len(linedata_labelremoved)) + attack_mean) * attack_intensity
        linedata_attacked = (np.array(linedata_labelremoved) + attack).tolist()

        linedata_abs = [abs(x) for x in linedata_attacked[0:len(linedata_attacked)]]

        attacked_testdata.append(linedata_abs)

elif attack_type == "Wave":
    for k in range(len(test_data)):
        linedata_labelremoved = [x for x in test_data[k][0:len(test_data[k]) - 1]]
        attack_noise = (attack_stddev * np.random.randn(len(linedata_labelremoved)) + attack_mean) * noise_intensity
        attack = 32 * [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        attack = (np.array(attack) + attack_noise) * attack_intensity
        linedata_attacked = (np.array(linedata_labelremoved) + attack).tolist()

        linedata_abs = [abs(x) for x in linedata_attacked[0:len(linedata_attacked)]]

        attacked_testdata.append(linedata_abs)

abs_traindata = pad_sequences(abs_traindata, maxlen=max_len)
abs_validdata = pad_sequences(abs_validdata, maxlen=max_len)
attacked_testdata = pad_sequences(attacked_testdata, maxlen=max_len)
train_label = np.array(train_label)
valid_label = np.array(valid_label)
test_label = np.array(test_label)

print('train size :', abs_traindata.shape)
print('valid size :', abs_validdata.shape)
print('test size :', attacked_testdata.shape)

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
history = model.fit(abs_traindata, train_label, validation_data=(abs_validdata, valid_label), epochs=400,
                    batch_size=64, use_multiprocessing=True, callbacks=[es, mc])

print("\n test accuracy: %.4f" % (model.evaluate(attacked_testdata, test_label, batch_size=64)[1]))

prediction = model.predict(attacked_testdata)
bin_prediction = tf.round(prediction)
print(classification_report(test_label, bin_prediction))
cm = confusion_matrix(test_label, bin_prediction)
print(cm)
print("Probability of Detection: %.4f" % (cm[0][0] / (cm[0][0] + cm[1][0])))
print("False Negative Probability: %.4f" % (cm[1][0] / (cm[0][0] + cm[1][0])))
print("False Positive Probability: %.4f" % (cm[0][1] / (cm[0][1] + cm[1][1])))
