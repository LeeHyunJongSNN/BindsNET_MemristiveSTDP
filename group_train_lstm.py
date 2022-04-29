import os
import sys
import random
import re

import collections

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

import numpy as np
from numpy import concatenate

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

#import tensorflow_federated as tff

# required for this machine
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def generate_series(data, maf, scaled_features, bit_features, forecast_features, ground_truth_features, n_in = 1, n_out = 1, frame_interval = 10, skip_frame = 999999999 ):
    df = DataFrame(data)

    full_series_length = df.shape[0]
    prev = -1
    error_positions = [ ] 
    split = None
    for i in range(0, full_series_length):
        curr_row = df.iloc[i]
        curr = curr_row["FrameID"]
        if prev >= 0 and curr - prev > frame_interval and abs(curr - prev) < skip_frame:
            error_positions.append( (i-1, i) )
        elif prev >= 0 and abs(curr - prev) >= skip_frame:
            print("Should split!!!!!", curr, prev, i-1, i, full_series_length)
            #print(error_positions)
            split = df.iloc[i:full_series_length]
            df = df.iloc[0:i]
            #print(df["FrameID"].values)
            #print(split["FrameID"].values)
            break
        prev = curr

    if len(error_positions) > 0:
        frameList = []
        positioner = 0
        for error_start_idx, error_end_idx in error_positions:
            #print("interpolation between idx", error_start_idx, error_end_idx)

            point_before_error = df.iloc[error_start_idx]
            point_after_error = df.iloc[error_end_idx]
            error_length = int(point_after_error["FrameID"] - point_before_error["FrameID"]) - 1
            #print("interpolation between", point_before_error["FrameID"], point_after_error["FrameID"], error_start_idx, error_end_idx)

            for cnt in range(1, error_length+1):
                added_row = np.zeros( (1, len(df.columns)) )
                for idx, name in enumerate(df.columns):
                    if name == "FrameID":
                        added_row[0, idx] = int(point_before_error["FrameID"]) + cnt * frame_interval
                    elif name in ground_truth_features:
                         added_row[0, idx] = -99999 # interpolated entry does not have ground truth
                    elif name not in scaled_features:
                        added_row[0, idx] = point_before_error[name]
                    else:
                        added_row[0, idx] = point_before_error[name] + (point_after_error[name] - point_before_error[name]) / float(error_length) * cnt
                if cnt == 1:
                    addedAll = added_row
                else:
                    addedAll = np.vstack( (addedAll , added_row) ) 

            addedFrame = DataFrame(addedAll)
            addedFrame.columns = df.columns
        
            bf = df.iloc[positioner:error_start_idx+1]
            af = df.iloc[error_end_idx].to_frame().transpose()

            frameList.append( bf )
            frameList.append( addedFrame )
            frameList.append( af )
            positioner = error_end_idx + 1

        if positioner < full_series_length:
            remains = df.iloc[positioner:full_series_length]
            frameList.append(remains)
        
        df = pd.concat(frameList)

    maf_col = None
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    features = scaled_features + bit_features
    no_maf_features = [ "Velocity", "Acceleration", "Vel_X", "Vel_Y", "Acc_X", "Acc_Y" ]
    for i in range(n_in, 0, -1):
        col = df.shift(i)[features]
        #print("maf before", maf_col)
        if maf_col is None:
            maf_col = col
        else:
            maf_col = concat( [ (1.0 - maf) * maf_col[forecast_features] + maf * col[forecast_features], col[ no_maf_features + bit_features] ], axis=1 )
        cols.append(maf_col)
        #print("orig", col)
        #print("maf", maf_col)
        names += [('%s(t-%d)' % (nt, i)) for nt in features]
    # forecast sequence (t, t+1, ... t+n)
    for i in [ n_out ]: # range(0, n_out):
        col = df.shift(-i)[forecast_features + ground_truth_features]
        cols.append(col)
        if i == 0:
            names += [('%s(t)' % nt ) for nt in forecast_features + ground_truth_features]
        else:
            names += [('%s(t+%d)' % (nt, i)) for nt in forecast_features + ground_truth_features]

    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    if split is not None:
        print("Processing split")
        split_series = generate_series(split, maf, scaled_features, bit_features, forecast_features, ground_truth_features, n_in = n_in, n_out = n_out, skip_frame = skip_frame)
        agg = pd.concat( [ agg, split_series ] ) 
        print("split processed and merged")

    return agg
    
def generate_series_per_vehicle(data, maf, scaled_features, bit_features, forecast_features, ground_truth_features, vehicle_indices, n_in = 1, n_out = 1, frame_interval = 10, skip_frame = 9999999 ):
    series = None
    for idx in vehicle_indices:
        cond = (data.UniqueSrcID == idx)
        df = data[cond]
        series_part = generate_series(df, maf, scaled_features, bit_features, forecast_features, ground_truth_features, n_in = n_in, n_out = n_out, frame_interval = frame_interval, skip_frame = skip_frame)
        if series is None:
            series = series_part
        else:
            series = series.append(series_part)
        
    return series 

def build_model(seq_len, n_features, n_output):
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_len, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(n_output, activation = 'sigmoid'))
    return model

def find_best_model(expname, site, expname_par):
    # need to revise: 2022-2-10
    #filepath="./model_%s/%s-%s-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5" % (EXPNAME, SITE, EXPNAME_PAR)
    filepath = "./model_%s/" % (expname)
    filelist = [ os.path.join(filepath, f) for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f)) ]
    last = -1
    for filename in filelist:
        temp = filename.split("-")
        temp = re.split('[-/]', filename)
        if len(temp) >= 6:
            temp_site = temp[2]
            temp_par = temp[3]
            if temp_site == site and temp_par == expname_par:
                idx = int(temp[6])
                if idx > last:
                    last = idx
                    selected_filename = filename
    print("Best model to be loaded", selected_filename)
    return selected_filename


mmax = np.array([ 220.560000, 1659.480000, 150, 150, 150, 150, 150, 150 ])
mmin = np.array([ 0.000000, 0.000000, -150, -150, -150, -150, -150, -150])

def printErrors(X, Y, Y_hat, bGroundTruth = False, bPrintTraj = False):
    inv_pred = DataFrame(Y_hat).mul(mmax[0:2]-mmin[0:2]).add(mmin[0:2])
    if not bGroundTruth:
        inv_truth = DataFrame(Y).mul(mmax[0:2]-mmin[0:2]).add(mmin[0:2])
    else:
        inv_truth = DataFrame(Y)

    filtering_idx = np.where(inv_truth[0].values != -99999)

    inv_pred = inv_pred.values[filtering_idx]
    inv_truth = inv_truth.values[filtering_idx]
    filtered_X = X[filtering_idx]

    diff = inv_truth - inv_pred
    abs_diff = np.abs(diff)
    squared = diff * diff
    squared_dist = squared[:, 0] + squared[:, 1]
    dist = np.sqrt(squared_dist)

    print("=====", np.min(abs_diff[:, 0]), np.quantile(abs_diff[:, 0], 0.25), np.median(abs_diff[:, 0]), np.mean(abs_diff[:, 0]), np.quantile(abs_diff[:, 0], 0.75), np.max(abs_diff[:, 0]) )
    print("=====", np.min(abs_diff[:, 1]), np.quantile(abs_diff[:, 1], 0.25), np.median(abs_diff[:, 1]), np.mean(abs_diff[:, 1]), np.quantile(abs_diff[:, 1], 0.75), np.max(abs_diff[:, 1]) )
    print("=====", np.min(dist), np.quantile(dist, 0.25), np.median(dist), np.mean(dist), np.quantile(dist, 0.75), np.max(dist) )
    
    if len(dist) < 10:
        tpq = 0
    else:
        tpq = 1.0 - 10.0 / len(dist)

    result = np.where(dist >  np.quantile(dist, tpq ))
    print("Truth")
    for row in inv_truth[result]:
        print(row)
    print("Pred")
    for row in inv_pred[result]:
        print(row)
    print("--------")
    if bPrintTraj:
        for row in filtered_X[result]:
            for entry in row:
                temp_item = entry[0:len(mmax)] * (mmax - mmin) + mmin
                for inner in temp_item:
                    print("%10f " % inner, end="")
                for inner in entry[len(mmax):]:
                    print("%2d " % inner, end="")
                print("")
            print("--------")
if __name__ == "__main__":
    training = True

    BI = int(sys.argv[1])
    EXPNAME = sys.argv[2]
    EXPNAME_PAR = sys.argv[3]
    SITE = sys.argv[4]
    MAF = int(sys.argv[5]) / 100.0
    frame_interval = int(BI / 100.0)
    SEQ_LEN = int(10 * 1000.0 / BI)
    N_OUT = int(1 * 1000.0 / BI)
    N_EPOCH = 50
    N_CHOICE = 30

    print("# configuration - BI:", BI, ", Frame Interval:", frame_interval, ", Observing", SEQ_LEN, "MAF",1.0-MAF, MAF, "for", N_OUT, ", Sampling", N_CHOICE)

    dataset = read_csv("./qualnet/%s/%s/output_%s_0_300_0.txt" % (EXPNAME, EXPNAME_PAR, SITE), header = 0, sep = ' ')
 
    #features = [ "Pos_X", "Pos_Y", "Velocity", "Acceleration", "Direction", "neighborMatrix" ] #, "Preceeding", "Following" ] 
    scaled_features = [ "Pos_X", "Pos_Y", "Velocity", "Acceleration", "Vel_X", "Vel_Y", "Acc_X", "Acc_Y" ]
    forecast_features = [ "Pos_X", "Pos_Y" ]
    ground_truth_features = [ "Pos_X_GT", "Pos_Y_GT" ]
    notScaledFeatures = [ "neighborMatrix" ]

    #features = scaledFeatures + notScaledFeatures

    bit_features =  [ "b8", "b7", "b6", "b5", "b4", "b3", "b2", "b1" ]
    final_features = scaled_features + bit_features
   
    actual_targets = [ "UniqueVehID", "UniqueSrcID", "FrameID" ] + ground_truth_features + final_features

    n_features = len(final_features)
    n_forecast_features = len(forecast_features)
    n_ground_truth_features = len(ground_truth_features)

    vehicles = np.unique(dataset["UniqueVehID"].to_numpy())

    print("Total", len(vehicles), "vehicles")
   
    choice = np.random.choice(len(vehicles), min(N_CHOICE,len(vehicles)), replace=False)

    all_train_X = None
    all_train_Y = None
    all_train_gt_Y = None

    test_X_map = {}
    test_gt_Y_map = {}

    def convert_dataframe(df):
        #print(np.unique(data["neighborMatrix"].to_numpy()))
        temp = df["neighborMatrix"].apply(int, base=16)

        b8 = temp.apply(lambda x: (x >> 7) & 0x01)
        b7 = temp.apply(lambda x: (x >> 6) & 0x01)
        b6 = temp.apply(lambda x: (x >> 5) & 0x01)
        b5 = temp.apply(lambda x: (x >> 4) & 0x01)
        b4 = temp.apply(lambda x: (x >> 3) & 0x01)
        b3 = temp.apply(lambda x: (x >> 2) & 0x01)
        b2 = temp.apply(lambda x: (x >> 1) & 0x01)
        b1 = temp.apply(lambda x: x & 0x01)

        scaled = df[scaled_features].sub(mmin).mul(1.0 / (mmax-mmin))
     
        cols = [ df[ [ "UniqueVehID", "UniqueSrcID", "FrameID", "Pos_X_GT", "Pos_Y_GT" ] ], scaled , b8, b7, b6, b5, b4, b3, b2, b1 ]
        scaled_data = concat(cols, axis=1)
        scaled_data.columns = actual_targets
        return scaled_data

    scaled_dataset = convert_dataframe(dataset)
    print("Normalization Done")

    for vehID in vehicles[choice]:
        print("Processing Vehicle", vehID)
        cond = (scaled_dataset.UniqueVehID == vehID)
        dataPerVeh = scaled_dataset[cond]
        vehicle_indices = np.unique(dataPerVeh["UniqueSrcID"].to_numpy())

        idx_train, idx_test, y_train, y_test = train_test_split(vehicle_indices, vehicle_indices, test_size=0.15)#, random_state = 0)

        train_series = generate_series_per_vehicle(dataPerVeh, MAF, scaled_features, bit_features, forecast_features, ground_truth_features, idx_train, n_in=SEQ_LEN, n_out=N_OUT, frame_interval = frame_interval, skip_frame = 600)
        print("Training Series generation done from", len(idx_train))

        train_X = train_series.values[:, :-(n_forecast_features+n_ground_truth_features)]
        train_X = train_X.reshape((train_X.shape[0], SEQ_LEN, n_features))
        train_Y = train_series.values[:, -(n_forecast_features+n_ground_truth_features):-n_ground_truth_features]
        train_gt_Y = train_series.values[:, -n_ground_truth_features:]

        for testVehID in idx_test:
            test_series = generate_series_per_vehicle(dataPerVeh, MAF, scaled_features, bit_features, forecast_features, ground_truth_features, [ testVehID ], n_in=SEQ_LEN, n_out=N_OUT, frame_interval = frame_interval, skip_frame = 600)

            test_X = test_series.values[:, :-(n_forecast_features+n_ground_truth_features)]
            test_X = test_X.reshape((test_X.shape[0], SEQ_LEN, n_features))
            test_gt_Y = test_series.values[:, -n_ground_truth_features:]
            if test_X.shape[0] > 0 and testVehID not in test_X_map:
                test_X_map[testVehID] = test_X
                test_gt_Y_map[testVehID] = test_gt_Y

        print("Testing Series generation done from", len(idx_test))

        if all_train_X is None:
            all_train_X = train_X
            all_train_Y = train_Y
            all_train_gt_Y = train_gt_Y
        else:
            all_train_X = np.concatenate((all_train_X, train_X))
            all_train_Y = np.concatenate((all_train_Y, train_Y))
            all_train_gt_Y = np.concatenate((all_train_gt_Y, train_gt_Y))

    #print(all_train_X.shape, all_test_X.shape)

    model = build_model( SEQ_LEN, n_features, n_forecast_features )
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')
    filepath="./model_%s/%s-%s-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5" % (EXPNAME, SITE, EXPNAME_PAR)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = model.fit(all_train_X, all_train_Y, epochs=N_EPOCH, batch_size=50, verbose=2, callbacks = callbacks_list, shuffle=False)

    model = build_model( SEQ_LEN, n_features, n_forecast_features )
    model.load_weights(find_best_model(EXPNAME, SITE, EXPNAME_PAR))
    model.compile(loss='mean_squared_error', optimizer='adam')

    pred = model.predict(all_train_X)
    print("===== Training Result")
    printErrors(all_train_X, all_train_gt_Y, pred, bGroundTruth = True)

    all_test_X = None
    all_test_gt_Y = None
    for testVehID in test_X_map:
        test_X = test_X_map[testVehID]
        test_gt_Y = test_gt_Y_map[testVehID]
        pred = model.predict(test_X)
        print("===== Testing Result for", testVehID)
        printErrors(test_X, test_gt_Y, pred, bGroundTruth = True, bPrintTraj = True)
        if all_test_X is None:
            all_test_X = test_X
            all_test_gt_Y = test_gt_Y
        else:
            all_test_X = np.concatenate((all_test_X, test_X))
            all_test_gt_Y = np.concatenate((all_test_gt_Y, test_gt_Y))

    pred = model.predict(all_test_X)
    print("===== Testing Result")
    printErrors(all_test_X, all_test_gt_Y, pred, bGroundTruth = True, bPrintTraj = True)
