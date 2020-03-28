# Imports

from datetime import datetime
import math
import pandas as pd
from Timsort_lib import TimSort
import random
import time
import psutil
from cpuinfo import get_cpu_info
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import datetime as dt
import copy as cp
from math import sqrt
import os
import json

import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras import models
from keras import layers
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder





# Load data

all_res = {}

def get_data():
    df = pd.DataFrame({'Quantity': [], 'Best_Minrun': [], 'Best_Time': []})
    pd.options.mode.chained_assignment = None
    directory = os.path.join("/home/leha/Desktop/projects/all_timsort_data") # my path :))
    # Maybe i can write better, smth like str(os.path.abspath(__file__)[:-len(name_file)]) + '/all_timsort_data/'
    train_pairs = []
    global all_res
    z = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                db = pd.read_csv("D:/infa_Lesha/all_timsort_data/" + file)
                z_m = 0
                all_sizes = []
                for i in range(len(db['Best_minrun'])):
                    df['Best_Minrun'][i - z_m] = cp.deepcopy(db['Best_minrun'][i].replace('[', ''))
                    df['Best_Minrun'][i - z_m] = cp.deepcopy(df['Best_Minrun'][i - z_m].replace(']', ''))
                    df['Best_Minrun'][i - z_m] = cp.deepcopy(df['Best_Minrun'][i - z_m].replace(',', ''))
                    df['Best_Minrun'][i - z_m] = cp.deepcopy(df['Best_Minrun'][i - z_m]).split()
                for i in range(len(df['Best_Minrun'])):
                    for j in range(len(df['Best_Minrun'][i])):
                        df['Best_Minrun'][i][j] = int(df['Best_Minrun'][i][j])
                
                z_m = 0
                all_sizes = []
                for i in range(len(db['Quantity'])):
                    df['Quantity'][i - z_m] = int(db['Quantity'][i])
                
                z_m = 0
                all_sizes = []
                for i in range(len(db['Best_time'])):
                    df['Best_Time'][i - z_m] = cp.deepcopy(db['Best_time'][i].replace('[', ''))
                    df['Best_Time'][i - z_m] = cp.deepcopy(df['Best_Time'][i - z_m].replace(']', ''))
                    df['Best_Time'][i - z_m] = cp.deepcopy(df['Best_Time'][i - z_m].replace(',', ''))
                    df['Best_Time'][i - z_m] = cp.deepcopy(df['Best_Time'][i - z_m].split())
                
                for i in range(len(df['Quantity'])):
                    train_pairs.append([float(df['Quantity'][i]), float(df['Best_Minrun'][i][0])])
                    z += 1
                for i in range(len(df['Quantity'])):
                    all_res[df['Quantity'][i]] = {'minruns': df['Best_Minrun'][i], 'times':df['Best_Time'][i]} # for graphics

    return train_pairs

pairs = get_data()
pairs.sort()

data, data_labels = np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])

train = data[:50000]
train_labels = data_labels[:50000]

test = data[50000:]
test_labels = data_labels[50000:]

mean = train.mean(axis=0)
train -= mean
std = train.std(axis=0) # std - стандартное отклонение = sqrt(mean(abs(x - x.mean())**2)) mean - среднее арифметическое
train /= std

test -= mean
test /= std
data_p = {'mean': mean, 'std': std}
with open('data.json', 'w') as f_write:
    json.dump(data_p, f_write)





# Train and create model

def build_model():
    global train
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=1))
    model.add(layers.Dense(64))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.003), activation='sigmoid'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(96))
    model.add(layers.Dense(96, kernel_regularizer=regularizers.l2(0.003),  activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit(train, train_labels,
    epochs=85, batch_size=52, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test, test_labels)
print(test_mae_score)
model.save('network_big_data.h5')





# Compare model with standart Python sort

s_os = ''
s_os += get_cpu_info()['brand'] + '\n'

s_os += "="*40 + " System Information " +  "="*40 + '\n'
uname = platform.uname()
s_os += f"System: {uname.system}" + '\n'
s_os += f"Node Name: {uname.node}" + '\n'
s_os += f"Release: {uname.release}" + '\n'
s_os += f"Version: {uname.version}" + '\n'
s_os += f"Machine: {uname.machine}" + '\n'
s_os += f"Processor: {uname.processor}" + '\n'

s_os += "="*40 + " CPU Info " + "="*40 + '\n'
# number of cores
s_os += "Physical cores: " + str(psutil.cpu_count(logical=False)) + '\n'
s_os += "Total cores: " + str(psutil.cpu_count(logical=True)) + '\n'
# CPU frequencies
cpufreq = psutil.cpu_freq()
s_os += f"Max Frequency: {cpufreq.max:.2f}Mhz" + '\n'
s_os += f"Min Frequency: {cpufreq.min:.2f}Mhz" + '\n'
s_os += f"Current Frequency: {cpufreq.current:.2f}Mhz" + '\n'
# CPU usage
s_os += "CPU Usage Per Core:" + '\n'
for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
    s_os += f"Core {i}: {percentage}%" + '\n'
s_os += f"Total CPU Usage: {psutil.cpu_percent()}%" + '\n'

with open('data.json') as json_file:
    data = js.load(json_file)
mean, std = data.values()
# model = models.load_model('network.h5') # load model

plus = 100
max_p = 100000000
test2 = np.asarray([i + 0.0 for i in range(100, max_p, plus)])
test2 -= mean
test2 /= std
predictions = model.predict(test2)
z = 0

sr_sum = 1
k = 1
n_tr = 0

kol_blue = []
sizes_blue = []
sizes_red = []
times_blue = []
times_red = []
sizes_green = []
times_green = []

max_sec = 111600
time_f = time.time()

tr = True
i = 0
while i + plus < max_p and tr:
    try:
        if i % 10000 == 0:
            print(i, datetime.now())

        minrun = max(int(predictions[int(i / plus)]), 1)

        size = int(test2[int(i / plus)] * std + mean)
        arr1 = [random.randint(1, 10000) for j in range(size)]
        arr2 = cp.deepcopy(arr1)
        arr3 = cp.deepcopy(arr1)
        if time.time() - time_f >= max_sec:
            tr = False
        
        now_s = time.time()

        kol_blue.append(TimSort(arr1, minrun))

        now2_s = time.time()

        time_m = abs(now2_s - now_s)

        now_sg = time.time()

        arr2.sort()

        now2_sg = time.time()

        time_g = abs(now2_sg - now_sg)

        time_m *= 1000
        time_g *= 1000
        sizes_blue.append(size + int(z))
        sizes_green.append(size + int(z))
        times_blue.append(time_m)
        times_green.append(time_g)
        z *= 2

        i += plus
    except:
        tr = False

print(tr, i)

plt.scatter(sizes_blue, times_blue, c=[[0, 0, 1]], s=20)
plt.scatter(sizes_green, times_green, c=[[0, 1, 0]], s=20)

plt.xlabel('Size')
plt.ylabel('Time (* 1000)')
plt.gcf().set_size_inches((34, 13))
plt.grid(True)
plt.savefig('compr.png')
f = open('graph_bg.txt', 'w')
f.write(s_os)
f.close()
submission = pd.DataFrame({'Sizes_blue': sizes_blue, 'Times_blue': times_blue, 'Kol_blue': kol_blue, 'Sizes_red': sizes_green, 'Times_red': times_green})
submission.to_csv('graph_bg.csv', index=False)
plt.show()





# Test model using K-fold-cross validation

def build_model_test():
    global train
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=1))
    model.add(layers.Dense(64))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.003), activation='sigmoid'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(96))
    model.add(layers.Dense(96, kernel_regularizer=regularizers.l2(0.003),  activation='relu'))
    model.add(layers.Dense(56))
    model.add(layers.Dense(56, kernel_regularizer=regularizers.l2(0.001),  activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 3
epochs = 150
all_scores = []
all_mae_histories = []
num_samples = len(train) // k
for i in range(k):
    print('processing fold: ', i)
    val_data = train[i * num_samples: (i + 1) * num_samples]
    val_targets = train_labels[i * num_samples: (i + 1) * num_samples]
    
    partial_train_data = np.concatenate([train[:i * num_samples], train[(i + 1) * num_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_labels[:i * num_samples], train_labels[(i + 1) * num_samples:]], axis=0)
    
    model = build_model_test()
    history = model.fit(partial_train_data, partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=epochs, batch_size=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()





# Helpful graphic
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()



def build_model_by_arr(arr):
    global train
    model = models.Sequential()
    model.add(layers.Dense(1, activation='relu', input_dim=1))
    for i in range(len(arr)):
        if arr[i]["activation"] == "Dropout":
            model.add(layers.Dropout(arr[i]["drop_procent"]))
        elif "regularizer" in [*arr[i].keys()]:
            model.add(layers.Dense(arr[i]["neirons"], kernel_regularizer=regularizers.l2(arr[i]["regularizer"]), 
                                   activation=arr[i]["activation"]))
        elif "activation" in [*arr[i].keys()]:
            model.add(layers.Dense(arr[i]["neirons"], activation=arr[i]["activation"]))
        else:
            model.add(layers.Dense(arr[i]["neirons"]))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])
    return model


def test_numb_neirons(arr, last, epochs):
    best_err = 1000
    best_act = ""
    best_neirons = 0
    for j in range(1, 20):
        neirons = int(last * ((3 - 0.3) / 10) * j)
        arr2 = cp.deepcopy(arr)
        arr2.append({"activation": "relu", "neirons": neirons})
        model1 = build_model_by_arr(arr)
        model1.fit(train, train_labels, epochs=epochs, verbose=0)
        test_mae_score = model1.evaluate(test, test_labels)[1]
        if best_err > test_mae_score:
            best_act = "relu"
            best_neirons = neirons

        arr2 = cp.deepcopy(arr)
        arr2.append({"activation": "sigmoid", "neirons": neirons})
        model2 = build_model_by_arr(arr)
        model2.fit(train, train_labels, epochs=epochs, verbose=0)
        test_mae_score = model2.evaluate(test, test_labels)[1]
        if best_err > test_mae_score:
            best_act = "sigmoid"
            best_neirons = neirons
            best_err = test_mae_score

        arr2 = cp.deepcopy(arr)
        arr2.append({"activation": "softmax", "neirons": neirons})
        model3 = build_model_by_arr(arr)
        model3.fit(train, train_labels, epochs=epochs, verbose=0)
        test_mae_score = model3.evaluate(test, test_labels)[1]
        if best_err > test_mae_score:
            best_act = "softmax"
            best_neirons = neirons
            best_err = test_mae_score
    
    arr.append({"activation": best_act, "neirons": best_neirons})
    return [arr, best_err]


def test_numb_dropout(arr, min_drop, max_drop, epochs):
    best_err = 1000
    best_drop = 0
    for j in range(1, 10):
        drop = (max_drop - min_drop) * j / 10
        arr2 = cp.deepcopy(arr)
        arr2.append({"activation": "Dropout", "drop_procent": drop})
        model = build_model_by_arr(arr)
        model.fit(train, train_labels, epochs=epochs, verbose=0)
        test_mae_score = model.evaluate(test, test_labels)[1]
        if best_err > test_mae_score:
            best_drop = drop
            best_err = test_mae_score
    
    arr.append({"activation": "Dropout", "drop_procent": best_drop})
    return [arr, best_err]


def test_numb_regularizers(arr, last, min_reg, max_reg, epochs):
    best_err = 1000
    best_act = ""
    best_reg = 0
    best_neirons = 0
    for i in range(1, 6):
        for j in range(1, 10):
            neirons = int(last * ((3 - 0.3) / 10) * j)
            reg = ((max_reg - min_reg) / 6.0) * i
            arr2 = cp.deepcopy(arr)
            arr2.append({"activation": "relu", "neirons": neirons, "regularizer": reg})
            model1 = build_model_by_arr(arr)
            model1.fit(train, train_labels, epochs=epochs, verbose=0)
            test_mae_score = model1.evaluate(test, test_labels)[1]
            if best_err > test_mae_score:
                best_act = "relu"
                best_reg = reg
                best_neirons = neirons
                best_err = test_mae_score
            
            arr2 = cp.deepcopy(arr)
            arr2.append({"activation": "sigmoid", "neirons": neirons, "regularizer": reg})
            model2 = build_model_by_arr(arr)
            model2.fit(train, train_labels, epochs=epochs, verbose=0)
            test_mae_score = model2.evaluate(test, test_labels)[1]
            if best_err > test_mae_score:
                best_act = "sigmoid"
                best_reg = reg
                best_neirons = neirons
                best_err = test_mae_score
            
            arr2 = cp.deepcopy(arr)
            arr2.append({"activation": "softmax", "neirons": neirons, "regularizer": reg})
            model3 = build_model_by_arr(arr)
            model3.fit(train, train_labels, epochs=epochs, verbose=0)
            test_mae_score = model3.evaluate(test, test_labels)[1]
            if best_err > test_mae_score:
                best_act = "softmax"
                best_reg = reg
                best_neirons = neirons
                best_err = test_mae_score
    
    arr.append({"activation": best_act, "neirons": best_neirons, "regularizer": best_reg})
    return [arr, best_err]


def get_network(data, data_labels, test, test_labels, exam, exam_labels, settings):
    left_epoches, right_epoches = settings['epoches']
    arr = []
    best_err_now = 1000
    best_epoches = 100
    while right_epoches - left_epoches >= 4:
        tr = False
        last = -1
        for line in reversed(arr):
            if last == -1 and "neirons" in [*line.keys()]:
                last = line["neirons"]
        
        arr1, best_err1 = test_numb_neirons(arr, last, int((left_epoches + right_epoches) / 2))
        arr2, best_err2 = test_numb_dropout(arr, settings["dropouts"][0], settings["dropouts"][1], 
                                            int((left_epoches + right_epoches) / 2))
        arr3, best_err3 = test_numb_regularizers(arr, settings["reg_params"]["l2"][0], 
                                                settings["reg_params"]["l2"][1], last, 
                                                int((left_epoches + right_epoches) / 2))
        if best_err_now > min(best_err1, best_err2, best_err3):
            if best_err1 == min(best_err1, best_err2, best_err3):
                arr = cp.deepcopy(arr1)
            elif best_err2 == min(best_err1, best_err2, best_err3):
                arr = cp.deepcopy(arr2)
            elif best_err3 == min(best_err1, best_err2, best_err3):
                arr = cp.deepcopy(arr3)
            best_err_now = min(best_err1, best_err2, best_err3)
            tr = True
        
        m1 = left_epoches + (right_epoches - left_epoches) / 3
        model1 = build_model_by_arr(arr)
        model1.fit(train, train_labels, epochs=m1, verbose=0)
        test_mse_score1 = model1.evaluate(test, test_labels)
        
        m2 = right_epoches - (right_epoches - left_epoches) / 3
        model2 = build_model_by_arr(arr)
        model2.fit(train, train_labels, epochs=m2, verbose=0)
        test_mse_score2 = model2.evaluate(test, test_labels)
        
        m3 = left_epoches + ((right_epoches - left_epoches) / 3) * 2
        model3 = build_model_by_arr(arr)
        model3.fit(train, train_labels, epochs=m3, verbose=0)
        test_mse_score3 = model3.evaluate(test, test_labels)
        
        if best_err_now > min(test_mse_score1, test_mse_score2, test_mse_score3):
            if test_mse_score1 == min(test_mse_score1, test_mse_score2, test_mse_score3):
                r = m2
                best_epoches = m1
            elif test_mse_score2 == min(test_mse_score1, test_mse_score2, test_mse_score3):
                r = int((m2 + m3) / 2)
                l = int((m1 + m2) / 2)
                best_epoches = m2
            elif test_mse_score3 == min(test_mse_score1, test_mse_score2, test_mse_score3):
                l = m2
                best_epoches = m3
            best_err_now = min(test_mse_score1, test_mse_score2, test_mse_score3)
            tr = True
            
        if tr:
            right_epoches = left_epoches
    
    model = build_model_by_arr(arr)
    model.fit(train, train_labels, epochs=best_epoches, verbose=0)
    test_mse_score = model.evaluate(exam, exam_labels)
    print(test_mse_score)
    return model


min_epoch, max_epoch = 50, 300
min_drop, max_drop = [0.1, 0.5]
min_l1, max_l1 = [0.001, 0.01]
min_l2, max_l2 = [0.001, 0.01]
min_layer, max_layer = 3, 10
min_neirons, max_neirons = 10, 250
model = get_network(data, data_labels, test, test_labels, exam, exam_labels, {'epoches': [min_epoch, max_epoch], 
                                'optimizers': ['rmsprop'], 'loses': ['mse', 'mae'], 'metrics': ['mse', 'mae'], 
                                'activations': ['relu', 'sigmoid'], 'regularizers': ['l1', 'l2', 'l1_l2'], 
                                'reg_params': {'l1': [min_l1, max_l1], 'l2': [min_l2, max_l2]}, 
                                'dropouts': [min_drop, max_drop], 'layers': [min_layer, max_layer],
                                'neirons': [min_neirons, max_neirons]})
model.save('auto_network.h5')
