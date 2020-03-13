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
