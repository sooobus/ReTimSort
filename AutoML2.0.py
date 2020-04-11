from datetime import datetime
import math
import pandas as pd
from Timsort_lib import TimSort
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import datetime as dt
from itertools import product
from functools import reduce
import copy as cp
import os
import sys
import time
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

from math import sqrt
from math import floor

TRAIN = 6171
EXAM = 500

from math import sqrt
from math import floor

TRAIN = 6171
EXAM = 500

def get_data():
    def get_arr_by_str(s):
        return s.replace('[', '').replace(']', '').replace(',', '').split()

    def get_data_from_file(filename, path):
        db = pd.read_csv(path + '/' + file)
        df = pd.DataFrame({'Quantity': [], 'Best_Minrun': [], 'Best_Time': []})
        for i in range(len(db['Best_minrun'])):
            df['Best_Minrun'][i] = get_arr_by_str(db['Best_minrun'][i])

        for i in range(len(df['Best_Minrun'])):
            for j in range(len(df['Best_Minrun'][i])):
                df['Best_Minrun'][i][j] = int(df['Best_Minrun'][i][j])

        for i in range(len(db['Quantity'])):
            df['Quantity'][i] = int(db['Quantity'][i])

        for i in range(len(db['Best_time'])):
            df['Best_Time'][i] = get_arr_by_str(db['Best_time'][i])

        local_train_pairs = {}
        for i in range(len(df['Quantity'])):
            pair = float(df['Quantity'][i]), float(df['Best_Minrun'][i][0])
            local_train_pairs[pair] = 0
        return local_train_pairs

    pd.options.mode.chained_assignment = None
    path = sys.argv[1]
    # path = "/home/leha/Desktop/projects/all_timsort_data"
    directory = os.path.join(path)
    train_pairs = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                train_pairs.update(get_data_from_file(file, path))

    return train_pairs


def get_parsed_data():
    def normalize_data(mean, std, data):
        data -= mean
        data /= std

    def divide_data(data):
        return [data[:floor(len(data) / 4)],
                data[floor(len(data) / 4):floor(len(data) / 2)],
                data[floor(len(data) / 2):-floor(len(data) / 4)],
                data[-floor(len(data) / 4):]]

    pairs = get_data()
    pairs = sorted([*pairs.keys()])

    data, data_labels = np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])

    train, train_labels = data[:TRAIN], data_labels[:TRAIN]

    exam, exam_labels = np.asarray([]), np.asarray([])
    # test_p, test_labels_p = np.asarray([]), np.asarray([])
    test_pairs = []
    train_p, train_labels_p = np.asarray([]), np.asarray([])

    left, right = TRAIN + 1, len(data)
    for i in range(left, right):
        if i % 2 == 0:
            test_pairs += [(data[i], data_labels[i])]
        else:
            if len(exam) < EXAM and i % 3 != 0:
                exam = np.append(exam, data[i])
                exam_labels = np.append(exam_labels, data_labels[i])
            else:
                train_p = np.append(train_p, data[i])
                train_labels_p = np.append(train_labels_p, data_labels[i])
    random.shuffle(test_pairs)
    test = np.asarray([pair[0] for pair in test_pairs])
    test_labels = np.asarray([pair[1] for pair in test_pairs])

    train, train_labels = np.append(train, train_p), np.append(train_labels, train_labels_p)

    mean = train.mean(axis=0)
    train -= mean
    std = train.std(axis=0)
    train /= std

    normalize_data(mean, std, test)
    normalize_data(mean, std, exam)

    test, test_labels = divide_data(test), divide_data(test_labels)

    data_p = {'mean': mean, 'std': std}
    with open('data.json', 'w') as f_write:
        json.dump(data_p, f_write)

    return [train, train_labels, test, test_labels, exam, exam_labels]


class Model_keeper:
    def __init__(self, arr, epochs):
        """
        arr -> [{"activation": 'relu'///'sigmoid'///'softmax'///'dropout', "regulaizer": float,
        "neirons": int, "drop_procent": float}, {layer2}, {layer3}, ...]
        """
        self.epochs = epochs
        self.arr = arr
        self.model = self.build_model_by_arr(self.arr)

    def build_model_by_arr(self, arr):
        model = models.Sequential()
        model.add(layers.Dense(1, input_dim=1))
        for i in range(len(arr)):
            if arr[i]["activation"] == "dropout":
                model.add(layers.Dropout(arr[i]["neirons"]))
            elif arr[i]["regularizer"] != 0:
                model.add(layers.Dense(int(arr[i]["neirons"]),
                                       kernel_regularizer=regularizers.l2(arr[i]["regularizer"]),
                                       activation=arr[i]["activation"]))
            elif arr[i]["activation"] != "":
                model.add(layers.Dense(int(arr[i]["neirons"]), activation=arr[i]["activation"]))
            else:
                model.add(layers.Dense(int(arr[i]["neirons"])))
        model.add(layers.Dense(1))
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])
        return model

    def test_model(self, train, train_labels, test, test_labels, num_test_data):
        self.model.fit(train, train_labels, epochs=self.epochs, verbose=0)
        err = self.model.evaluate(test[num_test_data], test_labels[num_test_data])[1]  # not self
        return err


def get_act(new_layer, d_act):
    act = {}
    num_reg = 0
    drop_procent = 0
    if len(new_layer[0].split()) > 1:
        num_reg = ((d_act[new_layer[0].split()[0]][1] - d_act[new_layer[0].split()[0]][0]) 
                    / 100) * new_layer[1] * d_act[new_layer[0].split()[0]][2] + d_act[new_layer[0].split()[0]][0]

    num_neirons = ((d_act[new_layer[0].split()[-1]][1] - d_act[new_layer[0].split()[-1]][0]) 
                    / 100) * new_layer[1] * d_act[new_layer[0].split()[-1]][2] + d_act[new_layer[0].split()[-1]][0]

    act["activation"], act["neirons"] = new_layer[0].split()[-1], num_neirons
    act["regularizer"] = num_reg
    return act


MIN_NETWORKS = 1
EPOCHS = 300
BEST_NETWORKS = 10


def get_network(train, train_labels, test, test_labels, exam, exam_labels, settings):
    d_act = {"relu": [settings['neirons'][0], settings['neirons'][1], 1]}
    d_act["sigmoid"], d_act["softmax"], d_act[""] = d_act["relu"], d_act["relu"], d_act["relu"]
    d_act["dropout"] = [settings['dropouts'][0], settings['dropouts'][1], 1]
    d_act["regulaizer"] = [settings['reg_params']['l2'][0], settings['reg_params']['l2'][1], 3]
    arr = []
    last_best_err = 10000
    num_test_data = 0
    
    best_networks = {}
    for new_layer in product(["relu", "sigmoid", "softmax", "regulaizer relu", "regulaizer sigmoid", "dropout", ""], [_ for _ in range(1, 100, 30)]):
        if new_layer[1] * d_act[new_layer[0].split()[0]][2] > 100:
            continue
        
        act = get_act(new_layer, d_act)
        
        network = Model_keeper([act], EPOCHS)
        err = network.test_model(train, train_labels, test, test_labels, num_test_data)
        best_networks[err] = network
    print("end", "****")
    networks_quantity = len([*best_networks.keys()])
    while reduce((lambda x, y: x + y),
                 [*best_networks.keys()]) / networks_quantity < last_best_err and networks_quantity >= MIN_NETWORKS:
        last_best_arr = reduce((lambda x, y: x + y), [*best_networks.keys()]) / networks_quantity
        best_errs, best_acts = {}, {}
        best_networks_now = {}
        for key in sorted([*best_networks.keys()])[:BEST_NETWORKS]:
            network = best_networks[key]
            best_err_now = {10000: "relu", 10001: "relu"}
            for new_layer in product(["relu", "sigmoid", "softmax", "regulaizer relu", "regulaizer sigmoid", "dropout", ""], [_ for _ in range(1, 100, 10)]):
                if new_layer[1] * d_act[new_layer[0].split()[0]][2] > 100:
                    continue

                act = get_act(new_layer)

                network_now = Model_keeper(network.arr + [act], EPOCHS)
                err = network_now.test_model(train, train_labels, test, test_labels, num_test_data)

                print(max(best_err_now))
                if max(best_err_now) > err:
                    del best_err_now[max(best_err_now)]
                    best_err_now[err] = network_now.arr[-1]

                num_test_data = (num_test_data + 1) % len(test)

            networks_quantity += 1
            if [*best_err_now.keys()][0] in [*best_networks.keys()] or [*best_err_now.keys()][0] in [
                *best_networks_now.keys()]:
                best_err_now[[*best_err_now.keys()][0] + 0.000001] = best_err_now[[*best_err_now.keys()][0]]
            if [*best_err_now.keys()][1] in [*best_networks.keys()] or [*best_err_now.keys()][1] in [
                *best_networks_now.keys()]:
                best_err_now[[*best_err_now.keys()][1] + 0.000001] = best_err_now[[*best_err_now.keys()][1]]

            best_networks_now[[*best_err_now.keys()][0]] = Model_keeper(network.arr + [[*best_err_now.values()][0]],
                                                                        EPOCHS)
            best_networks_now[[*best_err_now.keys()][1]] = Model_keeper(network.arr + [[*best_err_now.values()][1]],
                                                                        EPOCHS)

            best_errs[[*best_err_now.keys()][0]] = best_err_now[[*best_err_now.keys()][0]]
            best_errs[[*best_err_now.keys()][1]] = best_err_now[[*best_err_now.keys()][1]]

        num_elements = {}
        for act in [*best_errs.values()]:
            if act in [*num_elements.keys()]:
                num_elements[str(act)] += 1
            else:
                num_elements[str(act)] = 1

        best_act = {}
        if sorted([*num_elements.values()])[-1] > 1:
            best_num = sorted([*num_elements.values()])[-1]
            for key in [*num_elements.keys()]:
                if best_num == num_elements[key]:
                    best_act = dict(key)
        else:
            best_act = best_errs[reduce(lambda a, b: a if (a < b) else b, best_errs.keys())]

        for err in sorted([*best_networks.keys()])[BEST_NETWORKS:]:
            best_networks[err] = Model_keeper(best_networks[err].arr + [best_act], EPOCHS)
            err_n = best_networks[err].test_model(train, train_labels, test, test_labels, num_test_data)
            while err_n in [*best_networks.keys()] or err_n in [*best_networks_now.keys()]:
                err_n += 0.000001
            best_networks_now[err_n] = best_networks[err]
            num_test_data = (num_test_data + 1) % len(test)

        best_networks = {err: best_networks_now[err] for err in
                         sorted([*best_networks_now.keys()])[:floor((len(best_networks_now) + 1) / 2)]}
        networks_quantity = len([*best_networks.keys()])
        num_test_data = (num_test_data + 1) % len(test)

    return [best_networks[err] for err in sorted([*best_networks.keys()])[:min(3, len([*best_networks.keys()]))]]


def get_best_epochs(network, train, train_labels, test, test_labels, settings):
    left, right = settings["epoches"][0], settings["epoches"][1]
    last_err = 10000
    tr = True
    while right - left > 5 and tr:
        m1 = int((left + right) / 3)
        m3 = right - m1
        m2 = int((m1 + m3) / 2)
        sr1, sr2, sr3 = 0, 0, 0
        for num_test in range(len(test)):
            network.epochs = m1
            sr1 += network.test_model(train, train_labels, test, test_labels, num_test)
            network.epochs = m2
            sr2 += network.test_model(train, train_labels, test, test_labels, num_test)
            network.epochs = m3
            sr3 += network.test_model(train, train_labels, test, test_labels, num_test)

        sr1, sr2, sr3 = [sr1 / len(test), sr2 / len(test), sr3 / len(test)]
        if min(sr1, sr2, sr3) < last_err:
            last_err = min(sr1, sr2, sr3)
            if min(sr1, sr2, sr3) == sr1:
                r = m2
            elif min(sr1, sr2, sr3) == sr2:
                l = int((m1 + m2) / 2)
                r = int((m2 + m3) / 2)
            else:
                l = m2
        else:
            tr = False

    sr_d = [(10000, i) for i in range(left, right + 1)]
    for num_test_data in range(len(test)):
        for i in range(right - left + 1):
            network.epochs = sr_d[i][1]
            sr_d[i][0] += network.test_model(train, train_labels, test, test_labels, num_test_data)
    return ((sorted(sr_d)[0][0] / len(test)), sorted(sr_d)[0][0])


train, train_labels, test, test_labels, exam, exam_labels = get_parsed_data()

min_epoch, max_epoch = 50, 350
min_drop, max_drop = [0.1, 0.5]
min_l1, max_l1 = [0.001, 0.01]
min_l2, max_l2 = [0.001, 0.01]
min_layer, max_layer = 3, 10
min_neirons, max_neirons = 10, 250
settings = {'epoches': [min_epoch, max_epoch],
            'optimizers': ['rmsprop'], 'loses': ['mse', 'mae'], 'metrics': ['mse', 'mae'],
            'activations': ['relu', 'sigmoid'], 'regularizers': ['l1', 'l2', 'l1_l2'],
            'reg_params': {'l1': [min_l1, max_l1], 'l2': [min_l2, max_l2]},
            'dropouts': [min_drop, max_drop], 'layers': [min_layer, max_layer],
            'neirons': [min_neirons, max_neirons]}
networks = get_network(train, train_labels, test, test_labels, exam, exam_labels, settings)

num = 1
for network in networks:
    network.model.save('auto_network{}.h5'.format(num))
    num += 1

best_model, best_err = 0, 10000
num = 0
for network in networks:
    err_now, epochs = get_best_epochs(network, train, train_labels, test, test_labels, settings)
    network.save('{}auto_network.h5'.format(num))
    print(err_now)
    if err_now < best_err:
        best_model, best_err = network, err_now
    num += 1

best_model.model.save('best_auto_network.h5')
