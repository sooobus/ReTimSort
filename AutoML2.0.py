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
import time
from datetime import datetime
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
    # path = sys.argv[0]
    path = "/home/leha/Desktop/projects/all_timsort_data"
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
    test_p, test_labels_p = np.asarray([]), np.asarray([])
    train_p, train_labels_p = np.asarray([]), np.asarray([])
    
    left, right = TRAIN + 1, len(data)
    for i in range(left, right):
        if i % 2 == 0:
            test_p = np.append(test_p, data[i])
            test_labels_p = np.append(test_labels_p, data_labels[i])
        else:
            if len(exam) < EXAM and i % 3 != 0:
                exam = np.append(exam, data[i])
                exam_labels = np.append(exam_labels, data_labels[i])
            else:
                train_p = np.append(train_p, data[i])
                train_labels_p = np.append(train_labels_p, data_labels[i])

    test, test_labels = test_p, test_labels_p

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
        self.epochs = epochs
        self.arr = arr
        self.model = self.build_model_by_arr(self.arr)
    
    def build_model_by_arr(self, arr):
        model = models.Sequential()
        model.add(layers.Dense(1, activation='relu', input_dim=1))
        for i in range(len(arr)):
            if "activation" in [*arr[i].keys()] and arr[i]["activation"] == "dropout":
                model.add(layers.Dropout(arr[i]["drop_procent"]))
            elif "regularizer" in [*arr[i].keys()]:
                model.add(layers.Dense(arr[i]["neirons"], 
                                       kernel_regularizer=regularizers.l2(arr[i]["regularizer"]), activation=arr[i]["activation"]))
            elif "activation" in [*arr[i].keys()]:
                model.add(layers.Dense(arr[i]["neirons"], activation=arr[i]["activation"]))
            else:
                model.add(layers.Dense(arr[i]["neirons"]))
        model.add(layers.Dense(1))
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])
        return model
    
    def add_layer_and_test(self, new_layer, num_test_data):
        self.arr += [new_layer]
        model = self.build_model_by_arr(self.arr)
        self.model = model
        return self.test_model(num_test_data)
    
    def add_layer_and_test_not_save(self, new_layer, num_test_data):
        model = self.build_model_by_arr(self.arr + [new_layer])
        return self.test_model_not_self(model, 250, num_test_data)
    
    def get_layers(self):
        return self.arr
    
    def test_model(self, num_test_data):
        global train, train_labels
        global test, test_labels
        self.model.fit(train, train_labels, epochs=self.epochs, verbose=0)
        return self.model.evaluate(test[num_test_data], test_labels[num_test_data])[1]
    
    def build_model_and_save(self):
        model = self.build_model_by_arr(self.arr)
        self.model = model
    
    def get_model(self):
        return self.model
    
    def test_model_not_self(self, model, epochs, num_test_data):
        global train, train_labels
        global test, test_labels
        model.fit(train, train_labels, epochs=epochs, verbose=0)
        return model.evaluate(test[num_test_data], test_labels[num_test_data])[1]



def get_act_err(new_layer, network, min_max, num_test_data):
    err = 0
    act = {}
    if new_layer[0] in ["relu", "sigmoid", "softmax"]:
        num_neirons = int((min_max['neirons'][1] - min_max['neirons'][0]) / 100 * new_layer[1] + min_max['neirons'][0])
        act = {"activation": new_layer[0], "neirons": num_neirons}
        err = network.add_layer_and_test_not_save(act, num_test_data)
    elif new_layer[0] == "regulaizer_l2":
        best_err_reg = 10000
        best_act_reg = "relu"
        for activation in product(["relu", "sigmoid", "softmax"], [_ for _ in range(1, 100, 30)]):
            num_neirons = int((min_max['neirons'][1] - min_max['neirons'][0]) / 100 * activation[1] + min_max['neirons'][0])
            num_reg = (min_max['reg_params']['l2'][1] - min_max['reg_params']['l2'][0]) / 100 * new_layer[1] + min_max['reg_params']['l2'][0]
            act = {"activation": activation[0], "neirons": num_neirons, "regularizer": num_reg}
            err = network.add_layer_and_test_not_save(act, num_test_data)
            if best_err_reg > err:
                best_err_reg, best_act_reg = err, act
        err = best_err_reg
        act = best_act_reg
    elif new_layer[0] == "dropout":
        num_drop = (min_max['dropouts'][1] - min_max['dropouts'][0]) / 100 * new_layer[1] + min_max['dropouts'][0]
        act = {"activation": new_layer[0], "drop_procent": num_drop}
        err = network.add_layer_and_test_not_save(act, num_test_data)
    else:
        num_neirons = int((min_max['neirons'][1] - min_max['neirons'][0]) / 100 * new_layer[1] + min_max['neirons'][0])
        act = {"neirons": num_neirons}
        err = network.add_layer_and_test_not_save(act, num_test_data)
    return (err, act)







def get_network(data, data_labels, test, test_labels, exam, exam_labels, settings):
    left_epoches, right_epoches = settings['epoches']
    arr = []
    last_best_err = 10000
    networks_quantity = 40
    epochs = 300
    const_best_networks = 5
    best_networks = {10000 + _: Model_keeper([], epochs) for _ in range(networks_quantity)}
    tr = True
    num_test_data = 0
    while tr or reduce((lambda x, y: x + y), [*best_networks.keys()]) / networks_quantity > last_best_err and networks_quantity >= const_best_networks:
        print(networks_quantity, datetime.now())
        last_best_arr = reduce((lambda x, y: x + y), [*best_networks.keys()]) / networks_quantity
        tr = False
        best_errs = {}
        best_acts = {}
        for key in sorted([*best_networks.keys()])[:const_best_networks]:
            network = best_networks[key]
            best_err_now = {10000: "relu", 10001: "relu"}
            best_networks_now = {}
            for new_layer in product(["relu", "sigmoid", "softmax", "regulaizer_l2", "dropout", ""], 
                                 [_ for _ in range(1, 100, 10)]):
                
                err, act = get_act_err(new_layer, network, {"neirons": [settings['neirons'][0], settings['neirons'][1]],
                                                            "reg_params": [settings['reg_params']['l2'][0], settings['reg_params']['l2'][1]],
                                                            "dropouts": [settings['dropouts'][0], settings['dropouts'][1]]}, num_test_data)
                if max(best_err_now) > err:
                    del best_err_now[max(best_err_now)]
                    best_err_now[err] = act
            networks_quantity += 1
            if [*best_err_now.keys()][0] in [*best_networks.keys()] or [*best_err_now.keys()][0] in [*best_networks_now.keys()]:
                best_err_now[[*best_err_now.keys()][0] + 0.000001] = best_err_now[[*best_err_now.keys()][0]]
                # del best_err_now[[*best_err_now.keys()][0]]
            if [*best_err_now.keys()][1] in [*best_networks.keys()] or [*best_err_now.keys()][1] in [*best_networks_now.keys()]:
                best_err_now[[*best_err_now.keys()][1] + 0.000001] = best_err_now[[*best_err_now.keys()][1]]
                # del best_err_now[[*best_err_now.keys()][1]]
            
            best_networks_now[[*best_err_now.keys()][0]] = Model_keeper(network.get_layers() + [[*best_err_now.values()][0]], epochs)
            best_networks_now[[*best_err_now.keys()][1]] = Model_keeper(network.get_layers() + [[*best_err_now.values()][1]], epochs)
            
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
            min_err = 10000
            for err in best_errs.keys():
                if min_err > err:
                    min_err = err
                    best_act = best_errs[err]
        
        for err in sorted([*best_networks.keys()])[const_best_networks + 1:]:
            err_n = best_networks[err].add_layer_and_test(best_act, num_test_data)
            while err_n in [*best_networks.keys()] or err_n in [*best_networks_now.keys()]:
                err_n += 0.000001
            best_networks_now[err_n] = best_networks[err] 
        
        best_networks = {err: best_networks_now[err] for err in sorted([*best_networks_now.keys()])[:int((len(best_networks_now) - (len(best_networks_now) % 2)) / 2)]}
        networks_quantity = len([*best_networks.keys()])
        num_test_data += 1
        num_test_data %= len(test)
    
    return [best_networks[err] for err in sorted([*best_networks.keys()])[:5]]







def get_best_epochs(network, test, test_labels, settings):
    left, right = settings["epoches"][0], settings["epoches"][1]
    last_err = 10000
    tr = True
    while right - left > 5 and tr:
        m1 = int((left + right) / 3)
        m3 = right - m1
        m2 = int((m1 + m3) / 2)
        sr1, sr2, sr3 = 0, 0, 0
        for num_test in range(len(test)):
            network.epoches = m1
            sr1 += network.test_model(num_test)
            network.epoches = m2
            sr2 += network.test_model(num_test)
            network.epoches = m3
            sr3 += network.test_model(num_test)
        sr1 /= len(test)
        sr2 /= len(test)
        sr3 /= len(test)
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
    if tr:
        sr_d = [(10000, i) for i in range(left, right + 1)]
        for num_test in range(len(test)):
            for i in range(right - left + 1):
                network.epoches = sr_d[i][1]
                sr_d[i][0] += network.test_model(num_test)
        return ((sorted(sr_d)[0][0] / len(test)), sorted(sr_d)[0][0])
    else:
        sr_l, sr_r = 0, 0
        for num_test in range(len(test)):
            network.epoches = left
            sr_l += network.test_model(num_test)
            network.epoches = right
            sr_r += network.test_model(num_test)
        if sr_l < sr_r:
            return (sr_l / len(test), left)
        else:
            return (sr_r / len(test), right)



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
networks = get_network(data, data_labels, test, test_labels, exam, exam_labels, settings)



best_model, best_err = 0, 10000
for network in networks:
    err_now, epochs = get_best_epochs(network, test, test_labels, settings)
    if err_now < best_err:
        best_model, best_err = network, err_now
best_model.build_model_and_save()
best_model.get_model().save('auto_network.h5')
