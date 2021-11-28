import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm_notebook
from multiprocessing import Process, Manager, Pool
import os
import datetime
from sklearn.utils import shuffle


def train_validation_test_split(class_num, params):
    data_name = os.listdir(params["multilabel_data_folder"])
    random.shuffle(data_name)
    
    train_id_tmp = data_name[0:round(len(data_name)*0.8)]
    val_id_tmp = data_name[round(len(data_name)*0.8)+1:round(len(data_name)*0.9)]
    test_id_tmp = data_name[round(len(data_name)*0.9)+1:len(data_name)-1]
    
    train_label = np.array([np.load(params["multilabel_label_folder"] + i) for i in tqdm_notebook(train_id_tmp)])
    val_label = np.array([np.load(params["multilabel_label_folder"] + i) for i in tqdm_notebook(val_id_tmp)])
    test_label = np.array([np.load(params["multilabel_label_folder"] + i) for i in tqdm_notebook(test_id_tmp)])
    
    train_id = np.array([params["multilabel_data_folder"] + i for i in train_id_tmp]).reshape(-1, 1)
    val_id = np.array([params["multilabel_data_folder"] + i for i in val_id_tmp]).reshape(-1, 1)
    test_id = np.array([params["multilabel_data_folder"] + i for i in test_id_tmp]).reshape(-1, 1)
    
    train_id = train_id.reshape(-1).tolist()
    val_id = val_id.reshape(-1).tolist()
    test_id = test_id.reshape(-1).tolist()
    
    train_label = dict(zip(train_id, train_label))
    val_label = dict(zip(val_id, val_label))
    test_label = dict(zip(test_id, test_label))

    return train_id, train_label, val_id, val_label,test_id, test_label