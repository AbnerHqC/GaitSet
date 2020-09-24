# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):
    print("Initializing data source...")
    # data_loder.py: load_data(config.py: conf@'data', whether need cache)
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    if train:
        print("Loading training data...")
        # Dataset Train.load_all_data
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    print("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    # A deep copy constructs a new compound object and then,
    # recursively, inserts copies into it of the objects found in the original.
    #    "model": {
    #     'hidden_dim': 256,
    #     'lr': 1e-4,
    #     'hard_or_full_trip': 'full',
    #     'batch_size': (8, 16),
    #     'restore_iter': 0,
    #     'total_iter': 80000,
    #     'margin': 0.2,
    #     'num_workers': 3,
    #     'frame_num': 30,
    #     'model_name': 'GaitSet',
    # },
    model_param = deepcopy(model_config)
    # add other parameters in the model dictionary
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']
    # Batch size is a term used in machine learning
    # and refers to the number of training examples utilized in one iteration.
    batch_size = int(np.prod(model_config['batch_size']))
    # Define the saved model name
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    # create Model object
    m = Model(**model_param)
    print("Model initialization complete.")
    return m, model_param['save_name']

# Note that all config refer to config.py.conf
def initialization(config, train=False, test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH) # Change current work path to WORK_PATH
    # os.environ[“CUDA_VISIBLE_DEVICES”] = “0,1”
    # 设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, test_source)