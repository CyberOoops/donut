import numpy as np
import pandas as pd
from donut import complete_timestamp, standardize_kpi

import os
import time
import sys
import importlib
importlib.reload(sys)
from metric import best_f1,delay_f1
# Read the raw data.

data_name = sys.argv[1]



train_values = np.load(os.path.join('./data/','{}_train_value.npy'.format(data_name)))
valid_values = np.load(os.path.join('./data/','{}_valid_value.npy'.format(data_name)))
test_values = np.load(os.path.join('./data/','{}_test_value.npy'.format(data_name)))

train_labels = np.load(os.path.join('./data/','{}_train_label.npy'.format(data_name)))
valid_labels = np.load(os.path.join('./data/','{}_valid_label.npy'.format(data_name)))
test_labels = np.load(os.path.join('./data/','{}_test_label.npy'.format(data_name)))

train_missing = np.load(os.path.join('./data/','{}_train_missing.npy'.format(data_name)))
valid_missing = np.load(os.path.join('./data/','{}_valid_missing.npy'.format(data_name)))
test_missing = np.load(os.path.join('./data/','{}_test_missing.npy'.format(data_name)))


train_exclude_ori = np.load(os.path.join('./data/','{}_train_exclude.npy'.format(data_name)))
valid_exclude_ori = np.load(os.path.join('./data/','{}_valid_exclude.npy'.format(data_name)))
test_exclude_ori = np.load(os.path.join('./data/','{}_test_exclude.npy'.format(data_name)))
train_exclude = np.zeros_like(train_values,dtype=bool)
for i in train_exclude_ori:
    train_exclude[int(i)]=True
valid_exclude = np.zeros_like(valid_values,dtype=bool)
for i in valid_exclude_ori:
    valid_exclude[int(i)]=True


# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=train_missing)
valid_values, _, _ = standardize_kpi(valid_values, mean=mean, std=std)
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

import tensorflow as tf
from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential

# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )

from donut import DonutTrainer, DonutPredictor

trainer = DonutTrainer(model=model, model_vs=model_vs)
predictor = DonutPredictor(model)


with tf.Session().as_default():
    trainer.fit(train_values, train_labels, train_missing, mean, std,train_values=train_values,valid_values=valid_labels,train_labels=train_labels,valid_labels=valid_labels,train_missing=train_missing,valid_missing=valid_missing,train_exclude=train_exclude,valid_exclude=valid_exclude)
    time1 = time.time()
    test_score = -predictor.get_score(test_values, test_missing)
    time2 = time.time()
    
    print('test_time',time2-time1)
    print(len(test_score))
    print(len(test_labels))
    #label = test_labels[119:]
    mask = np.ones_like(test_labels,dtype=bool)
    mask2 = np.ones_like(test_score,dtype=bool)
    for i in test_exclude_ori:
        mask[int(i)+1:int(i)+120]=False
        mask2[int(i)+1-119:int(i)+120-119]=False
    mask[:119]=False
    label = test_labels[mask]
    test_score = test_score[mask2]
    print(len(label),len(test_score))
    
    kk=7
    if sys.argv[1]=='Yahoo':
        kk=3
    elif sys.argv[1]=='NAB':
        kk=150
    max_f1,max_pre,max_recall,predict = best_f1(score=test_score,label=label)
    d_f1,d_pre,d_recall,d_predict = delay_f1(score=test_score,label=label,k=kk)
    with open('./all_result.txt','a') as f:
        f.write('time: %f f1: %f %f %f %f %f %f\n'%(time2-time1,max_f1,max_pre,max_recall,d_f1,d_pre,d_recall))