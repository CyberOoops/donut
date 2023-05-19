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
#file = os.path.join(sys.argv[1],sys.argv[2])
df = pd.read_csv(sys.argv[1])
timestamp, values, labels = df['timestamp'],df['value'],df['label']
# data_name = sys.argv[1]
# values = np.load(os.path.join('./data/','{}_value.npy'.format(data_name)))
# labels = np.load(os.path.join('./data/','{}_label.npy'.format(data_name)))
# missing = np.load(os.path.join('./data/','{}_missing.npy'.format(data_name)))

# print(values.shape,labels.shape,missing.shape)

# timestamp = np.array([])
# values = np.array([])
# labels = np.array([])
# missing = np.array([])

# for file in file_list:
#     df_now = pd.read_csv(os.path.join(sys.argv[1],file))
#     timestamp_now, values_now, labels_now = df_now['timestamp'],df_now['value'],df_now['label']
#     timestamp_now, missing_now, (values_now, labels_now) = \
#     complete_timestamp(timestamp_now, (values_now, labels_now))
#     values_now = values_now.astype(float)
#     missing2_now = np.isnan(values_now)
#     values_now[np.where(missing2_now==1)[0]] = 0
#     missing_now = np.logical_or(missing_now,missing2_now)

#     timestamp = np.append(timestamp,timestamp_now)
#     values = np.append(values,values_now)
#     labels = np.append(labels,labels_now)
#     missing = np.append(missing,missing_now)


# If there is no label, simply use all zeros.
#labels = np.zeros_like(values, dtype=np.int32)

# Complete the timestamp, and obtain the missing point indicators.
timestamp, missing, (values, labels) = \
    complete_timestamp(timestamp, (values, labels))

values = values.astype(float)
missing2 = np.isnan(values)
values[np.where(missing2==1)[0]] = 0
labels[np.where(missing2==1)[0]] = 0
missing = np.logical_or(missing,missing2)

# Split the training and testing data.
test_portion = 0.5
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
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
    trainer.fit(train_values, train_labels, train_missing, mean, std)
    time1 = time.time()
    test_score = -predictor.get_score(test_values, test_missing)
    time2 = time.time()
    
    print('test_time',time2-time1)
    print(len(test_score))
    print(len(test_labels))
    label = test_labels[119:]
    all_score = np.load('./{}_score.npy'.format(sys.argv[2]))
    all_label = np.load('./{}_label.npy'.format(sys.argv[2]))
    all_score = np.concatenate((all_score,test_score))
    all_label = np.concatenate((all_label,label))
    np.save('./{}_score.npy'.format(sys.argv[2]),all_score)
    np.save('./{}_label.npy'.format(sys.argv[2]),all_label)
    kk=7
    if sys.argv[2]=='Yahoo':
        kk=3
    max_f1,max_pre,max_recall,predict = best_f1(score=test_score,label=label)
    d_f1,d_pre,d_recall,d_predict = delay_f1(score=test_score,label=label,k=kk)
    with open('./all_result.txt','a') as f:
        f.write('time: %f f1: %f %f %f %f %f %f\n'%(time2-time1,max_f1,max_pre,max_recall,d_f1,d_pre,d_recall))