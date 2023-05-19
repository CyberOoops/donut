import os
import sys
import pandas as pd
import numpy as np
from donut import complete_timestamp, standardize_kpi
if __name__  == '__main__':
    train_exclude = np.array([])
    valid_exclude = np.array([])
    test_exclude = np.array([])

    train_value = np.array([])
    valid_value = np.array([])
    test_value = np.array([])
    train_label = np.array([])
    valid_label = np.array([])
    test_label = np.array([])
    train_missing = np.array([])
    valid_missing = np.array([])
    test_missing = np.array([])
    file_list = os.listdir(sys.argv[1])
    for file in file_list:
        df_now = pd.read_csv(os.path.join(sys.argv[1],file))
        #df_now = df_now.fillna(0)
        timestamp_now, values_now, labels_now = df_now['timestamp'],df_now['value'],df_now['label']
        timestamp_now, missing_now, (values_now, labels_now) = \
            complete_timestamp(timestamp_now, (values_now, labels_now))
        values_now = values_now.astype(float)
        missing2_now = np.isnan(values_now)
        values_now[np.where(missing2_now==1)[0]] = 0
        missing_now = np.logical_or(missing_now,missing2_now)
        labels_now[np.where(missing_now==1)[0]] = 0
        len_now = len(values_now)
        train_value = np.append(train_value,values_now[:int(0.35*len_now)])
        train_label = np.append(train_label,labels_now[:int(0.35*len_now)])
        train_missing = np.append(train_missing,missing_now[:int(0.35*len_now)])

        valid_value = np.append(valid_value,values_now[int(0.35*len_now):int(0.5*len_now)])
        valid_label = np.append(valid_label,labels_now[int(0.35*len_now):int(0.5*len_now)])
        valid_missing = np.append(valid_missing,missing_now[int(0.35*len_now):int(0.5*len_now)])

        test_value = np.append(test_value,values_now[int(0.5*len_now):])
        test_label = np.append(test_label,labels_now[int(0.5*len_now):])
        test_missing = np.append(test_missing,missing_now[int(0.5*len_now):])
        
        train_exclude = np.append(train_exclude,len(train_value)-1)
        valid_exclude = np.append(valid_exclude,len(valid_value)-1)
        test_exclude = np.append(test_exclude,len(test_value)-1)
    value = np.concatenate((train_value,valid_value,test_value))
    label = np.concatenate((train_label,valid_label,test_label))
    missing = np.concatenate((train_missing,valid_missing,test_missing))
    np.save('./data/{}_train_value.npy'.format((sys.argv[1])[7:]),train_value)
    np.save('./data/{}_valid_value.npy'.format((sys.argv[1])[7:]),valid_value)
    np.save('./data/{}_test_value.npy'.format((sys.argv[1])[7:]),test_value)
    np.save('./data/{}_train_label.npy'.format((sys.argv[1])[7:]),train_label)
    np.save('./data/{}_valid_label.npy'.format((sys.argv[1])[7:]),valid_label)
    np.save('./data/{}_test_label.npy'.format((sys.argv[1])[7:]),test_label)
    np.save('./data/{}_train_missing.npy'.format((sys.argv[1])[7:]),train_missing)
    np.save('./data/{}_valid_missing.npy'.format((sys.argv[1])[7:]),valid_missing)
    np.save('./data/{}_test_missing.npy'.format((sys.argv[1])[7:]),test_missing)


    print(train_exclude)
    np.save('./data/{}_train_exclude.npy'.format((sys.argv[1])[7:]),train_exclude)
    np.save('./data/{}_valid_exclude.npy'.format((sys.argv[1])[7:]),valid_exclude)
    np.save('./data/{}_test_exclude.npy'.format((sys.argv[1])[7:]),test_exclude)
    # np.save('./anomaly-transformer/{}_train.npy'.format((sys.argv[1])[7:]),np.concatenate((train_value,valid_value)))
    # np.save('./anomaly-transformer/{}_test.npy'.format((sys.argv[1])[7:]),test_value)
    # np.save('./anomaly-transformer/{}_test_label.npy'.format((sys.argv[1])[7:]),test_label)