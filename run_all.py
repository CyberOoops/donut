import os
import sys
import numpy as np
from metric import best_f1,delay_f1
file_list = os.listdir(sys.argv[1])
all_score = np.array([])
all_label = np.array([])
np.save('./{}_score.npy'.format(sys.argv[1][7:]),all_score)
np.save('./{}_label.npy'.format(sys.argv[1][7:]),all_label)
for file in file_list:
    ff = os.path.join(sys.argv[1],file)
    os.system('python cpu_train_single.py {} {}'.format(ff,sys.argv[1][7:]))

all_score = np.load('./{}_score.npy'.format(sys.argv[1][7:]))
all_label = np.load('./{}_label.npy'.format(sys.argv[1][7:]))

kk=7
if sys.argv[1][7:]=='Yahoo':
    kk=3
max_f1,max_pre,max_recall,predict = best_f1(score=all_score,label=all_label)
d_f1,d_pre,d_recall,d_predict = delay_f1(score=all_score,label=all_label,k=kk)
with open('./all_f1.txt','a') as f:
    f.write('f1: %f %f %f %f %f %f\n'%(max_f1,max_pre,max_recall,d_f1,d_pre,d_recall))