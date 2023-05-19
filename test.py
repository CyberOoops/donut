import numpy as np

l  = np.load('./data/real-world_label.npy')
l = np.isnan(l)
print(np.sum(l))