import os
import torch
import numpy as np
import pickle
from tqdm import tqdm

dataset = "gym99"
split = "split_14"

# split_1 = [4,19,31,47,51]
# split_2 = [12,29,32,44,59]
# split_3 = [7,20,28,39,58]

# split_4 = [3, 18, 26, 38, 41, 60, 87, 99, 102, 110]
# split_5 = [5, 12, 14, 15, 17, 42, 67, 82, 100, 119]
# split_6 = [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]

# split_7 = [1, 9, 20, 34, 50]
# split_8 = [3, 14, 29, 31, 49]
# split_9 = [2, 15, 39, 41, 43]

# ntu 60
split_1 = [10, 11, 19, 26, 56]
split_2 = [0, 8, 15, 28, 46]
split_3 = [15, 19, 23, 47, 50]
split_4 = [29, 37, 38, 45, 55]

# ntu 120
split_5 = []
split_6 = [0, 4, 6, 7, 24, 37, 54, 59, 97, 113]  # split 2
split_7 = [63, 79, 86, 92, 98, 100, 103, 110, 111, 117]  # split 3
split_8 = [9, 14, 17, 44, 60, 75, 81, 89, 108, 110]  # split 4

# pku51
split_9 = [10, 19, 27, 38, 48]  # split 1
split_10 = [0, 9, 17, 30, 42]  # split 2
split_11 = [18, 24, 31, 43, 45]  # split 3

# gym99
split_12 = [2, 35, 61, 66, 70, 74, 81, 85]  # split 2
split_13 = [2, 3, 26, 48, 52, 82, 83, 92]  # split 3
split_14 = [11, 22, 33, 35, 37, 40, 93, 98]  # split 4

train_path = './sourcedata/'+dataset+'_frame50/xsub/train_position.npy'
test_path = './sourcedata/'+dataset+'_frame50/xsub/val_position.npy'
train_label_path = './sourcedata/'+dataset+'_frame50/xsub/train_label.pkl'
test_label_path = './sourcedata/'+dataset+'_frame50/xsub/val_label.pkl'

os.makedirs("./data/zeroshot/"+dataset+"/"+split, exist_ok=True)
_ = open('./data/zeroshot/'+dataset+'/'+split+'/__init__.py', 'w')

seen_train_data_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_train_data.npy"
seen_train_label_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_train_label.npy"
seen_test_data_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_test_data.npy"
seen_test_label_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_test_label.npy"
unseen_data_path = "./data/zeroshot/"+dataset+"/"+split+"/unseen_data.npy"
unseen_label_path = "./data/zeroshot/"+dataset+"/"+split+"/unseen_label.npy"

with open(train_label_path, 'rb') as f:
    _, train_label = pickle.load(f)

with open(test_label_path, 'rb') as f:
    _, test_label = pickle.load(f)

train_data = np.load(train_path)
test_data = np.load(test_path)

print("train size:",train_data.shape)
print("test size:",test_data.shape)

seen_train_data = []
seen_train_label = []
seen_test_data = []
seen_test_label = []
unseen_data = []
unseen_label = []

for i in range(len(train_label)):
    if train_label[i] not in eval(split):
        seen_train_label.append(train_label[i])
        seen_train_data.append(train_data[i])

for i in range(len(test_label)):
    if test_label[i] in eval(split):
        unseen_label.append(test_label[i])
        unseen_data.append(test_data[i])
    else:
        seen_test_label.append(test_label[i])
        seen_test_data.append(test_data[i])

seen_train_data = np.array(seen_train_data)
seen_test_data = np.array(seen_test_data)
unseen_data = np.array(unseen_data)

print(seen_train_data.shape)
print(len(seen_train_label))
print(seen_test_data.shape)
print(len(seen_test_label))

np.save(seen_train_data_path, seen_train_data)
np.save(seen_train_label_path, seen_train_label)
np.save(seen_test_data_path, seen_test_data)
np.save(seen_test_label_path, seen_test_label)
np.save(unseen_data_path, unseen_data)
np.save(unseen_label_path, unseen_label)
