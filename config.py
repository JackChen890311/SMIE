import os
from sacred import Experiment

ex = Experiment("CMIM", save_git_info=False)
# Parameters Setting
# Dataset: NTU60   NTU120  PKU51  GYM99 
# LR:      1e-5    1e-4    1e-5   1e-5
# Margin:  0.1     0.5     0.01   0.01

@ex.config
def my_config():
    track = "main"
    split = '14'
    dataset = "gym99"
    lr = 1e-5
    margin = 0.01
    weight_decay = 0
    epoch_num = 100
    batch_size = 128
    weight_path = './module/gcn/model/split_'+split+".pt"
    log_path = './output/log/split_'+split+'.log'
    save_path = "./output/model/"+split+"_train.pt"
    ############################## ST-GCN ###############################
    in_channels = 2
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'coco',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ############################# downstream #############################
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

    unseen_label = eval('split_'+split)
    visual_size = 256
    language_size = 512
    max_frame = 50
    language_path = "./data/language/"+dataset+"_embeddings.npy"
    train_list = "./data/zeroshot/"+dataset+"/split_"+split+"/seen_train_data.npy"
    train_label = "./data/zeroshot/"+dataset+"/split_"+split+"/seen_train_label.npy"
    test_list = "./data/zeroshot/"+dataset+"/split_"+split+"/unseen_data.npy"
    test_label = "./data/zeroshot/"+dataset+"/split_"+split+"/unseen_label.npy"
    ############################ sota compare ############################
    # sota_split = "5"
    # unseen_label_5 = [10,11,19,26,56]
    # unseen_label_12 = [3,5,9,12,15,40,42,47,51,56,58,59]
    # unseen_label_10 = [4,13,37,43,49,65,88,95,99,106]
    # unseen_label_24 = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]
    # sota_unseen = eval('unseen_label_'+sota_split)
    # sota_train_list = "/mnt/petrelfs/zhouyujie/zeroshot/sota/split_"+sota_split+"/train.npy"
    # sota_train_label = "/mnt/petrelfs/zhouyujie/zeroshot/sota/split_"+sota_split+"/train_label.npy"
    # sota_test_list = "/mnt/petrelfs/zhouyujie/zeroshot/sota/split_"+sota_split+"/test.npy"
    # sota_test_label = "/mnt/petrelfs/zhouyujie/zeroshot/sota/split_"+sota_split+"/test_label.npy"
# %%
