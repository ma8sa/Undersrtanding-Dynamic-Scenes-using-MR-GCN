import os
import pandas as pd
import dgl
from dgl import DGLGraph
import torch
from random import shuffle
import time
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import matplotlib.pyplot as plt
from random import sample
import tqdm

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MiniGCDataset(object):

    def __init__(self, num_graphs):
        super(MiniGCDataset, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = []
        self.labels = []

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __add__(self,g,l):
        self.graphs.append(g)
        self.labels.append(l)
        
    def __getitem__(self, idx):
      
        return self.graphs[idx], self.labels[idx]

def create_g(file_path,use_cuda=False):
    # print(file_path)
    npz=np.load(file_path,allow_pickle=True)
    labels=npz['labels']
    fts_nodes=npz['fts_node']
    edge_type=npz['edge_type'].tolist()
    edge_norm=npz['edge_norm'].tolist()
    edges=npz['edges']
    # print(npz['nums'],len(labels),len(fts_nodes))
    num_nodes=len(labels)
    # print(np.dtype(fts_nodes))
    
    g = DGLGraph()
    g.add_nodes(num_nodes)

    edge_type=np.array(edge_type)
    edge_norm=np.array(edge_norm)

    for i in edges:                                              #BASIC EDGES
        g.add_edge( i[0].item() , i[1].item() )

    edge_type = torch.from_numpy(edge_type)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
    edge_type = edge_type.long()
    edge_norm = edge_norm.float()

    # fts_nodes = fts_nodes.astype(float)
    # fts_nodes = fts_nodes.astype(int)
    fts_nodes = torch.from_numpy(fts_nodes)
    fts_nodes = fts_nodes.long()

    labels = torch.from_numpy(labels)

    if(use_cuda):
        labels = labels.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        fts_nodes = fts_nodes.cuda()
        
    g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
    g.ndata['id']=fts_nodes

    return [g,labels,fts_nodes]

def seperate_seq(graphs,ratio,data_set):
    print('seperate seqs splitting')
    graphs = sorted(graphs)
    seqs_list = {}
    #createing train list and val list ; since graphs are of form seq_image.npz
    
    for i in graphs:
        seqs_list[ i.split('_')[0] ]=0
    seqs_list = [ v for v in seqs_list.keys() ]

    train_list = []
    val_list = []
    val_seqs = []

    train_seqs = sample(seqs_list,int(ratio*len(seqs_list)))

    for i in (seqs_list):
        if(i in train_seqs):
            continue
        val_seqs.append(i)
    
    # For Honda
    if(data_set=='Honda'):
        train_seqs = []
        val_seqs = [
        '0009','0010','0011','0012','0013',
        '0014','0015','0016','0017','0018',
        '0019','0020',
        
        '0025','0026','0027','0028','0029','0030','0031','0032','0033','0034','0036',
        '0000','0001','0002','0003','0004','0005','0006','0007','0008',
        '0021','0022','0023','0024'
        ]

    # for KITTI
    if(data_set=='Kitti'):
        train_seqs = []
        val_seqs = ['0004','0005','0010']

    #For Indian
    if(data_set=='Indian'):
        train_seqs = []
        val_seqs = ['0001','0002']

    # train_seqs = train_seqs[0:5]
    # val_seqs = val_seqs[0:1]

    print((train_seqs))
    print(val_seqs)
    
    for i in train_seqs:
        # print(seqs_list[i])
        for j in graphs:
            # print(j.split('_')[0])
            if((i == j.split('_')[0])):# or (i == j.split('_')[0]+'_'+j.split('_')[1] ) ):
                train_list.append(j)

    for i in val_seqs:
        for j in graphs:
            if(i == j.split('_')[0]):# or (i == j.split('_')[0]+'_'+j.split('_')[1] ) ):
                val_list.append(j)

    print("train and val list lenhgths ",len(train_list),len(val_list))

    for i in train_list:
        if(i in val_list):
            print('biscuit')

    # print(train_list,val_list)
    return [train_list,val_list]

def create_dataset(num_classes,data_set,data_path,ratio,split_meth=1,use_cuda=False):

    graphs_dir = data_path 
    graph_files=sorted(os.listdir(graphs_dir))
    ln = len(graph_files)
    total_idx = list(range(ln))

    # train_list = random.sample(list(range(ln)), int(0.7 * ln) )
    if(split_meth==1):
        [train_list,val_list] = shuffle_seq(graph_files,ratio)
    else:
        [train_list,val_list] = seperate_seq(graph_files,ratio,data_set)

    f=open('./train_idxs.txt','w')
    for j in train_list:
        f.write(j+'\n')

    f=open('./val_idxs.txt','w')
    for j in val_list:
        f.write(j+'\n')
    

    print("data split ",ratio)    
    print("break point ",None)              #corrsponding to value in function only2

    # val_list = list(range(int(ln*0.3)))
    # train_list=[]
    # for i in total_idx:
    #     if(i not in val_list):
    #         train_list.append(i)

    # print("train list ",train_list)
    # print("val list ",val_list)

    # train_list = train_list[0:10]
    # val_list = val_list[0:10]
    
    trainset = MiniGCDataset(len(train_list))
    testset = MiniGCDataset(len(val_list))
    train_edges=0
    val_idx_nodes =0
    train_idx_nodes = 0
    count_class_train = [1.0] * num_classes
    count_class_val = [1.0] * num_classes
    count_train_overall = [1.0] * num_classes
    augment_1 = []
    augment_5 = []

    shuffle(train_list)
    for i in tqdm.tqdm(train_list):
        [g_curr,l_curr,node_features]=create_g(data_path + i,use_cuda)
        test = only2(l_curr,data_path + i)
        # if(test):
        #     continue
        if(not(no_labs(l_curr,1))):
            augment_1.append(i)

        if(not(no_labs(l_curr,5))):
            augment_5.append(i)

        train_edges += g_curr.number_of_edges()
        train_idx_nodes += g_curr.number_of_nodes()
        trainset.__add__(g_curr,l_curr)
        for m in range(len(l_curr)):
                # print(l_curr[m].item())
                count_train_overall[l_curr[m].item()] += 1
                if(node_features[m].item()==0):
                    count_class_train[l_curr[m].item()] += 1

    print("intial data ",len(trainset))
    # print(len(augment_1),len(augment_3))
    
    #augmentation for class 1
    # k=0
    # while(k<1):
    #     shuffle(augment_1)
    #     for i in augment_1:
    #         # print(i)
    #         [g_curr,l_curr,node_features]=create_g( data_path + i,use_cuda)
    #         train_edges += g_curr.number_of_edges()
    #         train_idx_nodes += g_curr.number_of_nodes()
    #         trainset.__add__(g_curr,l_curr)
    #         for m in range(len(l_curr)):
    #                 count_train_overall[l_curr[m].item()] += 1
    #                 if(node_features[m].item()==0):
    #                     count_class_train[l_curr[m].item()] += 1
    #     k += 1

    # #augmentation for class 0
    # k=0
    # while(k<1):
    #     shuffle(augment_5)
    #     for i in augment_5:
    #         [g_curr,l_curr,node_features]=create_g( data_path + i,use_cuda)
    #         train_edges += g_curr.number_of_edges()
    #         train_idx_nodes += g_curr.number_of_nodes()
    #         trainset.__add__(g_curr,l_curr)
    #         for m in range(len(l_curr)):
    #                 count_train_overall[l_curr[m].item()] += 1
    #                 if(node_features[m].item()==0):
    #                     count_class_train[l_curr[m].item()] += 1
    #     k += 1

    print("updated data ",len(trainset))
    print("done trainset")
    for i in tqdm.tqdm(val_list):
        [g_curr,l_curr,node_features]=create_g(data_path + i,use_cuda)
        train_edges += g_curr.number_of_edges()
        val_idx_nodes += g_curr.number_of_nodes()
        testset.__add__(g_curr,l_curr)
        for m in range(len(l_curr)):
                if(node_features[m].item()==0):
                    count_class_val[l_curr[m].item()] += 1
    print("done valset")
    print(len(testset),"validation set count") 
    return [train_list,train_idx_nodes, trainset,testset,count_class_train,count_class_val,count_train_overall]


def collate(samples):

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labs=[]
    for i in labels:
            labs.extend(i)
    return [batched_graph, torch.tensor(labs)]

def create_batch(trainset,testset):

    # data_loader = DataLoader(trainset, batch_size=1, shuffle=True,
    #                          collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True,
                             collate_fn=collate)
    print('batch_sizes ',1,1)
    # test_X, test_Y = map(list, zip(*testset))
    # test_bg = dgl.batch(test_X)

    # labs=[]
    # for i in test_Y:
    #         labs.extend(i)
    # test_Y = torch.tensor(labs).view(-1, 1)
    # test_Y = test_Y.squeeze(dim=1)
    data_loader = []
    return [data_loader,test_loader]