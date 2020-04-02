import os
import dgl
from dgl import DGLGraph
import torch
from random import shuffle
import time
import os
import sys
import random
import numpy as np
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
from random import sample
import tqdm

# seed = 0
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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

def seperate_seq(graphs,ratio):
    # print('seperate seqs splitting')
    graphs = sorted(graphs)
    print("total graphs = ",len(graphs))
    
    train_list = []
    val_list = []
    val_seqs = []

    train_seqs = ['0169', '0056','0104', '0068',
                '0157','0138', '0022' ,'0118', '0110','0164','0026', '0124','0040', '0004', '0168','0123', '0050', '0018', '0137', '0116', '0149', '0115', '0139', '0061', '0141', '0014', '0016', '0146', '0003', '0156', '0034', '0070', '0023', '0002', '0031', '0010', '0043', '0015', '0038', '0055', '0134', '0142', '0037', '0148', '0064', '0006', '0111', '0041', '0125'
                '0025', '0107', '0045', '0028', '0011', '0013', '0102', '0069', '0143', '0170', '0042', '0032',
                'a_0002', 'a_0011', 'a_0014', 'a_0015',  'a_0051', 'a_0056', 'a_0057', 'a_0064', 'a_0069', 'a_0070', 'a_0102', 'a_0103', 'a_0111', 'a_0114', 'a_0115', 'a_0116', 'a_0124', 'a_0127', 'a_0137', 'a_0141', 'a_0143', 'a_0146', 'a_0148', 'a_0149', 'a_0151', 'a_0156', 'a_0157', 'a_0164', 'a_0168', 'a_0169', 'a_0170', 'a_0171','200','201','202','203','204','206','207','208','209',
                'a_0016', 'a_0024', 'a_0027', 'a_0037', 'a_0041', 'a_0042', 'a_0046', 'a_0048', 'a_0049',
                '300','301','302','303','304','306','307','308','309'
                ]

    val_seqs = ['0114', '0101', '0046', '0033', '0103', '0171', '0024', '0007', '0106', '0027', '0035', '0030', '0001', '0051', '0151', '0127', '0131', '0039', '0049', '0145', '0048', '0135', '0128',
             '0017', '0057',  '0029', '0130'         
            ]

    # print((train_seqs))
    # print(val_seqs)
    
    for i in train_seqs:
        for j in graphs:
            if((i == j.split('_')[0]) or (i == j.split('_')[0]+'_'+j.split('_')[1] ) ):
                train_list.append(j)

    for i in val_seqs:
        for j in graphs:
            if(i == j.split('_')[0] or (i == j.split('_')[0]+'_'+j.split('_')[1] ) ):
                val_list.append(j)

    print("train and val list lenhgths ",len(train_list),len(val_list))

    for i in train_list:
        if(i in val_list):
            print(i,"biscuit")

    # print(train_list,val_list)
    return [train_list,val_list]

def create_dataset(num_classes,data_path,ratio,split_meth=1,use_cuda=False):

    graphs_dir = data_path 
    graph_files=sorted(os.listdir(graphs_dir))
    ln = len(graph_files)
    total_idx = list(range(ln))

    [train_list,val_list] = seperate_seq(graph_files,ratio)

    f=open('./train_idxs.txt','w')
    for j in train_list:
        f.write(j+'\n')

    f=open('./val_idxs.txt','w')
    for j in val_list:
        f.write(j+'\n')


    # train_list = train_list[0:20]    
    # val_list = val_list[0:20]    

    trainset = MiniGCDataset(len(train_list))
    testset = MiniGCDataset(len(val_list))
    train_edges=0
    val_idx_nodes =0
    train_idx_nodes = 0
    count_class_train = [1.0] * num_classes
    count_class_val = [1.0] * num_classes
    count_train_overall = [1.0] * num_classes
    augment_1 = []
    augment_3 = []

    shuffle(train_list)
    for i in tqdm.tqdm(train_list):
        [g_curr,l_curr,node_features]=create_g(data_path + i,use_cuda)
        
        train_edges += g_curr.number_of_edges()
        train_idx_nodes += g_curr.number_of_nodes()
        trainset.__add__(g_curr,l_curr)
        for m in range(len(l_curr)):
                # print(l_curr[m].item())
                count_train_overall[l_curr[m].item()] += 1
                if(node_features[m].item()==0):
                    count_class_train[l_curr[m].item()] += 1

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
    # The input `samples` is a list of pairs
    #  (graph, label).

    #     print((samples))
    
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labs=[]
    for i in labels:
            labs.extend(i)
    return [batched_graph, torch.tensor(labs)]

def create_batch(trainset,testset):

    data_loader = DataLoader(trainset, batch_size=1, shuffle=True,
                             collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True,
                             collate_fn=collate)
    print('batch_sizes ',1,1)
    
    return [data_loader,test_loader]
