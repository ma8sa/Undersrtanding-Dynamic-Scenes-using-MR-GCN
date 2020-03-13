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

from src_lstm import rgcn_layer
from src_lstm.rgcn_layer import *

time_stamps = 10
class main_model(nn.Module):
    def __init__(self,num_node_fts,n_hidden,num_classes,num_rels,h_dim2,dropout,n_bases,n_hidden_layers,h_dim3,layers_lstm,dropout_lstm,use_cuda=False,bidirectional=True):
        super(main_model, self).__init__()        
        self.num_node_fts = num_node_fts
        self.n_hidden=n_hidden
        self.num_classes=num_classes
        self.num_rels=num_rels
        self.h_dim2=h_dim2
        self.dropout=dropout
        self.n_bases=n_bases
        self.n_hidden_layers=n_hidden_layers
        self.h_dim3=h_dim3
        self.layers_lstm=layers_lstm
        self.dropout_lstm=dropout_lstm
        self.use_cuda=use_cuda
        self.bidirectional=bidirectional
        
        #Definfning each layer with given parameters
        self.rgcn = Model(self.num_node_fts,
                      self.n_hidden,
                      self.num_classes,
                      self.num_rels,self.h_dim2,
                      self.dropout,
                      num_bases=self.n_bases,
                      num_hidden_layers=self.n_hidden_layers,use_cuda=False)
        self.lstm = torch.nn.LSTM(input_size = self.h_dim2, hidden_size = self.h_dim3,num_layers = self.layers_lstm ,dropout = self.dropout_lstm ,bidirectional = self.bidirectional,batch_first=True)
        self.lin_layer=nn.Linear(2*h_dim3,2*h_dim3)
        self.transformer = nn.TransformerEncoderLayer(2*h_dim3,16,1024)
        self.pool = torch.nn.AvgPool1d(time_stamps)
        self.final_layer=nn.Linear(2*h_dim3,num_classes)
        self.final_act = F.relu
        self.norm1 = nn.LayerNorm([time_stamps,2*h_dim3])
        self.norm2 = nn.LayerNorm(2*h_dim3)

        if(self.use_cuda):
            self.rgcn.cuda()
            self.lstm.cuda()
            self.final_layer.cuda()
            self.lin_layer.cuda()
            self.pool.cuda()
            self.norm1.cuda() 
            self.norm2.cuda() 
            self.transformer.cuda()

    def forward_rgcn(self,trainsets,time_stamps,batch_size,j,skipped,skipped_val,flag):
        k=j
        labels2 = []
        logits5 = []
        nodefts = []
        while(k<j+batch_size):
            check = 0
            if(k>=len(trainsets)):
                break
            trainset = trainsets[k]
            gcn_ops=[]
            num_nodes = (trainset.graphs[0]).number_of_nodes()
            
            for t in range(time_stamps):    
                g = trainset.graphs[t]
                nodes_fts = g.ndata['id'].to('cpu')
                labs = trainset.labels[t]
                logits = self.rgcn(g)
                gcn_ops.append(logits)

            memory = torch.cat(gcn_ops ,dim=1)
            memory2 = torch.reshape(memory, (num_nodes,time_stamps,self.h_dim2))

            # memory2 = F.dropout(memory2,p=0.25)
            logits2,hidden_op = self.lstm(memory2)
            logits2 = self.norm1(logits2)
            logits2 = F.relu(logits2)

            #to get into t x N x d format since transformers take that input
            logits2 = logits2.permute(1,0,2)
            logits3 = self.transformer(logits2)
  
            # Now in t x n x d format
            ####MAX pooling over time dimension
            logits3 = logits3.permute(1,0,2)
            # Now in n x t x d
            logits3 = logits3.permute(0,2,1)
            # Now in n x d x t
            logits3 = self.pool(logits3)
            logits3 = logits3.squeeze(dim=2)

            # Now in n x d
            logitsn = self.final_layer(logits3)
            logits4 = logitsn
            
            logits5.extend(logits4)
            labels2.extend(labs)
            nodefts.extend(nodes_fts)

            k += 1
        logits6 = torch.stack(logits5,dim=0)
        labels3 = torch.stack(labels2,dim=0)
        
        loss_func = nn.CrossEntropyLoss()
        loss2 = loss_func(logits6,labels3)
        return [loss2,logits6,labels3,nodefts,skipped,skipped_val]
