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

from src_rel import RGCN_layer
from src_rel.RGCN_layer import *

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, num_rels,h_dim2,h_dim3,h_dim4,dropout,use_cuda=False):
        super(Classifier, self).__init__()
        
        self.in_dim = in_dim
        self.h_dim = hidden_dim
        self.h_dim2 = h_dim2
        self.h_dim3 = h_dim3
        self.h_dim4 = h_dim4
        self.out_dim = n_classes
        self.num_rels = num_rels
        self.dropout = dropout
        self.skip = True
        self.k = 1             # layer from which skip_connection needs to be added
        self.Fusion = False
        self.dims_list = []             #to have list of dimesnions at each layer
        self.use_cuda = use_cuda
        self.dims_list.extend([1,self.h_dim,self.h_dim2,self.h_dim3,self.h_dim4])
        self.num_bases = -1
        self.gated = True
        self.attention = True
        
        self.final_layer=nn.Linear(self.h_dim4,self.out_dim)
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList([
            self.build_embedding_layer(self.in_dim,self.h_dim),
            self.build_hidden_layer(1,self.h_dim,self.h_dim2,activation=F.relu),
            self.build_hidden_layer(2,self.h_dim2,self.h_dim3,activation=F.relu),
            self.build_hidden_layer(3,self.h_dim3,self.h_dim4,Fusion=self.Fusion,activation=None)
            ])

    def build_embedding_layer(self,dim_1,dim_2):
        return Embed_Layer(dim_1, dim_2,activation=F.relu, use_cuda=self.use_cuda)

    def build_hidden_layer(self,num_curr_layers,dim_1,dim_2,Fusion=False,activation=F.relu):
        if(num_curr_layers>=self.k+1):
            return RGCNLayer(dim_1, dim_2, self.num_rels,self.dims_list[num_curr_layers-self.k+1],self.skip,self.dropout,self.num_bases,use_cuda=self.use_cuda,
                         activation=activation,gated=self.gated,Fusion=Fusion,attention=self.attention)
        else:
            return RGCNLayer(dim_1, dim_2, self.num_rels,1,False,self.dropout,self.num_bases,use_cuda=self.use_cuda,
                         activation=activation,gated=self.gated,Fusion=Fusion,attention=self.attention)

    def forward(self, g):
        self.hps = []
        for i in range(len(self.layers)):
            conv = self.layers[i]
            if(i>1):
                h,w = conv(g,i,self.hps[i-self.k]['h'],self.hps)
            else:
                h,w = conv(g,i,np.ones(g.number_of_nodes()),self.hps)

            self.hps.append({'h':h,'w':w})

        return g.ndata.pop('h')