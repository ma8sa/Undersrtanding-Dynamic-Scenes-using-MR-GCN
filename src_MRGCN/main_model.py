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

from src_MRGCN import RGCN_layer
from src_MRGCN.RGCN_layer import *

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# In[5]:
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
    
        self.layers = nn.ModuleList([
            RGCN(self.in_dim, self.h_dim, self.num_rels, -1,self.dropout,use_cuda,
                         activation=F.relu,is_input_layer=True),
            RGCN(self.h_dim, self.h_dim2,self.num_rels, -1 ,self.dropout,use_cuda,
                         activation=F.relu),

            RGCN(self.h_dim2, self.out_dim, self.num_rels, -1,self.dropout,use_cuda,
                         activation=None)])#partial(F.softmax, dim=1))])
        

    def forward(self, g):
        for conv in self.layers:
            conv(g)
        return g.ndata.pop('h')
