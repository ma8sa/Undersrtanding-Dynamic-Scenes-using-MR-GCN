#RCGCN START
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

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Embed_Layer(nn.Module):
    def __init__(self,input_dim,h_dim,activation=None,use_cuda=False):
        super(Embed_Layer, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.use_cuda = use_cuda
        self.embed = nn.Embedding(self.input_dim , self.h_dim)
        # self.embed = nn.Parameter(torch.Tensor(self.input_dim,self.h_dim))
        # self.embed = nn.Linear(self.input_dim-1,self.h_dim, bias=True)
        self.activation = activation 

    def forward(self, g,layer_num,h_skip,hps):
        #ids are 0/1 based on car/static-points. Hence only a 2xd matrix is the embedding matrix. 
        ids = g.ndata['id']
        # using a lookup table type-embedding
        h = self.embed(ids)

        ############### DONOT USE THIS ###################
        # using a linear layer 1xd for learning the embedding
        # ids = ids.float()
        # h = self.embed(ids.unsqueeze(1))
        ############### DONOT USE THIS ###################

        # if self.activation:
        #     h = self.activation(h)
        
        g.ndata['h'] = h 
        return (g.ndata['h'],h_skip)


#RCGCN START
class RGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,num_bases=-1,dropout=0.0,use_cuda=False,bias=None,
                 activation=None,is_input_layer=False):
        super(RGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.use_cuda = use_cuda
        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        if(dropout):
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                # print((edges.data['rel_type']).dtype , (self.in_feat) ,edges.src['id'].dtype ) 
                # print("##############################")
                # print(edges.src['id'])
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                # print(index)
#                 print(edges.data['rel_type'] , self.in_feat, edges.src['id'])
#                 print(index)
#                 print("inpt_layr",edges.data['rel_type'] * self.in_feat)
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
#             print(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}
        
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)

