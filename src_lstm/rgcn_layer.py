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

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,dropout=0.0,num_bases=-1,use_cuda=False, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.use_cuda = use_cuda
        # self.dropout = dropout
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
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
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
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


# MY MODULE for MODEL and its parameters
class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,h_dim2,dropout,
                 num_bases=-1,  num_hidden_layers=1,use_cuda=False):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.h_dim2 = h_dim2
        self.memory = []
        self.dropout = dropout
        # create rgcn layers
        self.build_model()
        

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers-1):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        self.layers.append(self.build_hidden_layer_final())

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.dropout ,self.num_bases,use_cuda=False,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels,self.dropout,self.num_bases,use_cuda=False,
                         activation=F.relu)
    def build_hidden_layer_final(self):
        return RGCNLayer(self.h_dim, self.h_dim2, self.num_rels,self.dropout ,self.num_bases,use_cuda=False,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim2, self.out_dim, self.num_rels,self.dropout,self.num_bases,use_cuda=False,
                         activation=partial(F.softmax, dim=1))

    def create_lstm_layers(self):
            lstm = LSTM(self.h_dim, self.out_dim)
            self.memory.append(lstm)

    def forward(self, g):
        for conv in self.layers:
            conv(g)
        return g.ndata.pop('h')
        
