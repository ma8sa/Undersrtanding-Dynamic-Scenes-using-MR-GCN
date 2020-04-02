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

# seed = 0
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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
class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,skip_dim,skip=False,dropout=0.0,num_bases=-1,use_cuda=False, bias=None,
                 activation=None, is_input_layer=False,gated=False,Fusion=False,attention=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.use_cuda = use_cuda
        self.gated=gated
        self.skip_dim = skip_dim
        self.skip = skip
        self.Fusion = Fusion
        self.attention = attention
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

        if self.gated:
            self.bias_gate = nn.Parameter(torch.Tensor(1))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))
        if self.gated:
            if self.is_input_layer:
                self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels,1, 1))
            else:
                self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels,self.in_feat, 1))
            
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))
    
        if self.attention:
            if(out_feat == 6):
                heads = 2
            else:
                heads = 2
            self.transformer = nn.TransformerEncoderLayer(out_feat,heads,int(out_feat/float(heads)) )
            
            self.attention_matrix1 = nn.Parameter(torch.Tensor(self.out_feat,1))
            nn.init.xavier_uniform_(self.attention_matrix1,gain=nn.init.calculate_gain('sigmoid'))
            
            self.attention_matrix2 = nn.Parameter(torch.Tensor(self.out_feat,1))
            nn.init.xavier_uniform_(self.attention_matrix2,gain=nn.init.calculate_gain('sigmoid'))
            
            self.attention_concat=nn.Linear(2*self.out_feat,self.out_feat)

            self.multihead_attn = nn.MultiheadAttention(self.out_feat, 1)
            
            if(self.use_cuda):
                self.transformer.cuda()

        self.skip_weight = nn.Linear(self.skip_dim,self.out_feat, bias=True)
        # self.w_skip = nn.Parameter(torch.Tensor(self.out_feat+32,self.out_feat))

        if self.Fusion:
            self.Fusion_dim1 = 64
            self.Fusion_dim2 = 32
            self.Fusion_dim3 = 32
            self.Fusion_weights = []
            self.Fusion_weight1 = nn.Linear(self.Fusion_dim1,self.out_feat, bias=True)
            self.Fusion_weight2 = nn.Linear(self.Fusion_dim2,self.out_feat, bias=True)
            self.Fusion_weight3 = nn.Linear(self.Fusion_dim3,self.out_feat, bias=True)
            self.Fusion_weights.append(self.Fusion_weight1)
            self.Fusion_weights.append(self.Fusion_weight2)
            self.Fusion_weights.append(self.Fusion_weight3)

    def forward(self, g,layer_num,h_skip,hps):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.gated:
            gate_weight = self.gate_weight

        self.node_h = g.ndata['h']
        num_nodes = g.number_of_nodes()
        
        adj = torch.zeros(size=(6,num_nodes, num_nodes)).cuda()
        normalizations = torch.zeros(size=(6,num_nodes, num_nodes)).cuda()
        edges_data = g.all_edges(order='eid')
        edges_rels = g.edata['rel_type']
        matrix_repr = [edges_rels, edges_data[0],edges_data[1]]
        adj[matrix_repr]=1.0 
        normalizations[matrix_repr] = 1.0

        hrs = []
        hrd = []
        
        # now each row v contains column u as 1, iff edge exists from u->v, After this transpose
        for r in range(6):
            adj[r] = adj[r].t()
            normalizations[r] = normalizations[r].t()

        #For GENERAL normalization
        added_norms = torch.sum(normalizations,dim=0)
        added_norms_full = (1/(1+torch.sum(added_norms,dim=1))).unsqueeze(1)

        # for relational normalization for gating
        added_norms_gate = (torch.sum(normalizations,dim=2)).unsqueeze(2)
        if(not(self.use_cuda)):
            added_norms_gate.cpu()
        # now in r x n

        adj_norm = []
        adj_norm_rel = []

        for r in range(6):
            adj_norm.append( adj[r] * added_norms_full ) 
            # print( (1/(1+added_norms_gate[r])).shape )
            adj_norm_rel.append( adj[r] * 1/(1+added_norms_gate[r]) ) 
            
        for r in range(6):
            # print(weight[r].get_device(),adj_norm_rel[r].get_device())
            if(not(self.use_cuda)):
                weight[r].cpu()
                adj_norm_rel[r] = adj_norm_rel[r].cpu()    
            # print(weight[r].get_device(),adj_norm_rel[r].get_device())

        if(not(self.use_cuda)):
            self.node_h.cpu()

        # print(weight[0].get_device(),adj_norm_rel[0].get_device(),self.node_h.get_device())
        for r in range(6):
            hrd.append(torch.matmul(self.node_h,weight[r]))
            hrs.append(torch.matmul(adj_norm_rel[r],torch.matmul(self.node_h,weight[r]) ) )

        gates = []
        if self.gated:
            for r in range(6):

                gates.append( torch.matmul( adj_norm_rel[r] * ( torch.sigmoid( torch.matmul(self.node_h,self.gate_weight[r]) + self.bias_gate) ).t()   , hrd[r] ) )  


        op = torch.stack(hrs,dim=1)
        gated_op = torch.stack(gates,dim=1)

        attentions_head1 = self.attention_matrix1.unsqueeze(0).repeat(num_nodes,1,1)
        # attention_maps_head1 = torch.bmm(op,attentions_head1)
        attention_maps_head1 = torch.bmm(op,attentions_head1)
        # now in n x r x 1
        attention_scores_1 = torch.softmax(attention_maps_head1,dim=1)
        attention_op_1 = attention_scores_1.repeat(1,1,self.out_feat)
        # now in n x r x d

        attentions_head2 = self.attention_matrix2.unsqueeze(0).repeat(num_nodes,1,1)
        # attention_maps_head2 = torch.bmm(op,attentions_head2)
        attention_maps_head2 = torch.bmm(op,attentions_head2)
        # now in n x r x 1
        attention_scores_2 = torch.softmax(attention_maps_head2,dim=1)
        attention_op_2 = attention_scores_2.repeat(1,1,self.out_feat)
        # now in n x r x d

        h_final51 = op * attention_op_1
        h_final52 = op * attention_op_2
        # here op is n x r x d, the original input to attention and output of rgcn
        # now in n x r x d

        #for having like multi head transformer layer, ie; concatentation
        h_combined = torch.cat((h_final51,h_final52),dim=2)
        h_final5 = self.attention_concat(h_combined)

        # print(h_final5.shape)
        # Now in n x r x d 

        # h_final6 = h_final5 + gated_op
        h_final6 = h_final5
        h_final6 = torch.sum(h_final6,1)
        # now in n x d
        # print(h_final6.shape)

        h = h_final6
        # print(self.activation)
        if self.bias:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)

        #skip-connections from i-2 th layer to the current layer                        
        if(self.skip):
            # print(hps[layer_num-1]['h']-h_skip)
            h = h + self.skip_weight(h_skip)
            
            if self.activation:
                h = self.activation(h)

        if self.Fusion:
            for i in range(len(hps)-1,-1,-1):
                h += (self.Fusion_weights[i])(hps[i]['h']) 

            if self.activation:
                h = self.activation(h)
        # print(h.shape)
        g.ndata['h'] = h

        return (g.ndata['h'],weight)
