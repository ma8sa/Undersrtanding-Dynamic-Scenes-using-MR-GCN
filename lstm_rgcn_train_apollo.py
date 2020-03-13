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


from src_lstm import graphs_preproc
from src_lstm.graphs_preproc import *

from src_lstm import main_model_train
from src_lstm.main_model_train import *

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    data_dir = '../../lstm_graphs_apollo/'
    use_cuda = 1
    if use_cuda:
            torch.cuda.set_device(0)

    ####################################################################
    #parameters for creating model 
    num_classes = 6
    n_hidden_layers = 1
    n_hidden = 128          #rgcn 1st layer dimension
    h_dim2 = 64             #rgcn 2nd layer dimension
    h_dim3 = 32             # FOR LSTM o/p dimension
    layers_lstm = 1
    dropout = 0.0           #dropout in rgcn
    dropout_lstm = 0.0
    
    num_node_fts = 2        #vehicle /static-points like lanes/poles; 0->vehicle, 1->lanes
    num_rels = 5            #relations are 4(top-left,bottom-left,top-right,bottom-right and self-edge)
    n_bases = -1
    ratio = 0.7

    print("data split",ratio)
    [trainsets,valsets,train_idx_nodes,count_class_train,count_class_val,count_overall_train] = create_data(num_classes,ratio,data_dir,2,use_cuda)
 
    #lenght of time steps for each sequence
    time_stamps = 10
    n_epochs = 1
    
    class_acc = [0.0]*num_classes
    model = main_model(num_node_fts,n_hidden,num_classes,num_rels,h_dim2,dropout,n_bases,n_hidden_layers,h_dim3,layers_lstm,dropout_lstm,use_cuda=use_cuda,bidirectional=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    whts=[]
    for i in range(num_classes):
        whts.append( ((train_idx_nodes-count_overall_train[i])/train_idx_nodes))
    whts = torch.from_numpy(np.array(whts)).float()
    
    if(use_cuda):
        model = model.cuda()
        whts = whts.cuda()

    print("created model")
    for param in model.parameters():
        print(param.shape)
        # torch.nn.init.xavier_uniform(param)

    print("----------------------------------train and val split--------------------------------")
    print("train class counts ",count_class_train)
    print("val class counts ",count_class_val)
    
    
    print('\n\n')
    for epoch in range(n_epochs):
        st=time.time()
        train_crct = 0.0
        train_cntr = 1.0
        val_crct = 0.0
        val_cntr = 0.0
        train_loss = 0
        val_loss = 0
        val_loss_schdlr = 0
        
        train_class_crcts =  [0.0]*num_classes
        train_class_cnts =  [1.0]*num_classes
        val_class_crcts =  [0.0]*num_classes
        val_class_cnts =  [1.0]*num_classes
        skipped = 0
        skipped_val = 0

        tendency = [[0.0] * num_classes]*num_classes
        tendency = np.array(tendency)

        shuffle(trainsets)
        model.train()
        ##### TRAINING
        batch_size_train = 1
        for j in tqdm.tqdm(range(0,len(trainsets),batch_size_train)):
            optimizer.zero_grad()
            loss2,logits6,labels3,nodefts,skipped,skipped_val = model.forward_rgcn(trainsets,time_stamps,batch_size_train,j,whts,skipped,skipped_val,0)
            
            loss2.backward()
            optimizer.step()

            loss = loss2.to('cpu')
            del(loss2)
            train_loss += loss.item()
            logits7 = logits6.to('cpu')
            del(logits6)
            results = (logits7.argmax(dim=1))
            labels = labels3.to('cpu')
            del(labels3)
            torch.cuda.empty_cache()
        
            #For overall accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    train_crct += 1
                    train_cntr += 1
                elif(nodefts[h].item() == 0):
                    train_cntr += 1

            #For class accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    train_class_crcts[labels[h].item()] += 1
                    train_class_cnts[labels[h].item()] += 1
                elif(nodefts[h].item() == 0):
                    train_class_cnts[labels[h].item()] += 1

     
        torch.cuda.empty_cache()    
        train_acc = train_crct / train_cntr
        
        #VALIDATION
        model.eval()
        batch_size_val = 1
        for j in tqdm.tqdm(range(0,len(valsets),batch_size_val)):
            loss2,logits6,labels3,nodefts,skipped,skipped_val = model.forward_rgcn(valsets,time_stamps,batch_size_val,j,whts,skipped,skipped_val,1)
            loss = loss2.to('cpu')
            del(loss2)
            val_loss += loss.item()
            val_loss_schdlr += loss.item()

            logits7 = logits6.to('cpu')
            del(logits6)

            results = (logits7.argmax(dim=1))
            labels = labels3.to('cpu')
            del(labels3)

            torch.cuda.empty_cache()    
            
            #For overall accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    val_crct += 1
                    val_cntr += 1
                elif(nodefts[h].item() == 0):
                    val_cntr += 1

            #For class accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    val_class_crcts[labels[h].item()] += 1
                    val_class_cnts[labels[h].item()] += 1
                elif(nodefts[h].item() == 0):
                    val_class_cnts[labels[h].item()] += 1

        torch.cuda.empty_cache()

        val_acc = val_crct / val_cntr
        
        print("Epoch {:05d} | ".format(epoch) +
              "Train Accuracy: {:.4f} | Train Loss: {:.4f} | Val acc: {:.4f} | Val Loss: {:.4f} |".format(
                  train_acc, train_loss, val_acc, val_loss))
        print('----------------------------------------------------------------------------------------')

        train_class_crcts =np.array(train_class_crcts)
        train_class_cnts =np.array(train_class_cnts)

        val_class_crcts =np.array(val_class_crcts)
        val_class_cnts =np.array(val_class_cnts)

        train_cl_ac = train_class_crcts/train_class_cnts
        val_cl_ac   = val_class_crcts/val_class_cnts
        
        print("cl_ac",train_cl_ac,val_cl_ac)
        print(train_class_cnts,val_class_cnts)
        end=time.time()
        print("epoch time=",end-st)

    ########################## END OF CODE ##########################