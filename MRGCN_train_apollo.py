import dgl
import torch.optim as optim
from dgl import DGLGraph
from functools import partial
import os
import random
from random import shuffle
import time
import copy 
from random import sample
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
import numpy as np
import tqdm

from src_MRGCN import graphs_preproc
from src_MRGCN.graphs_preproc import *

from src_MRGCN import main_model
from src_MRGCN.main_model import *

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = './graphs_apollo/'

######################### mian part of the function #######################
use_cuda = 0
if use_cuda:
        torch.cuda.set_device(0)

dropout = 0.0
# print('-------------DROPOUT = ', dropout,'------------------')
num_classes=6
num_node_fts=2

num_hidden_dim=64
h_dim2 = 32
h_dim3 = 32
h_dim4 = 32
tot_time = 10

num_rels=6
num_epochs=100
models = []
avg_acc_train=0
avg_acc_val=0
best_acc = 0
split_ratio = 0.7
print("train and val split...... creating val and train seperate sequences")

# [train_idx , train_idx_nodes ,trainset , testset,count_class_train,count_class_val,count_train_overall]= create_dataset(num_classes,data_dir,use_cuda)
print("total epochs = ",num_epochs)
for epoch in range(num_epochs):
    if(epoch%250 == 0):
        if(epoch > 0):
            print("###################################")
            print(avg_acc_train/250,avg_acc_val/250,best_acc)
            # checkpoint = {'model': best_model,
            #           'state_dict': best_model.state_dict(),
            #           'optimizer' : optimizer.state_dict()}
            # torch.save(checkpoint, wts_dir+'best_model_'+str(epoch)+'_'+str(best_acc)+'_'+str(class_acc[0])+'_'+str(class_acc[1])+'_'+str(class_acc[2])+'.pth')
        avg_acc_train=0
        avg_acc_val=0
        best_acc = 0
        class_acc = [0.0]*num_classes
        model = Classifier(num_node_fts,num_hidden_dim,num_classes,num_rels,h_dim2,h_dim3,h_dim4,dropout,use_cuda)

        for param in model.parameters():
            print(param.shape)        
        print("created new model")
        if(epoch==0):
            [train_idx , train_idx_nodes ,trainset , testset,count_class_train,count_class_val,count_train_overall]= create_dataset(num_classes,data_dir,split_ratio,2,use_cuda)
        
        [data_loader,test_loader] = create_batch(trainset,testset)
        print("created data")
        whts=[]
        for i in range(num_classes):
        	whts.append( ((train_idx_nodes-count_train_overall[i])/train_idx_nodes))

        print('\n')
        print("train_class_counts :",count_class_train)
        print("val_class_counts :",count_class_val)
        print('\n')
    
        whts = torch.from_numpy(np.array(whts))
        whts = whts.float()
        print("original whts ",whts)

        optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.00001,patience=8,verbose =True)       
        if(use_cuda):
            whts = whts.cuda()
            model.cuda()

        print("Using weights on cuda")
        loss_func = nn.CrossEntropyLoss(weight=whts)
        # loss_func = nn.CrossEntropyLoss()

    
    model.train()
    st=time.time()

    epoch_loss = 0

    cf_matrix_labels_train = []
    cf_matrix_pred_train = []

    cf_matrix_labels_val = []
    cf_matrix_pred_val = []

    train_crct = 0.0
    train_cntr = 0.0
    val_crct = 0.0
    val_cntr = 0.0
    train_loss = 0
    val_loss = 0
    
    train_class_crcts =  [0.0]*num_classes
    train_class_cnts =  [1.0]*num_classes
    val_class_crcts =  [0.0]*num_classes
    val_class_cnts =  [1.0]*num_classes

    tendency = [[0.0] * num_classes]*num_classes
    tendency = np.array(tendency)
    counter = 0
    st = time.time()
    
    y_true_precision = []
    y_pred_precision = []

    for iter, (bg, label) in enumerate(data_loader):
        nodes_fts = []
        curr_nodes2 = bg.ndata['id']
        curr_nodes = curr_nodes2.to('cpu')
        
        del(curr_nodes2)

        nodes_fts.extend(curr_nodes)
        prediction = model(bg)
        
        if(use_cuda):
            label=label.cuda()
        loss2 = loss_func(prediction,label)
        
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        loss = loss2.to('cpu')
        del(loss2)
        epoch_loss += loss.detach().item()

        label2=label.to('cpu')
        prediction2=prediction.to('cpu')
        del(label)
        del(prediction)
        torch.cuda.empty_cache()

        results = prediction2.argmax(dim=1)
        
        #For total accuracies
        for k in range(len(results)):
            if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                train_crct += 1
                train_cntr += 1
            elif(nodes_fts[k].item() == 0):
                train_cntr += 1

        #For class accuracies
        for k in range(len(results)):
            if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                train_class_crcts[label2[k].item()] += 1
                train_class_cnts[label2[k].item()] += 1
            elif(nodes_fts[k].item() == 0):
                train_class_cnts[label2[k].item()] += 1


    # print(iter,epoch_loss)
    epoch_loss /= (iter + 1)
    train_acc = train_crct / train_cntr
    avg_acc_train += train_acc

    #FOR VALIDATION PART
    model.eval()
    val_loss_schdlr = (torch.tensor(0.0)).to('cpu')
    for iter, (bg, label) in enumerate(test_loader):
        nodes_fts = []
        curr_nodes2 = bg.ndata['id']
        curr_nodes = curr_nodes2.to('cpu')
        
        del(curr_nodes2)
        nodes_fts.extend(curr_nodes)

        prediction = model(bg)
        if(use_cuda):
            label=label.cuda()

        loss2 = loss_func(prediction, label)
        loss = loss2.to('cpu')
        
        del(loss2)
        val_loss += loss.item()
        val_loss_schdlr += loss.item()
        label2=label.to('cpu')
        prediction2=prediction.to('cpu')
        del(label)
        del(prediction)

        results = prediction2.argmax(dim=1)

        # #For total accuracies
        for k in range(len(results)):
    
            if(nodes_fts[k].item() == 0):
                cf_matrix_labels_val.append(label2[k])
                cf_matrix_pred_val.append(results[k])
                # print(label2[k].item(),results[k].item())
                y_true_precision.append(label2[k].item())
                y_pred_precision.append(results[k].item())

            if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                val_crct += 1
                val_cntr += 1
            elif(nodes_fts[k].item() == 0):
                val_cntr += 1

        #For class accuracies
        for k in range(len(results)):
            if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                val_class_crcts[label2[k].item()] += 1
                val_class_cnts[label2[k].item()] += 1
            elif(nodes_fts[k].item() == 0):
                val_class_cnts[label2[k].item()] += 1

        torch.cuda.empty_cache()
    
    scheduler.step(val_loss_schdlr)

    val_loss /= (iter+1)
    val_acc = val_crct / val_cntr
    avg_acc_val += val_acc

    print('---------------------------------------------------------------------------------------------')
    print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} |  Validation Accuracy: {:.4f} | Validation Loss: {:.4f} ".format(train_acc, epoch_loss,val_acc,val_loss))

    train_class_crcts =np.array(train_class_crcts)
    train_class_cnts =np.array(train_class_cnts)

    val_class_crcts =np.array(val_class_crcts)
    val_class_cnts =np.array(val_class_cnts)

    train_cl_ac = train_class_crcts/train_class_cnts
    val_cl_ac   = val_class_crcts/val_class_cnts
    print("calss accrs",train_cl_ac,val_cl_ac)

    print(train_class_cnts,val_class_cnts)
    print('---------------------------------------------------------------------------------------------')

    end=time.time()
    print("epoch time = ",end-st)
    #IMPRTANT TO PT IT BACK ON GPU
    if(use_cuda):
        model.cuda()
