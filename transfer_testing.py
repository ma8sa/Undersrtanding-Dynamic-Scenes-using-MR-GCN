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

from src_rel import graphs_preproc_transfer
from src_rel.graphs_preproc_transfer import *

from src_rel import main_model
from src_rel.main_model import *

#seed = 0
#random.seed(seed)
#torch.manual_seed(seed)
#np.random.seed(seed)
#torch.cuda.manual_seed_all(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

data_set = sys.argv[1]
if(data_set=='Honda'):
	data_dir = './graphs_honda/'
if(data_set=='Kitti'):
	data_dir = './graphs_kitti/'
if(data_set=='Indian'):
	data_dir = './graphs_indian/'


######################### mian part of the function #######################
use_cuda = 1
if use_cuda:
        torch.cuda.set_device(0)

dropout = 0.0
# print('-------------DROPOUT = ', dropout,'------------------')
num_classes=6
num_node_fts=2

num_hidden_dim=64
h_dim2 = 32
h_dim3 = 32
h_dim4 = 6
tot_time = 10

num_rels=6
num_epochs=1
models = []
avg_acc_train=0
avg_acc_val=0
best_acc = 0
split_ratio = 0.7
print("train and val split...... creating val and train seperate sequences")

print("total epochs = ",num_epochs)
for epoch in range(num_epochs):
    if(epoch%250 == 0):
        if(epoch > 0):
            print("###################################")
            print(avg_acc_train/250,avg_acc_val/250,best_acc)

        avg_acc_train=0
        avg_acc_val=0
        best_acc = 0
        class_acc = [0.0]*num_classes
        model = Classifier(num_node_fts,num_hidden_dim,num_classes,num_rels,h_dim2,h_dim3,h_dim4,dropout,use_cuda)
        model_path = './rel-att-gcn.pth'
        model.load_state_dict(torch.load(model_path,map_location='cpu')['state_dict'])


        if(use_cuda):
            model.cuda()
        for param in model.parameters():
            print(param.shape)        
        if(epoch==0):
            [train_idx , train_idx_nodes ,trainset , testset,count_class_train,count_class_val,count_train_overall]= create_dataset(num_classes,data_set,data_dir,split_ratio,2,use_cuda)
        

        [data_loader,test_loader] = create_batch(trainset,testset)
        print("created data")

        print('\n')
        print("calss counts :",count_class_val)
        print('\n')
    

        optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.0001,patience=5,verbose =True)       
        if(use_cuda):
            model.cuda()
        print("Using weights on cuda")
        #loss_func = nn.CrossEntropyLoss(weight=whts)
        loss_func = nn.CrossEntropyLoss()

    
    model.train()
    st=time.time()

    epoch_loss = 0
    
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

    cf_matrix_labels_train = []
    cf_matrix_pred_train = []

    cf_matrix_labels_val = []
    cf_matrix_pred_val = []

    tendency = [[0.0] * num_classes]*num_classes
    tendency = np.array(tendency)
    counter = 0
    st = time.time()
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

        # optimizer.zero_grad()
        # loss2.backward()
        # optimizer.step()

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


    #FOR VALIDATION PART
    # model.to('cpu')
    model.eval()
    val_loss_schdlr = (torch.tensor(0.0)).to('cpu')
    for iter, (bg, label) in enumerate(test_loader):
        nodes_fts = []
        curr_nodes2 = bg.ndata['id']
        curr_nodes = curr_nodes2.to('cpu')

        if(len(curr_nodes)<=2):
                continue
        
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

        # train_list = list( range(len(prediction2)) )
        results = prediction2.argmax(dim=1)
        
        
        if(data_set=='Honda'):
            #For total accuracies
            for k in range(len(results)):

                if(label2[k].item()==5):
                    continue          

                if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                    val_crct += 1
                    val_cntr += 1
                elif(nodes_fts[k].item() == 0):
                    val_cntr += 1

            #For class accuracies
            for k in range(len(results)):

                if(label2[k].item()==5):
                    continue
                if(nodes_fts[k].item() == 0):
                    cf_matrix_labels_val.append(label2[k])
                    cf_matrix_pred_val.append(results[k])

                if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                    val_class_crcts[label2[k].item()] += 1
                    val_class_cnts[label2[k].item()] += 1
                elif(nodes_fts[k].item() == 0):
                    val_class_cnts[label2[k].item()] += 1

        else:
            #For total accuracies
            for k in range(len(results)):
                # neglecting complex classes like lane change and overtake
                if(results[k].item()>2):
                    results[k]=0

                if(results[k].item()==label2[k].item() and nodes_fts[k].item() == 0 ):
                    val_crct += 1
                    val_cntr += 1
                elif(nodes_fts[k].item() == 0):
                    val_cntr += 1

            #For class accuracies
            for k in range(len(results)):
                
                if(nodes_fts[k].item() == 0):
                    cf_matrix_labels_val.append(label2[k])
                    cf_matrix_pred_val.append(results[k])

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
          " Validation Accuracy: {:.4f} | Validation Loss: {:.4f} ".format(val_acc,val_loss))

    train_class_crcts =np.array(train_class_crcts)
    train_class_cnts =np.array(train_class_cnts)

    val_class_crcts =np.array(val_class_crcts)
    val_class_cnts =np.array(val_class_cnts)

    train_cl_ac = train_class_crcts/train_class_cnts
    val_cl_ac   = val_class_crcts/val_class_cnts
    print("calss accrs",val_cl_ac)

    print(val_class_cnts)

    print('---------------------------------------------------------------------------------------------')
    end=time.time()
    print("epoch time = ",end-st)
    #IMPRTANT TO PT IT BACK ON GPU
    if(use_cuda):
        model.cuda()

####################################### END OF CODE ########################################
