import numpy as np
import matplotlib.pyplot as plt
from utils import normalization, replace_nan, AdjustLearningRate

import torch.nn as nn
import torch.nn.functional as F 
import torch
import torch.optim as optim


data_path=('../train_test_data2')

XTrain=np.load(data_path+'/XTrain.npy')
YTrain=np.load(data_path+'/YTrain.npy')
XTest=np.load(data_path+'/XTest.npy')
YTest=np.load(data_path+'/YTest.npy')
    

minv=np.nanmin(np.concatenate(XTrain,axis=0),axis=0)
maxv=np.nanmax(np.concatenate(XTrain,axis=0),axis=0)

XTrain=normalization(XTrain,minv,maxv,1,5)
XTest=normalization(XTest,minv,maxv,1,5)

XTrain=replace_nan(XTrain,0)
XTest=replace_nan(XTest,0)





#optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-6)

#adj_lr = AdjustLearningRate(lr_step=100)



itrain_iter=10000
batch=64
pad_token=-1




X_lengths = [x.shape(0) for x in XTrain]
longest = np.max(X_lengths)
padded_X = np.ones((batch, longest)) * pad_token
for i, x_len in enumerate(X_lengths):
    sequence = XTrain[i]
    padded_X[i, 0:x_len] = sequence[:x_len]
    
XTrain=padded_X


dfsdf=dsfdsf


N=len(XTrain)


train_loss=[]
test_loss=[]
train_acc=[]
test_acc=[]
position=[]
train_acc_tmp=[]
train_loss_tmp=[]
test_acc_tmp=[]
test_loss_tmp=[]


for it in range(itrain_iter):
    
    net.train()
  
    optimizer.zero_grad()
  
    ind=np.random.randint(low=0,high=N,size=batch)

    xx=[]
    tt=[]
    for i in ind:
        tmp=encode_aa(XTrain[i])
        xx.append(torch.Tensor(tmp.astype(np.float32)))

        tmp=encode_nt(nt_train[i])
        tt.append(torch.Tensor(tmp.astype(np.float32)))   

