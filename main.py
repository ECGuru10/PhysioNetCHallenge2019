import numpy as np
import matplotlib.pyplot as plt
from utils import normalization, replace_nan, AdjustLearningRate, dice_loss, get_utility
from nets import LSTM_residual

import torch.nn as nn
import torch.nn.functional as F 
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence

from scipy.optimize import minimize_scalar

data_path=('../train_test_data2')

XTrain=np.load(data_path+'/XTrain.npy',allow_pickle=True)
YTrain=np.load(data_path+'/YTrain.npy',allow_pickle=True)
XTest=np.load(data_path+'/XTest.npy',allow_pickle=True)
YTest=np.load(data_path+'/YTest.npy',allow_pickle=True)
    

minv=np.nanmin(np.concatenate(XTrain,axis=0),axis=0)
maxv=np.nanmax(np.concatenate(XTrain,axis=0),axis=0)

XTrain=normalization(XTrain,minv,maxv,1,5)
XTest=normalization(XTest,minv,maxv,1,5)

XTrain=replace_nan(XTrain,0)
XTest=replace_nan(XTest,0)


in_size=XTrain[0].shape[1]
out_size=1
hiden_dim=200

net=LSTM_residual(in_size,hiden_dim,out_size).cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-6)

adj_lr = AdjustLearningRate(lr_step=3000)



train_iter=10000
batch=64
pad_token=-1


N=len(XTrain)
N_test=len(XTest)


train_loss=[]
test_loss=[]
position=[]
train_loss_tmp=[]
test_loss_tmp=[]
test_util=[]

for it in range(train_iter):
    if it%1==0:
        print(it)
    
    net.train()
  
    optimizer.zero_grad()
  
    
    ind=np.random.randint(low=0,high=N,size=batch)
    xx=[]
    tt=[]
    lengths=[]
    for i in ind:
        xx.append(torch.Tensor(XTrain[i]))
        tt.append(torch.Tensor(torch.Tensor(YTrain[i]))) 
        lengths.append(len(YTrain[i]))

    xx=pad_sequence(xx,batch_first=True,padding_value=pad_token).cuda()
    tt=pad_sequence(tt,batch_first=True,padding_value=pad_token).cuda()
    lengths=torch.Tensor(lengths)
    
    xx.requires_grad=True
    tt.requires_grad=True
    
    net.init_hiden(batch)
    

    
    
    yy=net(xx)
    yy=torch.sigmoid(yy)
    
    yy=yy[tt!=pad_token]
    tt=tt[tt!=pad_token]
    
    loss=dice_loss(yy,tt)
    
    loss.backward()
    
    nn.utils.clip_grad_norm_(net.parameters(), 10)
    
    optimizer.step()
    
    train_loss_tmp.append(loss.detach().cpu().numpy())
    
    
    if it%1000==0:# and it!=0:
        tt_store=[]
        yy_store=[]
        for itt in range(N_test):
            net.eval()
    
            ind=np.random.randint(low=0,high=N_test,size=batch)
            xx=[]
            tt=[]
            lengths=[]
            for i in ind:
                xx.append(torch.Tensor(XTest[i]))
                tt.append(torch.Tensor(torch.Tensor(YTest[i]))) 
                lengths.append(len(YTest[i]))
        
            xx=pad_sequence(xx,batch_first=True,padding_value=pad_token).cuda()
            tt=pad_sequence(tt,batch_first=True,padding_value=pad_token).cuda()
            lengths=torch.Tensor(lengths)
            
            xx.requires_grad=True
            tt.requires_grad=True
            
            net.init_hiden(batch)
            
            yy=net(xx)
            yy=torch.sigmoid(yy)
            
#            ttt=tt.detach().cpu().numpy()
#            yyy=yy.detach().cpu().numpy()
#            for k in range(batch):
#                tt_slice=ttt[k,:,:]
#                yy_slice=yyy[k,:,:]
#                tt_store.append(tt_slice[tt_slice!=pad_token])
#                yy_store.append(yy_slice[tt_slice!=pad_token])
            
            yy=yy[tt!=pad_token]
            tt=tt[tt!=pad_token]
            
            loss=dice_loss(yy,tt)
            
            test_loss_tmp.append(loss.detach().cpu().numpy())
            
            
            
        
            
        train_loss.append(np.mean(train_loss_tmp))
        test_loss.append(np.mean(test_loss_tmp))
        position.append(it)
        train_loss_tmp=[]
        test_loss_tmp=[]
        
        adj_lr.step(optimizer,iteration=it,loss=test_loss[-1])
        
#        
#        res=minimize_scalar(get_utility,bounds=(0.01, 0.99), method='bounded',args=(yy_store,tt_store),options=dict([('maxiter',20),('disp',1)]))
#        
#        test_util.append(get_utility(res.x,yy_store,tt_store))
        
#         + str(test_util[-1]) 
        
        for param_group in optimizer.param_groups:
            lr_act=param_group['lr']
            
        print(str(it) + ' train loss: ' + str(train_loss[-1]) + ' test loss: ' + str(test_loss[-1]) +'   util: '+ '  lr: ' +str(lr_act))
        
        plt.plot(position,train_loss)
        plt.plot(position,test_loss)
        plt.show()
        
#        plt.plot(position,test_util)
#        plt.show()
        
    
        
        
res=minimize_scalar(get_utility,bounds=(0.01, 0.99), method='bounded',args=(yy_store,tt_store),options=dict([('maxiter',50),('disp',1)]))

print(get_utility(res.x,yy_store,tt_store))        
    
