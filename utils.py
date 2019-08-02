import numpy as np
import time
import os

def normalization(data,minv,maxv,q,w):
    
    
    data_new=[]
    for k,pac in enumerate(data):
        tmp=q+(w-q)*(pac-minv)/(maxv-minv)
        data_new.append(tmp)
        
    return data_new



def replace_nan(data,q):
    data_new=[]
    for k,pac in enumerate(data):
        tmp=pac
        tmp[np.isnan(pac)]=q
        data_new.append(tmp)
        
    return data_new





class AdjustLearningRate():
    def __init__(self,lr_step=10):
        self.lr_step=lr_step
        self.best_loss=999999999
        self.best_loss_pos=0
        self.stopcount=0
        
    def step(self,optimizer,iteration=0,loss=0):

        try:
            with open('lr_change.txt', 'r') as f:
                x = f.readlines()
                lr=float(x[0])
            time.sleep(1)
            os.remove('lr_change.txt')
            
            print('lr was set to: ' + str(x))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        except:
            pass
        
        if  loss<self.best_loss:
            self.best_loss_pos=iteration
            self.best_loss=loss
#        
        print(self.best_loss,loss)
#        
#        print(str(self.batch*iteration-self.best_loss_pos) + '////' + str(self.lr_step))
        if  iteration-self.best_loss_pos>self.lr_step:
            self.stopcount+=1
            self.best_loss_pos=iteration
            self.best_loss=loss
            print('lr down')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] *0.5
                
        if  self.stopcount>=6:
            return 1
        else:
            return 0

