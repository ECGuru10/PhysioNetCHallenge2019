
import torch.nn as nn
import torch.nn.functional as F 
import torch
from lstms.lstm import LayerNormGalLSTM





class LSTM0(nn.Module):
  def __init__(self,x_size,h_size,y_size,lstm_layers=3,dropout=0.5):
    super(LSTM, self).__init__()
    
    self.lstm_layers=lstm_layers
    self.h_size=h_size

    self.lstm=nn.LSTM(x_size,h_size,batch_first=True,num_layers=self.lstm_layers,dropout=dropout)    
    
    self.linear1=nn.Linear(h_size+x_size,h_size)##x2 for bidirectional
    self.do=nn.Dropout(p=dropout)
    self.linear2=nn.Linear(h_size,h_size)
    self.linear3=nn.Linear(h_size,y_size)

  def forward(self, x):
    
    y,(self.h,self.c)=self.lstm(x,(self.h,self.c))
    
    y=self.linear1(torch.cat((x,y),2))   ### concatenation of input and lstm output  - "residual conection"
    
    y=F.relu(y)
    y=self.do(y)
    
    y=self.linear2(y)
    y=F.relu(y)
    
    y=self.linear3(y)
        
    return y
  
  def init_hiden(self,batch):
    self.h=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()##x2 for bidirectional
    self.c=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()##x2 for bidirectional
    
    
    
    
class LSTM_residual(nn.Module):
  def __init__(self,x_size,h_size,y_size,dropout=0.3,blocks=5):
    super(LSTM_residual, self).__init__()
    
    
    self.h_size=h_size
    self.blocks=blocks
    self.dropout=dropout
    
    self.linear_first=nn.Linear(x_size,h_size)
    
    
    self.linears1=nn.ModuleList()
    self.lstms=nn.ModuleList()
    self.linears2=nn.ModuleList()
    self.dos=nn.ModuleList()
    self.linears3=nn.ModuleList()
    
    for k in range(self.blocks):
        self.linears1.append(nn.Linear(x_size+h_size*2,h_size))
#        self.lstms.append(nn.LSTM(x_size,h_size,batch_first=True,num_layers=1,dropout=dropout))
        self.lstms.append(LayerNormGalLSTM(h_size,h_size,dropout=dropout))
        self.linears2.append(nn.Linear(2*h_size,h_size))
        self.dos.append(nn.Dropout(p=dropout))
        self.linears3.append(nn.Linear(h_size,h_size))
    
    self.linear_last=nn.Linear(h_size,y_size)
    

  def forward(self, x):
      
      
    
    y=self.linear_first(x) 
    y_last=y
      
    for k in range(self.blocks):
        print(k)
        y=torch.cat((x,y,y_last),2)
        y=self.linears1[k](y)
        y_last=y
        y=F.relu(y)
        h=torch.zeros((y.size(0),1, self.h_size)).cuda()
        c=torch.zeros((y.size(0),1, self.h_size)).cuda()
        self.lstms[k].sample_mask()
        yy=[]
        for kk in range(y.size(1)):
            y_tmp=y[:,[kk],:]
            y_tmp,(h,c)=self.lstms[k](y_tmp,(h,c))
            yy.append(y_tmp)
            
        y=torch.cat(yy,1)
        y=torch.cat((y,y_last),2)
        y=self.linears2[k](y)
        y=self.dos[k](y)
        y=F.relu(y)
        y=self.linears3[k](y)
        y=F.relu(y)
    
    
    y=self.linear_last(y)

        
    return y
  
  def init_hiden(self,batch):
      pass
#    self.hs=[]
#    self.cs=[]
#    for k in range(self.blocks):
#        self.hs.append(torch.zeros((batch, 1, self.h_size)).cuda())##x2 for bidirectional
#        self.cs.append(torch.zeros((batch, 1, self.h_size)).cuda())##x2 for bidirectional
        
        
        
        
        