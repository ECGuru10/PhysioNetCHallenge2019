
import torch.nn as nn
import torch.nn.functional as F 
import torch


class LSTM(nn.Module):
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