# -*- coding: utf-8 -*-
"""
 
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,bias_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.2, init_max=0.2):
        super(ConcreteDropout, self).__init__()

        
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit_dense = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit_dense)
        
        out = layer(self._concrete_dropout(x, p))
        
        sum_kernel=torch.sum(torch.pow(layer.weight, 2)) 
        sum_bias=torch.sum(torch.pow(layer.bias, 2)) 
        
        weights_regularizer = self.weight_regularizer * sum_kernel / (1 - p)
        bias_regularizer = self.bias_regularizer * sum_bias
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality =x.shape[-1] # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer +bias_regularizer +dropout_regularizer
        return out, regularization,p.cpu().detach().numpy()
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

class ConcreteDropout_lstm(nn.Module):
    def __init__(self, weight_regularizer=1e-6,bias_regularizer=1e-6, dropout_regularizer=1e-5, 
                 hiddensize=40,init_min=0.2, init_max=0.2):
        super(ConcreteDropout_lstm, self).__init__()

        
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer=bias_regularizer
        self.dropout_regularizer = dropout_regularizer

        self.hiddensize=hiddensize
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p_logit_rec = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
    def forward(self, x,layer):
        p = torch.sigmoid(self.p_logit)
        p_rec = torch.sigmoid(self.p_logit_rec)
        
        batchsize=x.shape[0]
        timestep=x.shape[1]
        h1=torch.zeros(batchsize,self.hiddensize).cuda()
        c1=torch.zeros(batchsize,self.hiddensize).cuda()
   
        x_ou=[]
        rnn = layer
        for i in range(timestep):
            if i==0:
                x_,drop_mask_x=self._concrete_dropout_lstm_h(x[:,i,:],p)
                h1,drop_mask=self._concrete_dropout_lstm_h(h1,p_rec)
            else:
                x_  = torch.mul(x[:,i,:], drop_mask_x)
                x_ /= (1 - p)
                h1  = torch.mul(h1, drop_mask)
                h1 /= (1 - p_rec)
            h1,c1=rnn(x_,(h1,c1))
            x_ou.append(h1)
        x_out=torch.stack(x_ou, dim = 1)

        sum_kernel=torch.sum(torch.pow(rnn.weight_ih, 2)) 
        sum_rec=torch.sum(torch.pow(rnn.weight_hh, 2))
        sum_bias=torch.sum(torch.pow(rnn.bias_ih, 2)) + torch.sum(torch.pow(rnn.bias_hh, 2))
        
        weights_regularizer = self.weight_regularizer *( sum_kernel / (1 - p)+sum_rec/ (1 - p_rec))
        bias_regularizer=self.bias_regularizer*sum_bias
        
        dropout_regularizer_kernel = p * torch.log(p)
        dropout_regularizer_kernel += (1. - p) * torch.log(1. - p)
        dropout_regularizer_rec = p_rec * torch.log(p_rec)
        dropout_regularizer_rec += (1. - p_rec) * torch.log(1. - p_rec)
        
        input_dimensionality_kernel = x.shape[-1] # Number of elements of first item in batch
        input_dimensionality_rec = h1.shape[-1]
        dropout_regularizer = self.dropout_regularizer * (input_dimensionality_kernel*dropout_regularizer_kernel+input_dimensionality_rec*dropout_regularizer_rec)
        
        regularization = weights_regularizer + bias_regularizer+dropout_regularizer
        return x_out,h1, regularization,p.cpu().detach().numpy(),p_rec.cpu().detach().numpy()

    def _concrete_dropout_lstm_h(self, h,p_rec):
        eps = 1e-7
        temp = 0.1

        unif_noise_rec = torch.rand_like(h)

        drop_prob_rec = (torch.log(p_rec + eps)
                    - torch.log(1 - p_rec + eps)
                    + torch.log(unif_noise_rec + eps)
                    - torch.log(1 - unif_noise_rec + eps))
        
        drop_prob_rec = torch.sigmoid(drop_prob_rec / temp)
        random_tensor_rec = 1 - drop_prob_rec
        retain_prob_rec = 1 - p_rec
        
        h  = torch.mul(h, random_tensor_rec)
        h /= retain_prob_rec
        return h,random_tensor_rec

#%%
class Model(nn.Module):
    def __init__(self, weight_regularizer, bias_regularizer,dropout_regularizer,hiddensize):
        super(Model, self).__init__()

        self.linear_mu = nn.Linear(hiddensize[-1], 7)

        self.lstmcell1=nn.LSTMCell(hiddensize[0], hiddensize[1])
        self.lstmcell2=nn.LSTMCell(hiddensize[1], hiddensize[2])
        self.lstmcell3=nn.LSTMCell(hiddensize[2], hiddensize[3])
        self.lstm1 =ConcreteDropout_lstm(weight_regularizer=weight_regularizer,bias_regularizer=bias_regularizer,
                                    dropout_regularizer=dropout_regularizer,hiddensize=hiddensize[1])
        
        self.lstm2 =ConcreteDropout_lstm(weight_regularizer=weight_regularizer,bias_regularizer=bias_regularizer,
                                    dropout_regularizer=dropout_regularizer,hiddensize=hiddensize[2])
        
        self.lstm3 =ConcreteDropout_lstm(weight_regularizer=weight_regularizer,bias_regularizer=bias_regularizer,
                                   dropout_regularizer=dropout_regularizer,hiddensize=hiddensize[3])
        
        self.mu=ConcreteDropout(weight_regularizer=weight_regularizer,bias_regularizer=bias_regularizer,
                                dropout_regularizer=dropout_regularizer)
        
    def forward(self, x):
        regularization = torch.empty(5, device=x.device)

        x1,h1, regularization[0],p1,p_rec1 = self.lstm1(x,self.lstmcell1)
        x2,h2, regularization[1],p2,p_rec2 = self.lstm2(x1,self.lstmcell2)
        x3,h3, regularization[2],p3,p_rec3 = self.lstm3(x2,self.lstmcell3)
        mean1, regularization[3],p_dense1 = self.mu(h3,self.linear_mu)
        #alpha=torch.exp(mean1)
        alpha=F.softplus(mean1)+1
       # prob=nn.functional.softmax(mean1,1)
        prob=alpha/torch.sum(alpha,-1).unsqueeze(1)

        return prob, torch.sum(alpha,-1).unsqueeze(1),regularization.sum(),[p1[0],p2[0],p_rec1[0],p_rec2[0],p_dense1[0]]

#format(sum(x.numel() for x in model.parameters()))
#print(list(model.parameters()))

#%%
class Model_RNN(nn.Module):
    def __init__(self, hiddensize):
        super(Model_RNN, self).__init__()
        self.mu = nn.Linear(hiddensize[-1], 7)
        
        self.Sp=nn.Softplus()
        self.lstm1=nn.LSTM(hiddensize[0], hiddensize[1],batch_first=True)
        self.lstm2=nn.LSTM(hiddensize[1], hiddensize[2],batch_first=True)
        self.lstm3=nn.LSTM(hiddensize[2], hiddensize[3],batch_first=True)
        
    def forward(self, x):
        x1,(h1,c1) = self.lstm1(x)
        x2,(h2,c2) = self.lstm2(x1)
        x3,(h3,c3) = self.lstm3(x2)
        means=self.mu(h3.squeeze(0))
        alpha=torch.exp(means)
        prob=alpha/torch.sum(alpha,-1).unsqueeze(1)
        return prob
    
#%%
class Model_CNN(nn.Module):
    def __init__(self,hiddensize,oc,h,kz,sr):
        super(Model_CNN, self).__init__()

        self.linear_mu = nn.Linear(16, 7)
        self.Sp=nn.Softplus()
        self.relu=nn.PReLU()
        self.con1=nn.Conv1d(in_channels=oc[0], out_channels=oc[1], kernel_size=h[0])
        self.bn1=nn.BatchNorm1d(oc[1])
        self.ma1=nn.MaxPool1d(kernel_size=kz,stride=sr)
        
        self.con2=nn.Conv1d(in_channels=oc[1], out_channels=oc[2], kernel_size=h[1])
        self.bn2=nn.BatchNorm1d(oc[2])
        self.ma2=nn.MaxPool1d(kernel_size=kz,stride=sr)
       
        self.con3=nn.Conv1d(in_channels=oc[2], out_channels=oc[3], kernel_size=h[2])
        self.bn3=nn.BatchNorm1d(oc[3])
        self.ma3=nn.MaxPool1d(kernel_size=kz,stride=sr)
        
        self.linear_mu1 = nn.Linear(hiddensize, 16)
        self.bn3=nn.BatchNorm1d(16) 
        
    def forward(self, x):

        x1 =self.con1(x)
        #print(x1)
        #x1=self.bn1(x1)
        x1=self.relu(x1)
        x1=self.ma1(x1)
        
        x2 =self.con2(x1)
       # x2=self.bn1(x2)
        x2=self.relu(x2)
        x2=self.ma2(x2)
        
        x2 =self.con3(x2)
       # x2=self.bn1(x2)
        x2=self.relu(x2)
        x2=self.ma3(x2)

        x4=x2.flatten(start_dim=1)

        x4 = self.linear_mu1(x4)
        #x4=self.bn3(x4)
        x4=self.relu(x4)
       
        mean1 = self.linear_mu(x4)
        #print(mean1.shape)
        alpha=self.Sp(mean1)
        prob=alpha/torch.sum(alpha,-1).unsqueeze(1)
        return prob