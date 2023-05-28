import torch
from torch import nn
class LR(torch.nn.Module):
    def __init__(self,d_in):
        super(LR, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d_in, 1, bias = True))
    def forward(self, x):
        logits = torch.nn.functional.sigmoid(self.linear_relu_stack(x))
        return logits
class LinR(torch.nn.Module):
    def __init__(self,d_in):
        super(LinR, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d_in, 1, bias = True))
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def penalty_method(method, X_0, y_0, X_1, y_1, num_itr ,lr,r,gamma,epsilon):
  torch.manual_seed(0)
  if method == 'logistic':
    model = LR(X_0.shape[1])
    LL = nn.BCELoss()
  if method == 'linear':
    model = LinR(X_0.shape[1])
    LL = nn.MSELoss()
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)])
  X = torch.vstack([X_0,X_1])
  t = torch.tensor(0.1,requires_grad=False).float()
  count = 0
  old_loss = -1
  while count < num_itr:
    count +=1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if count%100 == 0:
      t = 2*t
    reg = 0
    for p in model.parameters():
      reg += r*torch.norm(p)**2
    loss = LL(model.forward(X),y.reshape(-1,1)) + t*torch.nn.functional.relu(abs(LL(model.forward(X_0),y_0.unsqueeze(1)) -LL(model.forward(X_1),y_1.unsqueeze(1))) -gamma)**2+reg 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    with torch.no_grad():
      loss  = LL(model.forward(X).reshape(-1,), y.reshape(-1,))+reg
      if abs(loss-old_loss)<epsilon:
        break
      old_loss = loss
  return model, LL(model.forward(X_0),y_0.unsqueeze(1)).detach(), LL(model.forward(X_1),y_1.unsqueeze(1)).detach(),LL(model.forward(X),y.reshape(-1,1)).detach()

import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch


class CustomDataset(Dataset):
    """Custom Dataset.
    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)
    
    
class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch.
    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values in FairBatch.
        
    """
    def __init__(self, model, method, x_tensor, y_tensor, z_tensor, batch_size, alpha, target_fairness, gamma, replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.model = model
        self.model_type = method
        self.gamma = gamma
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z_data = z_tensor
        
        self.alpha = alpha
        self.fairness_type = target_fairness
        self.replacement = replacement
        
        self.N = len(z_tensor)
        
        self.batch_size = batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = list(set(z_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        #for tmp_y in self.y_item:
        #    self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        #for tmp_yz in self.yz_tuple:
        #    self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        for tmp_z in self.z_item:
            self.S[tmp_z] = self.batch_size * (self.z_len[tmp_z])/self.N

        
        self.lb1 = (self.S[1])/(self.S[1]+(self.S[0]))
        #self.lb2 = (self.S[-1,1])/(self.S[-1,1]+(self.S[-1,0]))
    
    
    def adjust_lambda(self):
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are decribed in the paper.
        """
        
        self.model.eval()
        logit = self.model(self.x_data)

        if self.model_type=='linear':
          criterion = torch.nn.MSELoss(reduction = 'none')
        else:
          criterion = torch.nn.BCELoss(reduction = 'none')
        
                
        if self.fairness_type == 'el':
            
            yhat_yz = {}
            yhat_y = {}
            yhat_z = {}
                        
            el_loss = criterion (logit, self.y_data.reshape(-1,1))
            
            for tmp_z in self.z_item:
                yhat_z[tmp_z] = float(torch.sum(el_loss[self.z_index[tmp_z]])) / self.z_len[tmp_z]

            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_z[1] - yhat_z[0]>self.gamma:
                self.lb1 += self.alpha
            elif yhat_z[1] - yhat_z[0]<-self.gamma:
                self.lb1 -= self.alpha
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1 
                

    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index

    
    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """

        if self.fairness_type == 'original':
            
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)
            
            for i in range(self.batch_num):
                yield sort_index[i]
            
        else:
        
            self.adjust_lambda() # Adjust the lambda values
            each_size = {}
            
            
            # Based on the updated lambdas, determine the size of each class in a batch
            if self.fairness_type == 'el':
                # lb1 * loss_z1 + (1-lb1) * loss_z0
                
                each_size[1] = round(self.lb1 * (self.S[0] + self.S[1]))
                each_size[0] = round((1-self.lb1) * (self.S[0] + self.S[1]))
               
            sort_index_z_1 = self.select_batch_replacement(each_size[1], self.z_index[1], self.batch_num, self.replacement)
            sort_index_z_0 = self.select_batch_replacement(each_size[0], self.z_index[0], self.batch_num, self.replacement)
            for i in range(self.batch_num):
                key_in_fairbatch = sort_index_z_0[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_z_1[i].copy()))
                random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch

                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)

def fair_batch(method, X_0, y_0, X_1, y_1, num_itr ,lr,r,alpha,gamma,seed,epsilon):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)]).reshape(-1,).to(device)
  m = y.shape[0]
  X = torch.vstack([X_0,X_1]).to(device)
  z = torch.vstack([torch.zeros(len(y_0)).unsqueeze(1),torch.ones(len(y_1)).unsqueeze(1)]).reshape(-1,).to(device)
  train_data = CustomDataset(X, y, z)
  torch.manual_seed(seed)
  
  if method == 'logistic':
    model = LR(X_0.shape[1]).to(device)
    criterion = nn.BCELoss()
  if method == 'linear':
    model = LinR(X_0.shape[1]).to(device)
    criterion = nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))

  losses = []
    
    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------

  sampler = FairBatch (model,method,train_data.x, train_data.y, train_data.z, batch_size = 100, alpha = alpha, target_fairness = 'el', gamma = gamma, replacement = False, seed = seed)
  train_loader = torch.utils.data.DataLoader (train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  old_loss = -1
  for epoch in range(num_itr):
      for batch_idx, (data, target, z) in enumerate (train_loader):
          data = data.to(device)
          target = target.to(device)
          z = z.to(device)
          label_predicted = model.forward(data)
          reg = 0
          optimizer = torch.optim.Adam(model.parameters(), lr)
          for p in model.parameters():
              reg += r*torch.norm(p)**2
          loss  = criterion(label_predicted.reshape(-1,), target.reshape(-1,))+reg
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step() 
      with torch.no_grad():
        loss  = criterion(model.forward(X).reshape(-1,), y.reshape(-1,))+reg
        if abs(loss-old_loss)<epsilon:
          break
        old_loss = loss
         
  return model, criterion(model.forward(X_0),y_0.unsqueeze(1)).detach(), criterion(model.forward(X_1),y_1.unsqueeze(1)).detach(),criterion(model.forward(X),y.reshape(-1,1)).detach()
