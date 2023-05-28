import torch
from Baseline import CustomDataset, FairBatch
from torch import nn
class NN1(torch.nn.Module):
    def __init__(self,d_in):
        super(NN1, self).__init__()
        self.layer1 = nn.Linear(d_in, 75)
        self.layer2 = nn.Linear(75, 1)        
    def forward(self, x):
        # define forward pass
        x = nn.functional.sigmoid(self.layer1(x))
        x = nn.functional.sigmoid(self.layer2(x))
        return x
class NN2(torch.nn.Module):
    def __init__(self,d_in):
        super(NN2, self).__init__()
        self.layer1 = nn.Linear(d_in, 125)
        self.layer2 = nn.Linear(125, 1)
    def forward(self, x):
        # define forward pass
        x = nn.functional.sigmoid(self.layer1(x)) 
        x = self.layer2(x) 
        return x

def train(method, X_0, y_0, X_1, y_1, num_itr ,lr,r,seed = 0):
  torch.manual_seed(seed)
  if method == 'logistic':
    model = NN1(X_0.shape[1])
    LL = nn.BCELoss()
  if method == 'linear':
    model = NN2(X_0.shape[1])
    LL = nn.MSELoss()
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)])
  X = torch.vstack([X_0,X_1])
  t = torch.tensor(0.1,requires_grad=False).float()
  count = 0
  while count < num_itr:
    count +=1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg = 0
    for p in model.parameters():
      reg += r*torch.norm(p)**2
    loss = LL(model.forward(X),y.reshape(-1,1))+reg 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  
  return model, LL(model.forward(X_0),y_0.unsqueeze(1)).detach(), LL(model.forward(X_1),y_1.unsqueeze(1)).detach(),LL(model.forward(X),y.reshape(-1,1)).detach()

def penalty_method2(method, X_0, y_0, X_1, y_1, num_itr ,lr,r,gamma,epsilon,seed = 0):
  torch.manual_seed(seed)
  if method == 'logistic':
    model = NN1(X_0.shape[1])
    LL = nn.BCELoss()
  if method == 'linear':
    model = NN2(X_0.shape[1])
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
      reg = 0
      for p in model.parameters():
        reg += r*torch.norm(p)**2
      loss  = LL(model.forward(X).reshape(-1,), y.reshape(-1,))+reg
      if abs(loss-old_loss)<epsilon:
        break
      old_loss = loss
  return model, LL(model.forward(X_0),y_0.unsqueeze(1)).detach(), LL(model.forward(X_1),y_1.unsqueeze(1)).detach(),LL(model.forward(X),y.reshape(-1,1)).detach()


def fair_batch2(method, X_0, y_0, X_1, y_1, num_itr ,lr,r,alpha,gamma,seed,epsilon):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)]).reshape(-1,).to(device)
  m = y.shape[0]
  X = torch.vstack([X_0,X_1]).to(device)
  z = torch.vstack([torch.zeros(len(y_0)).unsqueeze(1),torch.ones(len(y_1)).unsqueeze(1)]).reshape(-1,).to(device)
  train_data = CustomDataset(X, y, z)
  torch.manual_seed(seed)
  
  if method == 'logistic':
    model = NN1(X_0.shape[1]).to(device)
    criterion = nn.BCELoss()
  if method == 'linear':
    model = NN2(X_0.shape[1]).to(device)
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
        reg = 0
        for p in model.parameters():
              reg += r*torch.norm(p)**2
        loss  = criterion(model.forward(X).reshape(-1,), y.reshape(-1,))+reg
        if abs(loss-old_loss)<epsilon:
          break
        old_loss = loss
         
  return model, criterion(model.forward(X_0.to(device)),y_0.unsqueeze(1).to(device)).detach(), criterion(model.forward(X_1.to(device)),y_1.unsqueeze(1).to(device)).detach(),criterion(model.forward(X.to(device)),y.reshape(-1,1).to(device)).detach()
