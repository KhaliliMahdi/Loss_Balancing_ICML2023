import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment',choices=[1,2], type = int, default = 1)
parser.add_argument('--gamma',type=float,default=0.0)

args = parser.parse_args()
import warnings
warnings.simplefilter('ignore')

import torch
from data.law_data import *#law dataset used in the first experiment
from data.Adult_data import *#Adult dataset used in the first experiment
from Algorithms.Algorithms import *#Algorithm 1, Algorithm 2, and Algorithm 3 implemented in Convex_solver
from Algorithms.Baseline import penalty_method, fair_batch #This is the baseline

Table = args.experiment # if you want to see the result of table 2, set Tabel =2
if Table==1:
  dataset = law_data
  method = 'linear'
  LL = torch.nn.MSELoss()
elif Table==2:
  dataset = Adult_dataset
  method = 'logistic'
  LL = torch.nn.BCELoss()


#Please change the gamma, if you want to see the result for diffrent fairness level
gamma = args.gamma
ite = 1000 # maximum number of iterations for PM and FairBatch
lr = 0.001 # learning rate
r = 0.002 # regularizer parameter

#In this part, we can generate the metrics in Table 1 or Table 2 for the penalty method
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []


import time

tt = []
for i in range(5):#repeat the experiemnt for 5 times
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  X_0 = torch.tensor(X_0,requires_grad=False).float()
  X_1 = torch.tensor(X_1,requires_grad=False).float()
  y_0 = torch.tensor(y_0,requires_grad=False).float()
  y_1 = torch.tensor(y_1,requires_grad=False).float()
  start = time.time()
  model, l0,l1,l = fair_batch(method, X_0, y_0, X_1, y_1, ite, lr, r, 0.005, gamma, i,1e-6)
  end = time.time()
  tt.append(end-start)
  #training loss
  loss.append(l)
  loss_difference.append(abs(l0-l1))
  #preparing the test dataset
  X_test_0 = torch.tensor(X_test_0,requires_grad=False).float()
  X_test_1 = torch.tensor(X_test_1,requires_grad=False).float()
  y_test_0 = torch.tensor(y_test_0,requires_grad=False).float()
  y_test_1 = torch.tensor(y_test_1,requires_grad=False).float()
  y_test = torch.vstack([y_test_0.unsqueeze(1),y_test_1.unsqueeze(1)])
  X_test = torch.vstack([X_test_0,X_test_1])
  #test loss
  loss_difference_test.append(abs(LL(model.forward(X_test_0),y_test_0.unsqueeze(1))- LL(model.forward(X_test_1),y_test_1.unsqueeze(1))).detach())
  loss_test.append(LL(model.forward(X_test),y_test.reshape(-1,1)).detach())
  
loss_with_variance = (np.mean(loss),np.std(loss))
loss_difference_variance = (np.mean(loss_difference),np.std(loss_difference))
loss_with_variance_test = (np.mean(loss_test),np.std(loss_test))
loss_difference_variance_test = (np.mean(loss_difference_test),np.std(loss_difference_test))


print("Table {}".format(Table), "FairBatch")
print("gamma = {}".format(gamma))
print('training loss = {} +- {}'.format(loss_with_variance[0],loss_with_variance[1]))
print('training loss difference = {} +- {}'.format(loss_difference_variance[0],loss_difference_variance[1]))
print('test loss = {} +- {}'.format(loss_with_variance_test[0],loss_with_variance_test[1]))
print('test loss difference = {} +- {}'.format(loss_difference_variance_test[0],loss_difference_variance_test[1]))
print('time:{}+-{}'.format(np.mean(tt),np.std(tt)))

#In this part, we can generate the metrics in Table 1 or Table 2 for the penalty method


loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
tt = []
for i in range(5):#repeat the experiemnt for 5 times
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  X_0 = torch.tensor(X_0,requires_grad=False).float()
  X_1 = torch.tensor(X_1,requires_grad=False).float()
  y_0 = torch.tensor(y_0,requires_grad=False).float()
  y_1 = torch.tensor(y_1,requires_grad=False).float()
  start = time.time()
  model, l0,l1,l = penalty_method(method, X_0, y_0, X_1, y_1,ite,lr,r,gamma,1e-6)
  end = time.time()
  tt.append(end-start)
  #training loss
  loss.append(l)
  loss_difference.append(abs(l0-l1))
  #preparing the test dataset
  X_test_0 = torch.tensor(X_test_0,requires_grad=False).float()
  X_test_1 = torch.tensor(X_test_1,requires_grad=False).float()
  y_test_0 = torch.tensor(y_test_0,requires_grad=False).float()
  y_test_1 = torch.tensor(y_test_1,requires_grad=False).float()
  y_test = torch.vstack([y_test_0.unsqueeze(1),y_test_1.unsqueeze(1)])
  X_test = torch.vstack([X_test_0,X_test_1])
  #test loss
  loss_difference_test.append(abs(LL(model.forward(X_test_0),y_test_0.unsqueeze(1))- LL(model.forward(X_test_1),y_test_1.unsqueeze(1))).detach())
  loss_test.append(LL(model.forward(X_test),y_test.reshape(-1,1)).detach())
  
loss_with_variance = (np.mean(loss),np.std(loss))
loss_difference_variance = (np.mean(loss_difference),np.std(loss_difference))
loss_with_variance_test = (np.mean(loss_test),np.std(loss_test))
loss_difference_variance_test = (np.mean(loss_difference_test),np.std(loss_difference_test))

print("Table {}".format(Table), "PM")
print("gamma = {}".format(gamma))
print('training loss = {} +- {}'.format(loss_with_variance[0],loss_with_variance[1]))
print('training loss difference = {} +- {}'.format(loss_difference_variance[0],loss_difference_variance[1]))
print('test loss = {} +- {}'.format(loss_with_variance_test[0],loss_with_variance_test[1]))
print('test loss difference = {} +- {}'.format(loss_difference_variance_test[0],loss_difference_variance_test[1]))
print('time:{}+-{}'.format(np.mean(tt),np.std(tt)))

#In this part, we can generate the metrics in Table 1 for LinRe
#Please change the gamma, if you want to see the result for diffrent fairness level
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
tt = []
for i in range(5):
  #loading data
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  start = time.time()
  w,b, l0,l1,l = solve_lin_constrained_opt(X_0, y_0, X_1, y_1, gamma, r ,method)
  end = time.time()
  tt.append(end-start)
  #training loss
  loss.append(l)
  loss_difference.append(abs(l0-l1))
  #calculating test loss and test loss difference
  l0,l1,l = calculate_loss(w,b,X_test_1, y_test_1, X_test_0, y_test_0, method)
  loss_difference_test.append(abs(l0-l1))
  loss_test.append(l)
  
loss_with_variance = (np.mean(loss),np.std(loss))
loss_difference_variance = (np.mean(loss_difference),np.std(loss_difference))
loss_with_variance_test = (np.mean(loss_test),np.std(loss_test))
loss_difference_variance_test = (np.mean(loss_difference_test),np.std(loss_difference_test))

print("Table {}".format(Table), "LinRe")
print("gamma = {}".format(gamma))
print('training loss = {} +- {}'.format(loss_with_variance[0],loss_with_variance[1]))
print('training loss difference = {} +- {}'.format(loss_difference_variance[0],loss_difference_variance[1]))
print('test loss = {} +- {}'.format(loss_with_variance_test[0],loss_with_variance_test[1]))
print('test loss difference = {} +- {}'.format(loss_difference_variance_test[0],loss_difference_variance_test[1]))
print('time:{}+-{}'.format(np.mean(tt),np.std(tt)))

#In this part, we can generate the metrics in Table 1 for Algorithm 2
#Please change the gamma, if you want to see the result for diffrent fairness level
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
tt = []
for i in range(5):
  #loading data
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  start = time.time()
  w,b, l0,l1,l = Algorithm2(X_0, y_0, X_1, y_1, gamma, r, method)
  end = time.time()
  tt.append(end-start)
  #training loss
  loss.append(l)
  loss_difference.append(abs(l0-l1))
  #calculating test loss and test loss difference
  l0,l1,l = calculate_loss(w,b,X_test_1, y_test_1, X_test_0, y_test_0, method)
  loss_difference_test.append(abs(l0-l1))
  loss_test.append(l)
  
loss_with_variance = (np.mean(loss),np.std(loss))
loss_difference_variance = (np.mean(loss_difference),np.std(loss_difference))
loss_with_variance_test = (np.mean(loss_test),np.std(loss_test))
loss_difference_variance_test = (np.mean(loss_difference_test),np.std(loss_difference_test))



print("Table {}".format(Table), "Algorithm 2")
print("gamma = {}".format(gamma))
print('training loss = {} +- {}'.format(loss_with_variance[0],loss_with_variance[1]))
print('training loss difference = {} +- {}'.format(loss_difference_variance[0],loss_difference_variance[1]))
print('test loss = {} +- {}'.format(loss_with_variance_test[0],loss_with_variance_test[1]))
print('test loss difference = {} +- {}'.format(loss_difference_variance_test[0],loss_difference_variance_test[1]))
print('time:{}+-{}'.format(np.mean(tt),np.std(tt)))

#In this part, we can generate the metrics in Table 1 for Algorithm 3
#Please change the gamma, if you want to see the result for diffrent fairness level
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
tt = []
for i in range(5):
  #loading data
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  start = time.time()
  w,b, l0,l1,l = Algorithm3(X_0, y_0, X_1, y_1, gamma, r, method)
  end = time.time()
  tt.append(end-start)
  #training loss
  loss.append(l)
  loss_difference.append(abs(l0-l1))
  #calculating test loss and loss difference
  l0,l1,l = calculate_loss(w,b,X_test_1, y_test_1, X_test_0, y_test_0,method)
  loss_difference_test.append(abs(l0-l1))
  loss_test.append(l)
  
loss_with_variance = (np.mean(loss),np.std(loss))
loss_difference_variance = (np.mean(loss_difference),np.std(loss_difference))
loss_with_variance_test = (np.mean(loss_test),np.std(loss_test))
loss_difference_variance_test = (np.mean(loss_difference_test),np.std(loss_difference_test))

print("Table {}".format(Table), "Algorithm 3")
print("gamma = {}".format(gamma))
print('training MSE = {} +- {}'.format(loss_with_variance[0],loss_with_variance[1]))
print('training MSE difference = {} +- {}'.format(loss_difference_variance[0],loss_difference_variance[1]))
print('test MSE = {} +- {}'.format(loss_with_variance_test[0],loss_with_variance_test[1]))
print('training MSE difference = {} +- {}'.format(loss_difference_variance_test[0],loss_difference_variance_test[1]))
print('time:{}+-{}'.format(np.mean(tt),np.std(tt)))
