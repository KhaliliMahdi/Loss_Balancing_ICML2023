import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma',type=float,default=0.0)

args = parser.parse_args()

import warnings
warnings.simplefilter('ignore')

import torch
from data.law_data import *#law dataset used in the first experiment
from Algorithms.Algorithms import *#Algorithm 1, Algorithm 2, and Algorithm 3 implemented in Convex_solver
from Algorithms.Non_linear import train, penalty_method2, fair_batch2

Table = 3
dataset = law_data
method = 'linear'
LL = torch.nn.MSELoss()


#Please change the gamma, if you want to see the result for diffrent fairness level
gamma = args.gamma
ite = 10 # number of iterations
lr = 0.005 # learning rate
r = 0.002 # regularizer parameter

#training a DNN without fairness constraint
last_hidden = []
for i in range(5):#repeat the experiemnt for 5 times
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(seed = i)
  X_0 = torch.tensor(X_0,requires_grad=False).float()
  X_1 = torch.tensor(X_1,requires_grad=False).float()
  y_0 = torch.tensor(y_0,requires_grad=False).float()
  y_1 = torch.tensor(y_1,requires_grad=False).float()
  X_test_0 = torch.tensor(X_test_0,requires_grad=False).float()
  X_test_1 = torch.tensor(X_test_1,requires_grad=False).float()
  y_test_0 = torch.tensor(y_test_0,requires_grad=False).float()
  y_test_1 = torch.tensor(y_test_1,requires_grad=False).float()
  model, l0,l1,l = train(method, X_0, y_0, X_1, y_1,ite,lr,r,seed = i)
  dictionary = {}
  sigmoid = torch.nn.functional.sigmoid
  f = lambda x:  np.array(sigmoid(model.layer1(x)).detach())
  dictionary['X_0'] = f(X_0)
  dictionary['X_1'] = f(X_1)
  dictionary['X_test_0'] = f(X_test_0)
  dictionary['X_test_1'] = f(X_test_1)
  dictionary['y_0'] = np.array(y_0)
  dictionary['y_1'] = np.array(y_1)
  dictionary['y_test_0'] = np.array(y_test_0)
  dictionary['y_test_1'] = np.array(y_test_1)
  last_hidden.append(dictionary)

#In this part, we can generate the metrics in Table 1 or Table 2 for the penalty method
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []

for i in range(5):#repeat the experiemnt for 5 times
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  X_0 = torch.tensor(X_0,requires_grad=False).float().to(device)
  X_1 = torch.tensor(X_1,requires_grad=False).float().to(device)
  y_0 = torch.tensor(y_0,requires_grad=False).float().to(device)
  y_1 = torch.tensor(y_1,requires_grad=False).float().to(device)
  model, l0,l1,l = fair_batch2(method, X_0, y_0, X_1, y_1, ite, lr, r, 0.005, gamma, i,1e-6)
    #training loss
  loss.append(l)
  loss_difference.append(abs(l0-l1))
    #preparing the test dataset
  
  X_test_0 = torch.tensor(X_test_0,requires_grad=False).float().to(device)
  X_test_1 = torch.tensor(X_test_1,requires_grad=False).float().to(device)
  y_test_0 = torch.tensor(y_test_0,requires_grad=False).float().to(device)
  y_test_1 = torch.tensor(y_test_1,requires_grad=False).float().to(device)
  y_test = torch.vstack([y_test_0.unsqueeze(1),y_test_1.unsqueeze(1)])
  X_test = torch.vstack([X_test_0,X_test_1])
    #test loss
  loss_difference_test.append(abs(LL(model.forward(X_test_0),y_test_0.unsqueeze(1))- LL(model.forward(X_test_1),y_test_1.unsqueeze(1))).detach())
  loss_test.append(LL(model.forward(X_test),y_test.reshape(-1,1)).detach())

loss = torch.tensor(loss)
loss_difference = torch.tensor(loss_difference)
loss_test = torch.tensor(loss_test)
loss_difference_test = torch.tensor(loss_difference_test)
loss_with_variance = (torch.mean(loss),torch.std(loss))
loss_difference_variance=(torch.mean(loss_difference),torch.std(loss_difference))
loss_with_variance_test = (torch.mean(loss_test),torch.std(loss_test))
loss_difference_variance_test = (torch.mean(loss_difference_test),torch.std(loss_difference_test))

print("Table {}".format(Table), "FairBatch")
print("gamma = {}".format(gamma))
print('training loss = {} +- {}'.format(loss_with_variance[0],loss_with_variance[1]))
print('training loss difference = {} +- {}'.format(loss_difference_variance[0],loss_difference_variance[1]))
print('test loss = {} +- {}'.format(loss_with_variance_test[0],loss_with_variance_test[1]))
print('test loss difference = {} +- {}'.format(loss_difference_variance_test[0],loss_difference_variance_test[1]))



loss = []
loss_difference = []
loss_test = []
loss_difference_test = []

for i in range(5):#repeat the experiemnt for 5 times
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
  X_0 = torch.tensor(X_0,requires_grad=False).float()
  X_1 = torch.tensor(X_1,requires_grad=False).float()
  y_0 = torch.tensor(y_0,requires_grad=False).float()
  y_1 = torch.tensor(y_1,requires_grad=False).float()
  model, l0,l1,l = penalty_method2(method, X_0, y_0, X_1, y_1,ite,lr,r,gamma,1e-6,seed = i)
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

#In this part, we can generate the metrics in Table 1 for Algorithm 2
#Please change the gamma, if you want to see the result for diffrent fairness level
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
for i in range(5):
  #loading data
  dictionary = last_hidden[i]
  X_0 = dictionary['X_0']
  X_1 = dictionary['X_1']
  X_test_0 = dictionary['X_test_0']
  X_test_1 = dictionary['X_test_1']
  y_0 = dictionary['y_0']
  y_1 = dictionary['y_1']
  y_test_0 = dictionary['y_test_0']
  y_test_1 = dictionary['y_test_1']
  w,b, l0,l1,l = solve_lin_constrained_opt(X_0, y_0, X_1, y_1, gamma, r ,method)
  #training loss

  #calculating test loss and test loss difference
  l0,l1,l = calculate_loss(w,b,X_test_1, y_test_1, X_test_0, y_test_0, method)

  
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

#In this part, we can generate the metrics in Table 1 for Algorithm 2
#Please change the gamma, if you want to see the result for diffrent fairness level
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
for i in range(5):
  #loading data
  dictionary = last_hidden[i]
  X_0 = dictionary['X_0']
  X_1 = dictionary['X_1']
  X_test_0 = dictionary['X_test_0']
  X_test_1 = dictionary['X_test_1']
  y_0 = dictionary['y_0']
  y_1 = dictionary['y_1']
  y_test_0 = dictionary['y_test_0']
  y_test_1 = dictionary['y_test_1']
  w,b, l0,l1,l = Algorithm2(X_0, y_0, X_1, y_1, gamma, r, method)
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

#In this part, we can generate the metrics in Table 1 for Algorithm 3
#Please change the gamma, if you want to see the result for diffrent fairness level
loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
for i in range(5):
  #loading data
  dictionary = last_hidden[i]
  X_0 = dictionary['X_0']
  X_1 = dictionary['X_1']
  X_test_0 = dictionary['X_test_0']
  X_test_1 = dictionary['X_test_1']
  y_0 = dictionary['y_0']
  y_1 = dictionary['y_1']
  y_test_0 = dictionary['y_test_0']
  y_test_1 = dictionary['y_test_1']
  w,b, l0,l1,l = Algorithm3(X_0, y_0, X_1, y_1, gamma, r, method)
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