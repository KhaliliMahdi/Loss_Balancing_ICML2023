import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment',type=int,default=1, choices = [1,2])

args = parser.parse_args()

import warnings
warnings.simplefilter('ignore')

import numpy as np
import torch
from law_data import *#law dataset used in the first experiment
from Adult_data import *#Adult dataset used in the first experiment
from Algorithms import *#Algorithm 1, Algorithm 2, and Algorithm 3 implemented in Convex_solver
from Baseline import penalty_method, fair_batch #This is the baseline
from Non_linear import train, penalty_method2, fair_batch2  #This is the baseline

Table = args.experiment 
if Table==1:
  dataset = law_data
  method = 'linear'
  LL = torch.nn.MSELoss()
  Gamma = [0.025,0.05,0.1,0.15,0.2]
elif Table==2:
  dataset = Adult_dataset
  method = 'logistic'
  LL = torch.nn.BCELoss()
  Gamma = [0.02,0.04,0.06,0.08,0.1]

ite = 1000 # number of iterations
lr = 0.001 # learning rate
r = 0.002 # regularizer parameter

#training a DNN without fairness constraint
last_hidden = []
for i in range(5):#repeat the experiemnt for 5 times
  X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
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


print("==running FairBatch==")
loss_with_variance = []
loss_difference_variance = []
loss_with_variance_test = []
loss_difference_variance_test = []
for gamma in Gamma:
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
    model, l0,l1,l = fair_batch2(method, X_0, y_0, X_1, y_1, ite, lr, r, 0.005, gamma, i,1e-6)
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
  
  loss = torch.tensor(loss)
  loss_difference = torch.tensor(loss_difference)
  loss_test = torch.tensor(loss_test)
  loss_difference_test = torch.tensor(loss_difference_test)
  loss_with_variance.append((torch.mean(loss),torch.std(loss)))
  loss_difference_variance.append((torch.mean(loss_difference),torch.std(loss_difference)))
  loss_with_variance_test.append((torch.mean(loss_test),torch.std(loss_test)))
  loss_difference_variance_test.append((torch.mean(loss_difference_test),torch.std(loss_difference_test)))



print("==running PM==")

loss_with_variance0 = []
loss_difference_variance0 = []
loss_with_variance_test0 = []
loss_difference_variance_test0 = []


for gamma in Gamma:
  loss = []
  loss_difference = []
  loss_test = []
  loss_difference_test = []
  for i in range(5):
    #load data
    X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(i)
    X_0 = torch.tensor(X_0,requires_grad=False).float()
    X_1 = torch.tensor(X_1,requires_grad=False).float()
    y_0 = torch.tensor(y_0,requires_grad=False).float()
    y_1 = torch.tensor(y_1,requires_grad=False).float()
    #training data using the penalty method
    model, l0,l1,l = penalty_method2(method, X_0, y_0, X_1, y_1,ite,lr,r,gamma,1e-6,seed=i)
    loss.append(l)
    loss_difference.append(abs(l0-l1))
    #preparing the test dataset
    X_test_0 = torch.tensor(X_test_0,requires_grad=False).float()
    X_test_1 = torch.tensor(X_test_1,requires_grad=False).float()
    y_test_0 = torch.tensor(y_test_0,requires_grad=False).float()
    y_test_1 = torch.tensor(y_test_1,requires_grad=False).float()
    y_test = torch.vstack([y_test_0.unsqueeze(1),y_test_1.unsqueeze(1)])
    X_test = torch.vstack([X_test_0,X_test_1])
    #calculating test loss and test loss difference
    loss_difference_test.append(abs(LL(model.forward(X_test_0),y_test_0.unsqueeze(1))- LL(model.forward(X_test_1),y_test_1.unsqueeze(1))).detach())
    loss_test.append(LL(model.forward(X_test),y_test.reshape(-1,1)).detach())
  
  loss_with_variance0.append((np.mean(loss),np.std(loss)))
  loss_difference_variance0.append((np.mean(loss_difference),np.std(loss_difference)))
  loss_with_variance_test0.append((np.mean(loss_test),np.std(loss_test)))
  loss_difference_variance_test0.append((np.mean(loss_difference_test),np.std(loss_difference_test)))

print("==running  LineRe==")



loss_with_variance1 = []
loss_difference_variance1 = []
loss_with_variance_test1 = []
loss_difference_variance_test1 = []



for gamma in Gamma:
  loss = []
  loss_difference = []
  loss_test = []
  loss_difference_test = []
  for i in range(5):
    #load data
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
    loss.append(l)
    loss_difference.append(abs(l0-l1))
    #calculating  test loss and loss difference
    l0,l1,l = calculate_loss(w,b,X_test_1, y_test_1, X_test_0, y_test_0, method)
    loss_difference_test.append(abs(l0-l1))
    loss_test.append(l)
  
  loss_with_variance1.append((np.mean(loss),np.std(loss)))
  loss_difference_variance1.append((np.mean(loss_difference),np.std(loss_difference)))
  loss_with_variance_test1.append((np.mean(loss_test),np.std(loss_test)))
  loss_difference_variance_test1.append((np.mean(loss_difference_test),np.std(loss_difference_test)))

print("==running Alg2==")


loss_with_variance2 = []
loss_difference_variance2 = []
loss_with_variance_test2 = []
loss_difference_variance_test2 = []


for gamma in Gamma:
  loss = []
  loss_difference = []
  loss_test = []
  loss_difference_test = []
  for i in range(5):
    #load data
    dictionary = last_hidden[i]
    X_0 = dictionary['X_0']
    X_1 = dictionary['X_1']
    X_test_0 = dictionary['X_test_0']
    X_test_1 = dictionary['X_test_1']
    y_0 = dictionary['y_0']
    y_1 = dictionary['y_1']
    y_test_0 = dictionary['y_test_0']
    y_test_1 = dictionary['y_test_1']
    #model training using Algorithm 2
    w,b, l0,l1,l = Algorithm2(X_0, y_0, X_1, y_1, gamma, r, method)
    loss.append(l)
    loss_difference.append(abs(l0-l1))
    #calculating  test loss and loss difference
    l0,l1,l = calculate_loss(w,b,X_test_1, y_test_1, X_test_0, y_test_0, method)
    loss_difference_test.append(abs(l0-l1))
    loss_test.append(l)
  
  loss_with_variance2.append((np.mean(loss),np.std(loss)))
  loss_difference_variance2.append((np.mean(loss_difference),np.std(loss_difference)))
  loss_with_variance_test2.append((np.mean(loss_test),np.std(loss_test)))
  loss_difference_variance_test2.append((np.mean(loss_difference_test),np.std(loss_difference_test)))

print('==running Alg3==')
loss_with_variance3 = []
loss_difference_variance3 = []
loss_with_variance_test3 = []
loss_difference_variance_test3 = []

for gamma in Gamma:
  loss = []
  loss_difference = []
  loss_test = []
  loss_difference_test = []
  for i in range(5):
    #load the data 
    dictionary = last_hidden[i]
    X_0 = dictionary['X_0']
    X_1 = dictionary['X_1']
    X_test_0 = dictionary['X_test_0']
    X_test_1 = dictionary['X_test_1']
    y_0 = dictionary['y_0']
    y_1 = dictionary['y_1']
    y_test_0 = dictionary['y_test_0']
    y_test_1 = dictionary['y_test_1']
    #training the model using Algorithm 3
    w,b, l0,l1,l = Algorithm3(X_0, y_0, X_1, y_1, gamma, r, method)
    loss.append(l)
    loss_difference.append(abs(l0-l1))
    #calculating test loss and loss difference
    l0,l1,l = calculate_loss(w,b,X_test_0, y_test_0, X_test_1, y_test_1, method)
    loss_difference_test.append(abs(l0-l1))
    loss_test.append(l)
  
  loss_with_variance3.append((np.mean(loss),np.std(loss)))
  loss_difference_variance3.append((np.mean(loss_difference),np.std(loss_difference)))
  loss_with_variance_test3.append((np.mean(loss_test),np.std(loss_test)))
  loss_difference_variance_test3.append((np.mean(loss_difference_test),np.std(loss_difference_test)))




x = [a[0] for a in loss_with_variance_test]
y = [a[0] for a in loss_difference_variance_test]

x0 = [a[0] for a in loss_with_variance_test0]
y0 = [a[0] for a in loss_difference_variance_test0]

x1 = [a[0] for a in loss_with_variance_test1]
y1 = [a[0] for a in loss_difference_variance_test1]

x2 = [a[0] for a in loss_with_variance_test2]
y2 = [a[0] for a in loss_difference_variance_test2]

x3 = [a[0] for a in loss_with_variance_test3]
y3 = [a[0] for a in loss_difference_variance_test3]



import matplotlib.pyplot as plt
import matplotlib
from pylab import *

font = {'family' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(5,4))
p1, = plt.plot(y0,x0,'r*-',label=r'PM',linewidth=2)
p2, = plt.plot(y1,x1,'c>:',label=r'LinRe',linewidth=2)
p3, = plt.plot(y,x,'y^-',label=r'FairBatch',linewidth=2)
p4, = plt.plot(y2,x2,'bx--',label=r'Alg2',linewidth=2)
p5, = plt.plot(y3,x3,'ko-.',label=r'Alg3',linewidth=2)
l1 = plt.legend([p1,p2,p3,p4,p5],['PM','LinRe']+['FairBatch','Alg2','Alg3'], loc='upper right')


plt.title('Overall loss v.s. loss difference')
plt.xlabel(r'test loss difference $|\hat{L}_0-\hat{L}_1|$')
plt.ylabel(r'test loss $\hat{L}$')
plt.grid()
fig.savefig('MSE_nonlinear.eps',bbox_inches ="tight")
plt.show()


