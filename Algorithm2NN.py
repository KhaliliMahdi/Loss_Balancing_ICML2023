import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--experiment', type=int, default=1, choices=[1, 2])

args = parser.parse_args()

import warnings

warnings.simplefilter('ignore')

import numpy as np
import torch
from law_data import *  # law dataset used in the first experiment
from Adult_data import *  # Adult dataset used in the first experiment
from Algorithm2WithPytorch import train, penalty_method, Algorithm2_PT

Table = args.experiment
if Table == 1:
    dataset = law_data
    method = 'linear'
    LL = torch.nn.MSELoss()
elif Table == 2:
    dataset = Adult_dataset
    method = 'logistic'
    LL = torch.nn.BCELoss()

# Please change the gamma, if you want to see the result for diffrent fairness level
gamma = args.gamma
ite = 10000  # number of iterations
lr = 0.0002  # learning rate
r = 0.002  # regularizer parameter

loss = []
loss_difference = []
loss_test = []
loss_difference_test = []
for i in range(5):
    # loading data
    X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = dataset(seed=i)
    print(X_0)
    X_0 = torch.tensor(X_0, requires_grad=False).float()
    X_1 = torch.tensor(X_1, requires_grad=False).float()
    y_0 = torch.tensor(y_0, requires_grad=False).float()
    y_1 = torch.tensor(y_1, requires_grad=False).float()
    X_test_0 = torch.tensor(X_test_0, requires_grad=False).float()
    X_test_1 = torch.tensor(X_test_1, requires_grad=False).float()
    y_test_0 = torch.tensor(y_test_0, requires_grad=False).float()
    y_test_1 = torch.tensor(y_test_1, requires_grad=False).float()
    y_test = torch.vstack([y_test_0.unsqueeze(1), y_test_1.unsqueeze(1)])
    X_test = torch.vstack([X_test_0, X_test_1])
    model, l0, l1, l = Algorithm2_PT(X_0, y_0, X_1, y_1, ite, lr, r, gamma)
    # training loss
    loss.append(l)
    loss_difference.append(abs(l0 - l1))
    # calculating test loss and test loss difference
    l0, l1, l =  LL(model.forward(X_test_0),y_test_0.unsqueeze(1)).detach(), LL(model.forward(X_test_1),y_test_1.unsqueeze(1)).detach(),LL(model.forward(X_test),y_test.reshape(-1,1)).detach()

    loss_difference_test.append(abs(l0 - l1))
    loss_test.append(l)

loss_with_variance = (np.mean(loss), np.std(loss))
loss_difference_variance = (np.mean(loss_difference), np.std(loss_difference))
loss_with_variance_test = (np.mean(loss_test), np.std(loss_test))
loss_difference_variance_test = (np.mean(loss_difference_test), np.std(loss_difference_test))

print("Table {}".format(Table), "Algorithm 2")
print("gamma = {}".format(gamma))
print('training loss = {} +- {}'.format(loss_with_variance[0], loss_with_variance[1]))
print('training loss difference = {} +- {}'.format(loss_difference_variance[0], loss_difference_variance[1]))
print('test loss = {} +- {}'.format(loss_with_variance_test[0], loss_with_variance_test[1]))
print('test loss difference = {} +- {}'.format(loss_difference_variance_test[0], loss_difference_variance_test[1]))
