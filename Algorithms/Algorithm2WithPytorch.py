from torch import nn
import torch

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


def train(X,y,num_itr, lr, r, seed=0):
    torch.manual_seed(seed)
    model = NN2(X.shape[1])
    LL = nn.MSELoss()
    t = torch.tensor(0.1, requires_grad=False).float()
    count = 0
    while count < num_itr:
        count += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        reg = 0
        for p in model.parameters():
            reg += r * torch.norm(p) ** 2
        loss = LL(model.forward(X), y.reshape(-1, 1)) + reg
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return model, LL(model.forward(X), y.unsqueeze(1)).detach()

def penalty_method(X_0, y_0, X_1, y_1, num_itr ,lr,r,gamma,epsilon=0.01,seed = 0):
    torch.manual_seed(seed)
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
        loss = LL(model.forward(X_0),y_0.reshape(-1,1)) + t*torch.nn.functional.relu(abs(LL(model.forward(X_1),y_1.unsqueeze(1))-gamma))**2+reg
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
    return model, LL(model.forward(X_0),y_0.unsqueeze(1)).detach()


def ELminimizer_PT(X0, Y0, X1, Y1, num_itr, lr, r, gamma,seed=0):
    torch.manual_seed(seed)
    model = NN2(X0.shape[1])
    LL = nn.MSELoss()
    y = torch.vstack([Y0.unsqueeze(1), Y1.unsqueeze(1)])
    X = torch.vstack([X0, X1])
    model0, landa_start = train(X0,Y0,num_itr,lr,r)
    model1, l1 = train(X1, Y1, num_itr, lr, r)
    landa_end = LL(model1.forward(X0), Y0.unsqueeze(1)).detach()
    while landa_end - landa_start > 0.01:
        landa_mid = (landa_end + landa_start) / 2
        model, loss1 = penalty_method(X0, Y0, X1, Y1, num_itr, lr, r, landa_mid)
        landa = loss1 + gamma
        if landa >= landa_mid:
            landa_start = landa_mid
        else:
            landa_end = landa_mid

    return model, LL(model.forward(X0),Y0.unsqueeze(1)).detach(), LL(model.forward(X1),Y1.unsqueeze(1)).detach(),LL(model.forward(X),y.reshape(-1,1)).detach()

def Algorithm2_PT(X0, Y0, X1, Y1, num_itr, lr, r, gamma):
    model_minus, l0_minus, l1_minus, L_minus = ELminimizer_PT(X1, Y1, X0, Y0, num_itr, lr, r, gamma)
    model_plus, l0_plus, l1_plus, L_plus = ELminimizer_PT(X1, Y1, X0, Y0, num_itr, lr, r, gamma)
    if L_plus < L_minus:
        return model_plus, l0_plus, l1_plus, L_plus
    else:
        return model_minus, l0_minus, l1_minus, L_minus