import cvxpy
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
def calculate_loss(w,b,X0, Y0, X1, Y1, model='logistic'):
    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0).reshape(-1, )
    Y0 = Y0.reshape(-1, )
    Y1 = Y1.reshape(-1, )
    m, n = X.shape
    if model=='logistic':
        l1 = cp.sum(-cp.multiply(Y1, X1 @ w + b) + cp.logistic(X1 @ w + b)) / X1.shape[0]
        l1 = l1.value

        l0 = cp.sum(-cp.multiply(Y0, X0 @ w + b) + cp.logistic(X0 @ w + b)) / X0.shape[0]
        l0 = l0.value

        l = cp.sum(-cp.multiply(Y, X @ w + b) + cp.logistic(X @ w + b)) / m 
        l = l.value
    elif model=='linear':
        l1 = cp.sum_squares(Y1 - X1 @ w - b) / X1.shape[0]
        l1 = l1.value

        l0 = cp.sum_squares(Y0 - X0 @ w - b) / X0.shape[0]
        l0 = l0.value

        l = cp.sum_squares(Y - X @ w - b) / m 
        l = l.value
    return l0,l1,l

def solve_constrained_opt(X0, Y0, X1, Y1, eta, landa, model='logistic'):
    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0).reshape(-1, )
    Y0 = Y0.reshape(-1, )
    Y1 = Y1.reshape(-1, )
    m, n = X.shape
    np.random.seed(1)
    w = cp.Variable(n)
    b = cp.Variable(1)
    if model == 'logistic':

        loss = cp.sum(-cp.multiply(Y, X @ w + b) + cp.logistic(X @ w + b)) / m + eta * cp.norm(cvxpy.hstack([w, b]),
                                                                                              2) ** 2
        loss0 = cp.sum(-cp.multiply(Y0, X0 @ w + b) + cp.logistic(X0 @ w + b)) / X0.shape[0] +  eta * cp.norm(
            cvxpy.hstack([w, b]), 2) ** 2
        loss1 = cp.sum(-cp.multiply(Y1, X1 @ w + b) + cp.logistic(X1 @ w + b)) / X1.shape[0] +  eta * cp.norm(
            cvxpy.hstack([w, b]), 2) ** 2

        constraint = [loss0 <= landa]
        problem = cp.Problem(cp.Minimize(loss1), constraint)
        problem.solve(solver='ECOS_BB',verbose=False)

        l1 = cp.sum(-cp.multiply(Y1, X1 @ w + b) + cp.logistic(X1 @ w + b)) / X1.shape[0]
        l1 = l1.value

        l0 = cp.sum(-cp.multiply(Y0, X0 @ w + b) + cp.logistic(X0 @ w + b)) / X0.shape[0]
        l0 = l0.value

        l = cp.sum(-cp.multiply(Y, X @ w + b) + cp.logistic(X @ w + b)) / m + eta * cp.norm(cvxpy.hstack([w, b]), 2)**2
        l = l.value

    elif model=='linear':
        loss = cp.sum_squares(Y- X @ w - b) / m + eta * cp.norm(cvxpy.hstack([w, b]),2) ** 2
        loss0 = cp.sum_squares(Y0- X0 @ w - b) / X0.shape[0]  +  eta * cp.norm(cvxpy.hstack([w, b]), 2) ** 2
        loss1 = cp.sum_squares(Y1- X1 @ w - b)  / X1.shape[0] +  eta * cp.norm(cvxpy.hstack([w, b]), 2) ** 2

        constraint = [loss0 <= landa]
        problem = cp.Problem(cp.Minimize(loss1), constraint)
        problem.solve(verbose=False)

        l1 = cp.sum_squares(Y1- X1 @ w - b)  / X1.shape[0]
        l1 = l1.value

        l0 = cp.sum_squares(Y0- X0 @ w - b) / X0.shape[0]
        l0 = l0.value

        l = cp.sum_squares(Y- X @ w - b) / m + eta * cp.norm(cp.hstack([w, b]),2) ** 2
        l = l.value

    return w, b, l0, l1, l


def ELminimizer(X0, Y0, X1, Y1, gamma, eta, model='logistic'):
    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0).reshape(-1, )
    Y0 = Y0.reshape(-1, )
    Y1 = Y1.reshape(-1, )
    m, n = X.shape
    np.random.seed(1)
    if model == 'logistic':
        w = cp.Variable(n)
        b = cp.Variable(1)
        loss0 = cp.sum(-cp.multiply(Y0, X0 @ w + b) + cp.logistic(X0 @ w + b)) / X0.shape[0] + eta  * cp.norm(
            cp.hstack([w, b]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss0))
        problem.solve(solver='ECOS_BB')
        landa_start = loss0.value

        loss1 = cp.sum(-cp.multiply(Y1, X1 @ w + b) + cp.logistic(X1 @ w + b)) / X1.shape[0] + eta  * cp.norm(
            cvxpy.hstack([w, b]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss1))
        problem.solve(solver='ECOS_BB')
        loss0 = cp.sum(-cp.multiply(Y0, X0 @ w + b) + cp.logistic(X0 @ w + b)) / X0.shape[0] + eta  * cp.norm(
            cvxpy.hstack([w, b]), 2) ** 2
        landa_end = loss0.value

        while landa_end - landa_start > 0.01:
            landa_mid = (landa_end + landa_start) / 2
            w, b, l0, l1, L = solve_constrained_opt(X0, Y0, X1, Y1, eta, landa_mid, model)
            loss1 = cp.sum(-cp.multiply(Y1, X1 @ w + b) + cp.logistic(X1 @ w + b)) / X1.shape[0] + eta  * cp.norm(
                cvxpy.hstack([w, b]), 2) ** 2
            landa = loss1.value + gamma
            if landa >= landa_mid:
                landa_start = landa_mid
            else:
                landa_end = landa_mid

    elif model== 'linear':
        w0 = cp.Variable(n)
        b0 = cp.Variable(1)
        loss0 = cp.sum_squares(Y0- X0 @ w0 - b0)  / X0.shape[0] + eta  * cp.norm(cp.hstack([w0, b0]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss0))
        problem.solve(solver='ECOS_BB')
        landa_start = loss0.value
        w1 = cp.Variable(n)
        b1 = cp.Variable(1)
        loss1 = cp.sum_squares(Y1- X1 @ w1 - b1) / X1.shape[0] + eta  * cp.norm(cp.hstack([w1, b1]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss1))
        problem.solve(solver='ECOS_BB')
        loss0 = cp.sum_squares(Y0- X0 @ w1 - b1) / X0.shape[0] + eta  * cp.norm(cp.hstack([w1, b1]), 2) ** 2
        landa_end = loss0.value
        while landa_end - landa_start > 0.01:
            landa_mid = (landa_end + landa_start) / 2
            w, b, l0, l1, L = solve_constrained_opt(X0, Y0, X1, Y1, eta, landa_mid, model)
            loss1 = cp.sum_squares(Y1- X1 @ w - b)/ X1.shape[0] + eta  * cp.norm(cp.hstack([w, b]), 2) ** 2
            landa = loss1.value + gamma
            if landa >= landa_mid:
                landa_start = landa_mid
            else:
                landa_end = landa_mid


    return w, b, l0, l1, L


def Algorithm2(X0, Y0, X1, Y1, gamma, eta, model='logistic'):
    w_minus, b_minus, l0_minus, l1_minus, L_minus = ELminimizer(X1, Y1, X0, Y0, -gamma, eta, model)
    w_plus, b_plus, l0_plus, l1_plus, L_plus = ELminimizer(X1, Y1, X0, Y0, gamma, eta, model)
    if L_plus < L_minus:
        r = eta * cp.norm(cp.hstack([w_plus, b_plus]), 2) ** 2
        r = r.value
        return w_plus, b_plus, l0_plus, l1_plus, L_plus-r
    else:
        r = eta * cp.norm(cp.hstack([w_minus, b_minus]), 2) ** 2
        r = r.value
        return w_minus, b_minus, l0_minus, l1_minus, L_minus-r

def Algorithm3(X0, Y0, X1, Y1, gamma, eta, method='logistic'):
    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0).reshape(-1, )
    Y0 = Y0.reshape(-1, )
    Y1 = Y1.reshape(-1, )
    m, n = X.shape
    np.random.seed(1)
    if method == 'logistic':
        w0 = cp.Variable(n)
        b0 = cp.Variable(1)
        loss0 = cp.sum(-cp.multiply(Y0, X0 @ w0 + b0) + cp.logistic(X0 @ w0 + b0)) / X0.shape[0] + eta  * cp.norm(
            cp.hstack([w0, b0]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss0))
        problem.solve(solver='ECOS_BB')
        w = cp.Variable(n)
        b = cp.Variable(1)
        loss = cp.sum(-cp.multiply(Y, X @ w + b) + cp.logistic(X @ w + b)) / X1.shape[0] + eta  * cp.norm(
            cvxpy.hstack([w, b]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver='ECOS_BB')
        beta_s = 0
        beta_e = 1

        while beta_e - beta_s > 0.01:
            beta_m = (beta_e + beta_s) / 2
            w_beta = (1-beta_m)*w + (beta_m)*w0
            b_beta = (1 - beta_m) * b + (beta_m) * b0
            g = cp.sum(-cp.multiply(Y0, X0 @ w_beta + b_beta) + cp.logistic(X0 @ w_beta + b_beta)) / X0.shape[0] -cp.sum(-cp.multiply(Y1, X1 @ w_beta + b_beta) + cp.logistic(X1 @ w_beta + b_beta)) / X1.shape[0]
            g = g-gamma
            if g.value >= 0:
                beta_s = beta_m
            else:
                beta_e = beta_m
        w_beta = (1 - beta_m) * w + (beta_m) * w0
        b_beta = (1 - beta_m) * b + (beta_m) * b0
        loss0 = cp.sum(-cp.multiply(Y0, X0 @ w_beta + b_beta) + cp.logistic(X0 @ w_beta + b_beta)) / X0.shape[0]
        loss1 = cp.sum(-cp.multiply(Y1, X1 @ w_beta + b_beta) + cp.logistic(X1 @ w_beta + b_beta)) / X1.shape[0]
        loss = cp.sum(-cp.multiply(Y, X @ w_beta + b_beta) + cp.logistic(X @ w_beta + b_beta)) / X.shape[0]
    elif method== 'linear':
        w0 = cp.Variable(n)
        b0 = cp.Variable(1)

        loss0 = cp.sum_squares(Y0 - X0 @ w0 - b0) / X0.shape[0] + eta * cp.norm(cp.hstack([w0, b0]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss0))
        problem.solve()
        w = cp.Variable(n)
        b = cp.Variable(1)
        loss = cp.sum_squares(Y - X @ w - b) / X.shape[0]+ eta * cp.norm(cp.hstack([w, b]), 2) ** 2
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve()
        beta_s = 0
        beta_e = 1

        while beta_e - beta_s > 0.01:
            beta_m = (beta_e + beta_s) / 2
            w_beta = (1 - beta_m) * w + (beta_m) * w0
            b_beta = (1 - beta_m) * b + (beta_m) * b0
            g = cp.sum_squares(Y0 - X0 @ w_beta - b_beta) / X0.shape[0] - cp.sum_squares(Y1 - X1 @ w_beta - b_beta) / X1.shape[0]
            g = g - gamma
            if g.value >= 0:
                beta_s = beta_m
            else:
                beta_e = beta_m
        w_beta = (1 - beta_m) * w + (beta_m) * w0
        b_beta = (1 - beta_m) * b + (beta_m) * b0
        loss0 = cp.sum_squares(Y0 - X0 @ w_beta - b_beta) / X0.shape[0]
        loss1 = cp.sum_squares(Y1 - X1 @ w_beta - b_beta) / X1.shape[0]
        loss = cp.sum_squares(Y - X @ w_beta - b_beta) / X.shape[0]


    return w_beta, b_beta, loss0.value, loss1.value, loss.value

def solve_lin_constrained_opt(X0, Y0, X1, Y1, gamma, eta ,l='logistic'):
    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0).reshape(-1, )
    Y0 = Y0.reshape(-1, )
    Y1 = Y1.reshape(-1, )
    m, n = X.shape
    np.random.seed(1)
    if l == 'logistic':
        w = cp.Variable(n)
        b = cp.Variable(1)
        #y_hat = (1 + cp.exp(-X @ w - b))
        #loss0 = cp.sum(-cp.multiply(Y0, X0 @ w0 + b0) + cp.logistic(X0 @ w0 + b0)) / X0.shape[0] + eta  * cp.norm(
        #    cp.hstack([w0, b0]), 2) ** 2
        loss = cp.sum(-cp.multiply(Y, X @ w + b) + cp.logistic(X @ w + b)) / X.shape[0] + eta * cp.norm(cvxpy.hstack([w, b]),2) ** 2
        f0 = cp.sum((Y0-0.5)@(X0 @ w + b)) / X0.shape[0]
        f1 = cp.sum((Y1-0.5)@(X1 @ w + b)) / X1.shape[0]

        constraint = [f0 - f1 <= gamma, f1 - f0 <= gamma]
        problem = cp.Problem(cp.Minimize(loss), constraint)
        problem.solve(solver='ECOS_BB')

        l1 = cp.sum(-cp.multiply(Y1, X1 @ w + b) + cp.logistic(X1 @ w + b)) / X1.shape[0]
        l1 = l1.value

        l0 = cp.sum(-cp.multiply(Y0, X0 @ w + b) + cp.logistic(X0 @ w + b)) / X0.shape[0]
        l0 = l0.value

        l = cp.sum(-cp.multiply(Y, X @ w + b) + cp.logistic(X @ w + b)) / X.shape[0] #+ eta * cp.norm(cp.hstack([w, b]), 2) ** 2
        l = l.value
    elif l=='linear':
        w = cp.Variable(n)
        b = cp.Variable(1)
        loss = cp.sum_squares(Y- X @ w - b)  / X.shape[0] + eta * cp.norm(cp.hstack([w, b]), 2) ** 2
        f0 = cp.sum(Y0-(X0 @ w + b)) / X0.shape[0]
        f1 = cp.sum(Y1 - X1 @ w - b) / X1.shape[0]

        constraint = [f0 - f1 <= gamma, f1 - f0 <= gamma]
        problem = cp.Problem(cp.Minimize(loss), constraint)
        problem.solve(solver='ECOS_BB')

        l1 = cp.sum_squares(Y1- X1 @ w - b)   / X1.shape[0]
        l1 = l1.value

        l0 = cp.sum_squares(Y0 - X0 @ w - b)  / X0.shape[0]
        l0 = l0.value

        l = cp.sum_squares(Y - X @ w - b)/X.shape[0]  #+ eta * cp.norm(cp.hstack([w, b]), 2) ** 2
        l = l.value

    return w, b, l0, l1, l

