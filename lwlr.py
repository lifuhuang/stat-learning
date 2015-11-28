# -*- coding: utf-8 -*-
"""
Locally-weighted logistic regression 
"""
import numpy as np
import matplotlib.pyplot as plt

X_train = np.fromfile(r'E:\data\x.dat', sep = ' ').reshape((-1, 2))
y_train = np.fromfile(r'E:\data\y.dat', sep = ' ').reshape((-1, 1))

m = np.size(X_train, 0)
n = np.size(X_train, 1)

def predict(x):
    theta = np.zeros((n, 1))
    tau = 1
    lambda_ = 1e-4
    eps = 1e-6
    
    w = np.exp(-np.sum((X_train - x.T) ** 2, axis = 1) / (2 * tau))[:, np.newaxis]
    g = np.ones((n, 1))
    #i = 0
    while (g.T @ g)[0, 0] > eps:
        #i += 1
        # print ('theta#', i, '=', theta)
        h = 1 / (1 + np.exp(-(X_train @ theta)))
        D = np.diag((-w * h * (1 - h))[:, 0])
        H = X_train.T @ D @ X_train - np.identity(n) * lambda_
        g = X_train.T @ (w * (y_train - h)) - lambda_ * theta
        theta = theta - np.linalg.pinv(H) @ g
        
    result = 1 / (1 + np.exp(-(x.T @ theta)))
    print(theta)
    return True if result > 0.5 else False
    
if __name__ == '__main__':    
    x0, x1 = input('Please input x: ').split()
    x = np.array([[float(x0)], [float(x1)]])    
    predict(x)