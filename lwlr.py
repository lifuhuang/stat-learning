# -*- coding: utf-8 -*-
"""
Locally-weighted logistic regression 
"""
import numpy as np
import matplotlib.pyplot as plt

def predict(x, X_train, Y_train):    
    """Returns the hypothesis of x given training set X_train and y_train.
    """
    m = np.size(X_train, 0)
    n = np.size(X_train, 1)
    theta = np.zeros((n, 1))
    tau = 1
    lambda_ = 1e-4
    eps = 1e-6
    w = np.exp(-np.sum((X_train - x.T) ** 2, axis = 1) / (2 * tau ** 2))[:, np.newaxis]
    g = np.ones((n, 1))
    while g.T @ g > eps:
        h = 1 / (1 + np.exp(-(X_train @ theta)))
        D = np.diag((-w * h * (1 - h))[:, 0])
        H = X_train.T @ D @ X_train - np.identity(n) * lambda_
        g = X_train.T @ (w * (y_train - h)) - lambda_ * theta
        theta = theta - np.linalg.pinv(H) @ g
    result = 1 / (1 + np.exp(-(x.T @ theta)))
    return True if result > 0.5 else False

def get_precision(X_train, y_train):
    m = np.size(X_train, 0)
    n = np.size(X_train, 1)
    correct = sum(1 for i in range(m) if predict(X_train[i, :, np.newaxis], X_train, y_train) == bool(y_train[i]))
    return correct / m

if __name__ == '__main__':        
    X_train = np.fromfile(r'x.dat', sep = ' ').reshape((-1, 2))
    y_train = np.fromfile(r'y.dat', sep = ' ').reshape((-1, 1))
    X_train = np.hstack((np.ones((np.size(X_train, 0), 1)), X_train))
    print("Precision on training set:", get_precision(X_train, y_train))