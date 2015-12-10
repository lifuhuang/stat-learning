# -*- coding: utf-8 -*-
"""
Support Vector Machine implemented with convex optimization method
"""

import numpy as np
import cvxopt as co
import matplotlib.pyplot as plt
import functools
import itertools

def gaussian_kernel(x, z, tau):
    return np.exp(- np.linalg.norm(x - z) ** 2 / tau ** 2)

def linear_kernel(x, z):
    return x.T @ z
    
def train_svm_with_cvxopt(X, y, C, K = linear_kernel):
    m = np.size(X, 0)
    n = np.size(X, 1)
    P = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            P[i, j] = K(X[i, :, np.newaxis], X[j, :, np.newaxis])
    q = -np.ones(y.shape)
    G = np.vstack((np.identity(m), -np.identity(m)))
    h = np.vstack((C * np.ones((m, 1)), np.zeros((m, 1))))
    A = y.T
    b = np.ones((1, 1))    
    res = co.solvers.qp(co.matrix(P), co.matrix(q), co.matrix(G), co.matrix(h), co.matrix(A), co.matrix(b))    
    alpha = np.array(res['x'])
    w = np.sum(alpha * y * X, axis = 0)[:, np.newaxis]
    b = - (w.T @ X[max(filter(lambda i: y[i] == -1, range(m)), key = lambda i: w.T @ X[i, :, np.newaxis]), :, np.newaxis] + 
           w.T @ X[min(filter(lambda i: y[i] == 1, range(m)), key = lambda i: w.T @ X[i, :, np.newaxis]), :, np.newaxis]) / 2
           
    return (alpha, w, b)

def predict(x, X_train, y_train, alpha, b, K = linear_kernel):
    m = np.size(X_train, 0)
    n = np.size(X_train, 1)
    ans = b.copy()  
    for i in range(m):
        ans += alpha[i] * y_train[i] * K(X_train[i, :, np.newaxis], x)
    return float(ans)
    
if __name__ == '__main__':    
    X_train = np.fromfile('x.dat', sep = ' ').reshape(-1, 2)
    y_train = np.array(list(map(lambda x: x * 2 - 1, np.fromfile('y.dat', sep = ' ')))).reshape(-1, 1)
    
    m = np.size(X_train, 0)
    n = np.size(X_train, 1)
    kernel = functools.partial(gaussian_kernel, tau = 0.25)
    alpha, w, b = train_svm_with_cvxopt(X_train, y_train, 0.5, kernel)
    print('w:', w, 'b:', b, sep = '\n')
    cnt = 0
    for i in range(m):
        result = 1 if (predict(X_train[i, :, np.newaxis], X_train, y_train, alpha, b, kernel) >= 0) else -1
        if result == y_train[i]:
            cnt += 1
    print('Precision:', cnt / m)
    
    for i in range(m):
        if y_train[i] == -1:
            plt.plot(X_train[i, 0], X_train[i, 1], 'ob')
        else:
            plt.plot(X_train[i, 0], X_train[i, 1], 'xr')
    
    sz = 50
    X, Y = np.meshgrid(np.linspace(-1, 1, sz), np.linspace(-1, 1, sz))
    Z = np.empty((sz, sz))
    for i in range(sz):
        for j in range(sz):
            Z[i, j] = predict(np.array([[X[i][j]], [Y[i][j]]]), X_train, y_train, alpha, b, kernel)
    
    plt.contour(X, Y, Z, levels = [0])
    plt.show()
    